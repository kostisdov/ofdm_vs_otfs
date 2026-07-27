"""Deployable-technique comparison over a realistic doubly dispersive channel:

  (1) CP-OFDM, single-tap FD equalization        -- standard OFDM (floors)
  (2) ZP-OFDM, block ICI-aware banded equalizer   -- proposed
  (3) OTFS, banded delay-Doppler LMMSE            -- delay-Doppler competitor

All three use the SAME 3GPP-TDL-C channel, LDPC code, and interleaver, with
bias-corrected soft demapping. Perfect CSI (the CP-OFDM floor is set by ICI, not
by estimation error, so this is its best case).
"""

import argparse
import numpy as np

from channel import JakesTDLChannel
from cpofdm import CPOFDM
from zpofdm import ZPOFDM
from zakotfs import OTFS
from ldpc import LDPC, qpsk_soft_llr_persymbol, qpsk_mod


def run_point(sim_fn, wf, channel, coder, snr_db, n_frames, rng):
    n_coded = wf.frame_num_qam() * 2
    perm = np.random.default_rng(7).permutation(n_coded)
    inv = np.argsort(perm)
    berr = btot = 0
    for _ in range(n_frames):
        channel.new_realization()
        info = rng.integers(0, 2, coder.k)
        coded = coder.encode(info)[:n_coded]
        syms = qpsk_mod(coded[perm].astype(int))
        z, gamma = sim_fn(syms, channel, snr_db, rng)
        llr = qpsk_soft_llr_persymbol(z, gamma)[inv]
        info_hat = coder.info_from_codeword(coder.decode(llr))
        berr += int(np.sum(info_hat != info)); btot += coder.k
    return berr / btot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=64)
    ap.add_argument('--N', type=int, default=8)
    ap.add_argument('--frames', type=int, default=250)
    ap.add_argument('--snr', type=float, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 8])
    ap.add_argument('--eps_max', type=float, default=0.3)
    ap.add_argument('--scs_khz', type=float, default=30.0)
    ap.add_argument('--fc_ghz', type=float, default=3.5)
    ap.add_argument('--qguard', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='techniques.png')
    args = ap.parse_args()

    M, N = args.M, args.N
    fs = M * args.scs_khz * 1e3
    ts = 1.0 / fs
    fd = args.eps_max * args.scs_khz * 1e3
    delays_ns = [0, 200, 800, 1600]
    powers_db = [0, -2, -6, -10]
    channel = JakesTDLChannel(delays_ns, powers_db, fd, fs, seed=args.seed)
    v_kmh = fd * 3e8 / (args.fc_ghz * 1e9) * 3.6
    Lg = channel.l_max + 1
    Ld = channel.l_max
    Kd = min(N // 2, int(np.ceil(args.eps_max * N)) + 1)

    coder = LDPC(n=2 * M * N, dv=3, dc=6, seed=2, maxiter=50)
    cpofdm = CPOFDM(M, N, Lg, ts)
    zpofdm = ZPOFDM(M, N, Lg, ts, Q=None, q_guard=args.qguard, equalizer='mmse')
    otfs = OTFS(M, N, Lg, ts)
    Q = zpofdm._band_Q(channel)

    print(f"Channel: Jakes/TDL-C {delays_ns} ns, fd={fd:.0f} Hz (v={v_kmh:.0f} km/h "
          f"@{args.fc_ghz} GHz), P={channel.P} taps, l_max={channel.l_max}")
    print(f"M={M} N={N} eps_max={args.eps_max} | ZP band Q={Q} | OTFS DD (Ld={Ld},Kd={Kd}) "
          f"| LDPC n={coder.n} k={coder.k} | frames={args.frames}")

    methods = [
        ('CP-OFDM single-tap',   lambda s, c, snr, r: cpofdm.simulate(s, c, snr, r)),
        ('ZP-OFDM block ICI-aware', lambda s, c, snr, r: zpofdm.simulate(s, c, snr, r)),
        ('OTFS banded DD',       lambda s, c, snr, r: otfs.simulate(s, c, snr, r, dd_band=(Ld, Kd))),
    ]
    wf_of = {'CP-OFDM single-tap': cpofdm, 'ZP-OFDM block ICI-aware': zpofdm,
             'OTFS banded DD': otfs}

    print(f"{'SNR':>5} | " + " | ".join(f"{lab:>22}" for lab, _ in methods))
    results = {lab: [] for lab, _ in methods}
    for snr in args.snr:
        row = []
        for lab, fn in methods:
            rng = np.random.default_rng(1000 + int(snr))
            b = run_point(fn, wf_of[lab], channel, coder, snr, args.frames, rng)
            results[lab].append(b); row.append(b)
        print(f"{snr:5.1f} | " + " | ".join(f"{b:22.2e}" for b in row))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        res_floor = 1.0 / (args.frames * M * N * 2)
        snr_arr = np.array(args.snr, float)
        styles = {'CP-OFDM single-tap': 'o--', 'ZP-OFDM block ICI-aware': 's-',
                  'OTFS banded DD': '^-'}
        labels = {'CP-OFDM single-tap': 'CP-OFDM, single-tap (standard)',
                  'ZP-OFDM block ICI-aware': f'ZP-OFDM, block ICI-aware (proposed, Q={Q})',
                  'OTFS banded DD': f'OTFS, banded DD (Ld={Ld},Kd={Kd})'}
        for lab, _ in methods:
            y = np.array(results[lab], float)
            y[y <= 0] = np.nan
            plt.semilogy(snr_arr, y, styles[lab], label=labels[lab], lw=1.7, ms=6)
        plt.axhline(res_floor, color='gray', ls=':', lw=0.8,
                    label=f'MC resolution ($\\approx${res_floor:.0e})')
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('SNR [dB]'); plt.ylabel('Coded BER')
        plt.title(f'Waveform comparison ($\\varepsilon_{{max}}$={args.eps_max}, '
                  f'QPSK, LDPC r=0.5, TDL-C)')
        plt.legend(); plt.tight_layout()
        plt.savefig(args.out, dpi=130)
        print(f"\nSaved figure to {args.out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()
