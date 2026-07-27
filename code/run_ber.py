"""Domain-specific, equal-footing coded-BER comparison over a doubly-selective
channel:

  (1) ZP-OFDM, single-tap freq-domain equalizer     -- floor reference
  (2) ZP-OFDM, ICI-aware banded freq-domain equalizer (adaptive Q)
  (3) OTFS, banded delay-Doppler equalizer           -- domain-specific match to (2)
  (4) OTFS, full delay-Doppler LMMSE                  -- genie upper bound

(2) and (3) are the fair comparison: each waveform equalized in its natural
domain with a reduced-complexity banded equalizer. (1) and (4) bound the range.
All methods use the SAME convolutional code, interleaver, and per-frame channel.
"""

import argparse
import numpy as np

from channel import DoublySelectiveChannel, EstimatedChannel, JakesTDLChannel
from coding import ConvCode
from qam import QAM
from zpofdm import ZPOFDM
from zakotfs import OTFS
from ldpc import LDPC, qpsk_soft_llr_persymbol, qpsk_mod


def run_point(sim_fn, wf, channel, coder, qam, snr_db, n_frames, rng, soft, pilot_snr=None):
    bit_err, bit_tot = 0, 0
    n_coded = wf.frame_num_qam() * qam.bits_per_symbol
    info_len = coder.k if soft else coder.info_len_for(n_coded)
    perm = np.random.default_rng(7).permutation(n_coded)   # shared interleaver
    inv = np.argsort(perm)
    snr_lin = 10.0 ** (snr_db / 10.0)
    for _ in range(n_frames):
        channel.new_realization()
        est = (EstimatedChannel(channel, wf.N, wf.sym_len(), pilot_snr, rng)
               if pilot_snr is not None else None)      # imperfect CSI
        info = rng.integers(0, 2, info_len)
        coded = coder.encode(info)
        coded = (np.concatenate([coded, np.zeros(n_coded - len(coded), int)])
                 if len(coded) < n_coded else coded[:n_coded])
        tx_bits = coded[perm].astype(int)
        syms = qpsk_mod(tx_bits) if soft else qam.modulate(tx_bits)
        z, gamma = sim_fn(syms, channel, snr_db, rng, est)
        if soft:                                   # LDPC bias-corrected soft-decision
            llr = qpsk_soft_llr_persymbol(z, gamma)[inv]
            info_hat = coder.info_from_codeword(coder.decode(llr))
        else:                                      # CC hard-decision Viterbi
            rx_bits = qam.demodulate_hard(z)[inv]
            info_hat = coder.decode(rx_bits)[:info_len]
        bit_err += int(np.sum(info_hat != info))
        bit_tot += info_len
    return bit_err / max(bit_tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=16)
    ap.add_argument('--N', type=int, default=16)
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--qam', type=int, default=4)
    ap.add_argument('--snr', type=float, nargs='+', default=[0, 4, 8, 12, 16])
    ap.add_argument('--eps_max', type=float, default=0.3)
    ap.add_argument('--qguard', type=int, default=1)
    ap.add_argument('--code', choices=['cc', 'ldpc'], default='cc')
    ap.add_argument('--channel', choices=['toy', 'jakes'], default='toy',
                    help='toy 3-tap on-grid, or realistic Jakes/3GPP-TDL-C')
    ap.add_argument('--scs_khz', type=float, default=30.0)  # jakes only
    ap.add_argument('--fc_ghz', type=float, default=3.5)    # jakes only (for v report)
    ap.add_argument('--pilot_snr', type=float, default=None,
                    help='pilot SNR [dB] for imperfect CSI; omit for perfect CSI')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='ber.png')
    args = ap.parse_args()

    M, N = args.M, args.N
    if args.channel == 'jakes':
        # realistic 3GPP-TDL-C-like profile: fractional delays, classical Doppler
        fs = M * args.scs_khz * 1e3
        fd = args.eps_max * args.scs_khz * 1e3                 # eps = fd/SCS
        delays_ns = [0, 200, 800, 1600]
        powers_db = [0, -2, -6, -10]
        channel = JakesTDLChannel(delays_ns, powers_db, fd, fs, seed=args.seed)
        v_kmh = fd * 3e8 / (args.fc_ghz * 1e9) * 3.6
        Lg = channel.l_max + 1
        Ld = channel.l_max
        print(f"Channel: Jakes/TDL-C fractional delays {delays_ns} ns, fd={fd:.0f} Hz "
              f"(v={v_kmh:.0f} km/h @{args.fc_ghz} GHz), P={channel.P} taps, l_max={channel.l_max}")
    else:
        fs = 1e6
        delays = [0, 1, 3]
        gains_db = [0, -3, -8]
        dopplers = [0.0, args.eps_max * fs / M, -0.7 * args.eps_max * fs / M]
        Lg = max(delays) + 1
        Ld = max(delays)
        channel = DoublySelectiveChannel(delays, gains_db, dopplers, fs, seed=args.seed)
    ts = 1.0 / fs
    qam = QAM(args.qam)
    soft = (args.code == 'ldpc')
    if soft:
        assert args.qam == 4, "LDPC soft path assumes QPSK"
        coder = LDPC(n=M * N * qam.bits_per_symbol, dv=3, dc=6, seed=2, maxiter=50)
        print(f"Code: LDPC n={coder.n} k={coder.k} rate={coder.rate:.3f} (soft BP)")
    else:
        coder = ConvCode()
        print("Code: K=7 convolutional r=1/2 (hard Viterbi)")

    ofdm_1tap = ZPOFDM(M, N, Lg, ts, Q=0, equalizer='mmse')
    ofdm_band = ZPOFDM(M, N, Lg, ts, Q=None, q_guard=args.qguard, equalizer='mmse')
    otfs = OTFS(M, N, Lg, ts)

    Q_star = ofdm_band._band_Q(channel)                # freq-domain ICI band
    Kd = min(N // 2, int(np.ceil(args.eps_max * N)) + 1)   # DD Doppler band
    print(f"Scenario: M={M} N={N} eps_max={args.eps_max} | OFDM band Q*={Q_star} | "
          f"OTFS DD band (Ld={Ld},Kd={Kd}) | frames={args.frames}")

    methods = [
        ('OFDM 1-tap',      lambda s, c, snr, r, e: ofdm_1tap.simulate(s, c, snr, r, est_channel=e)),
        ('OFDM band-freq',  lambda s, c, snr, r, e: ofdm_band.simulate(s, c, snr, r, est_channel=e)),
        ('OTFS band-DD',    lambda s, c, snr, r, e: otfs.simulate(s, c, snr, r, dd_band=(Ld, Kd), est_channel=e)),
        ('OTFS full-LMMSE', lambda s, c, snr, r, e: otfs.simulate(s, c, snr, r, est_channel=e)),
    ]
    wf_of = {'OFDM 1-tap': ofdm_1tap, 'OFDM band-freq': ofdm_band,
             'OTFS band-DD': otfs, 'OTFS full-LMMSE': otfs}

    print(f"{'SNR':>5} | " + " | ".join(f"{lab:>14}" for lab, _ in methods))
    print('-' * (7 + 17 * len(methods)))
    results = {lab: [] for lab, _ in methods}
    results['snr'] = list(args.snr)
    for snr in args.snr:
        row = []
        for lab, fn in methods:
            rng = np.random.default_rng(1000 + int(snr))
            b = run_point(fn, wf_of[lab], channel, coder, qam, snr, args.frames, rng, soft,
                          pilot_snr=args.pilot_snr)
            results[lab].append(b)
            row.append(b)
        print(f"{snr:5.1f} | " + " | ".join(f"{b:14.2e}" for b in row))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        # Monte-Carlo resolution: below ~1/(bits) we observe zero errors and can
        # no longer measure BER. Plot only measured points; do NOT clamp to a flat
        # floor (which would masquerade as a real error floor).
        res_floor = 1.0 / (args.frames * otfs.frame_num_qam() * qam.bits_per_symbol)
        snr_arr = np.array(results['snr'], float)
        styles = {'OFDM 1-tap': 'o--', 'OFDM band-freq': 's-',
                  'OTFS band-DD': '^-', 'OTFS full-LMMSE': 'd:'}
        labels = {'OFDM 1-tap': 'ZP-OFDM single-tap',
                  'OFDM band-freq': f'ZP-OFDM ICI-aware banded (Q={Q_star})',
                  'OTFS band-DD': f'OTFS banded DD (Ld={Ld},Kd={Kd})',
                  'OTFS full-LMMSE': 'OTFS full LMMSE (genie)'}
        for lab, _ in methods:
            y = np.array(results[lab], float)
            y[y <= 0] = np.nan                     # below MC resolution -> not drawn
            plt.semilogy(snr_arr, y, styles[lab], label=labels[lab],
                         linewidth=1.6, markersize=6)
        plt.axhline(res_floor, color='gray', ls=':', lw=0.8,
                    label=f'MC resolution ($\\approx${res_floor:.0e})')
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('SNR [dB]'); plt.ylabel('Coded BER')
        code_lbl = f'LDPC r={coder.rate:.2f}' if soft else 'K=7 CC r=1/2'
        plt.title(f'Domain-specific equalizers ($\\epsilon_{{max}}$={args.eps_max}, '
                  f'{args.qam}-QAM, {code_lbl})')
        plt.legend(); plt.tight_layout()
        plt.savefig(args.out, dpi=130)
        print(f"\nSaved figure to {args.out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()
