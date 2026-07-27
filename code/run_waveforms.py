"""Four deployable receivers on one realistic channel, matched-complexity framework.

All are unitary precodings of the same circular time-domain signal r = G_t A_tx x,
equalized with a banded LMMSE in domain A_eq, then de-precoded by D:

  CP-OFDM single-tap       A_tx=A_eq=A_ofdm, band Q=0,       D=I
  ZP-OFDM block ICI-aware   A_tx=A_eq=A_ofdm, band Q,        D=I     (proposed)
  FDE-OTFS                  A_tx=A_otfs, A_eq=A_ofdm, band Q, D=A_otfs^H A_ofdm
  OTFS banded DD            A_tx=A_eq=A_otfs, DD band,       D=I

FDE-OTFS is "OTFS at OFDM complexity": the OTFS spreading precode with the *same*
per-symbol frequency-domain banded equalizer as ZP-OFDM, then de-spread. It tests
whether the spreading helps once the equalizer is the cheap per-symbol one.

Soft decoding uses a bias-corrected per-symbol LLR for the whole linear receiver
R = D W A_eq^H: with r = G_t A_tx x + n and end-to-end G = R G_t A_tx,
  mu_i    = G_ii                       (biased gain)
  rho_i   = (G G^H)_ii - |mu_i|^2 + s2 (R R^H)_ii   (leakage + noise)
  z_i     = x_hat_i / mu_i,  gamma_i = |mu_i|^2 / rho_i
so the full-channel leakage a banded/mismatched equalizer leaves is charged
honestly to each symbol's SINR.
"""

import argparse
import numpy as np

from channel import JakesTDLChannel, awgn
from ldpc import LDPC, qpsk_soft_llr_persymbol, qpsk_mod
from compare_cases import (build_precoders, time_channel_matrix,
                           ofdm_band_mask, dd_band_mask)


def linear_llr(A_tx, A_eq, mask, D, Gt, r, noise_var):
    """Bias-corrected (z, gamma) for the linear receiver R = D W A_eq^H."""
    MN = A_tx.shape[0]
    s2 = noise_var
    Aeqh = A_eq.conj().T
    H_eq = (Aeqh @ Gt @ A_eq) * mask                 # equalizer's banded model
    Heqh = H_eq.conj().T
    W = np.linalg.solve(Heqh @ H_eq + s2 * np.eye(MN), Heqh)   # MMSE in A_eq domain
    R = D @ (W @ Aeqh)                               # full receiver: r -> data est.
    G = R @ (Gt @ A_tx)                              # effective end-to-end
    x_hat = R @ r
    mu = np.diag(G).copy()
    GGh = np.real(np.sum(np.abs(G) ** 2, axis=1))    # (G G^H)_ii
    RRh = np.real(np.sum(np.abs(R) ** 2, axis=1))    # (R R^H)_ii
    rho = np.maximum(GGh - np.abs(mu) ** 2 + s2 * RRh, 1e-12)
    mu = np.where(np.abs(mu) < 1e-9, 1e-9, mu)
    z = x_hat / mu
    gamma = np.abs(mu) ** 2 / rho
    return z, gamma


def coded_ber(A_tx, A_eq, mask, D, channel, coder, snr_db, n_frames, rng):
    MN = A_tx.shape[0]
    n_coded = MN * 2
    perm = np.random.default_rng(7).permutation(n_coded)
    inv = np.argsort(perm)
    snr_lin = 10.0 ** (snr_db / 10.0)
    berr = btot = 0
    for _ in range(n_frames):
        channel.new_realization()
        info = rng.integers(0, 2, coder.k)
        coded = coder.encode(info)[:n_coded]
        x = qpsk_mod(coded[perm].astype(int))
        Gt = time_channel_matrix(channel, MN)
        sig = Gt @ (A_tx @ x)
        noise_var = np.mean(np.abs(sig) ** 2) / snr_lin
        r = awgn(sig, snr_db, rng)
        z, gamma = linear_llr(A_tx, A_eq, mask, D, Gt, r, noise_var)
        llr = qpsk_soft_llr_persymbol(z, gamma)[inv]
        info_hat = coder.info_from_codeword(coder.decode(llr))
        berr += int(np.sum(info_hat != info)); btot += coder.k
    return berr / btot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=64)
    ap.add_argument('--N', type=int, default=8)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--snr', type=float, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 8])
    ap.add_argument('--eps_max', type=float, default=0.3)
    ap.add_argument('--scs_khz', type=float, default=30.0)
    ap.add_argument('--fc_ghz', type=float, default=3.5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='waveforms.png')
    args = ap.parse_args()

    M, N, MN = args.M, args.N, args.M * args.N
    fs = M * args.scs_khz * 1e3
    fd = args.eps_max * args.scs_khz * 1e3
    eps = args.eps_max
    delays_ns = [0, 200, 800, 1600]; powers_db = [0, -2, -6, -10]
    channel = JakesTDLChannel(delays_ns, powers_db, fd, fs, seed=args.seed)
    v_kmh = fd * 3e8 / (args.fc_ghz * 1e9) * 3.6

    A_ofdm, A_otfs = build_precoders(M, N)
    I = np.eye(MN)
    D_fde = A_otfs.conj().T @ A_ofdm
    Q = int(np.ceil(eps)) + 1
    Kd = min(N // 2, int(np.ceil(eps * N)) + 1)
    om0 = ofdm_band_mask(M, N, 0)                     # single tap
    omQ = ofdm_band_mask(M, N, Q)                     # Doppler band
    dm = dd_band_mask(M, N, channel.l_max, Kd)
    coder = LDPC(n=2 * MN, dv=3, dc=6, seed=2, maxiter=50)

    recv = [
        ('CP-OFDM single-tap',        A_ofdm, A_ofdm, om0, I),
        ('ZP-OFDM block ICI-aware',   A_ofdm, A_ofdm, omQ, I),
        ('FDE-OTFS',                  A_otfs, A_ofdm, omQ, D_fde),
        ('OTFS banded DD',            A_otfs, A_otfs, dm,  I),
    ]
    print(f"Jakes/TDL-C fd={fd:.0f} Hz (v={v_kmh:.0f} km/h @{args.fc_ghz} GHz) "
          f"P={channel.P} l_max={channel.l_max} | M={M} N={N} eps={eps} "
          f"Q={Q} Kd={Kd} | LDPC n={coder.n} k={coder.k} frames={args.frames}")
    results = {}
    print(f"{'SNR':>5} | " + " | ".join(f"{name:>22}" for name, *_ in recv))
    for snr in args.snr:
        row = []
        for name, Atx, Aeq, mask, D in recv:
            rng = np.random.default_rng(3000 + int(snr))
            b = coded_ber(Atx, Aeq, mask, D, channel, coder, snr, args.frames, rng)
            results.setdefault(name, []).append(b); row.append(b)
        print(f"{snr:5.1f} | " + " | ".join(f"{b:22.2e}" for b in row))

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        res_floor = 1.0 / (args.frames * MN * 2)
        styles = {'CP-OFDM single-tap': 'o--', 'ZP-OFDM block ICI-aware': 's-',
                  'FDE-OTFS': 'D-.', 'OTFS banded DD': '^-'}
        labels = {'CP-OFDM single-tap': 'CP-OFDM, single-tap (standard)',
                  'ZP-OFDM block ICI-aware': f'ZP-OFDM, block ICI-aware (proposed, Q={Q})',
                  'FDE-OTFS': 'FDE-OTFS (OTFS at OFDM cost)',
                  'OTFS banded DD': f'OTFS, banded DD (Ld={channel.l_max},Kd={Kd})'}
        for name, *_ in recv:
            y = np.array(results[name], float); y[y <= 0] = np.nan
            plt.semilogy(args.snr, y, styles[name], label=labels[name], lw=1.7, ms=6)
        plt.axhline(res_floor, color='gray', ls=':', lw=0.8,
                    label=f'MC resolution ($\\approx${res_floor:.0e})')
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('SNR [dB]'); plt.ylabel('Coded BER')
        plt.title(f'Waveform comparison ($\\varepsilon_{{max}}$={eps}, QPSK, LDPC r=0.5, TDL-C)')
        plt.legend(fontsize=8.5); plt.tight_layout(); plt.savefig(args.out, dpi=130)
        print(f"\nSaved figure to {args.out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()
