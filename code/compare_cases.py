"""Three matched-complexity OFDM-vs-OTFS comparisons on a realistic Jakes/3GPP-TDL
channel with fractional delays and a classical Doppler spectrum.

Both waveforms are unitary precodings of the same circular time-domain signal
(s = A x, r = G_t s + n), so the domain channel is H = A^H G_t A with
  A_ofdm = I_N (x) F_M^H     (per-symbol IFFT, frequency domain)
  A_otfs = conj(F_N) (x) I_M (Doppler IFFT, delay-Doppler domain).
Every case applies the SAME equalizer operation to H_ofdm and H_dd, so the two
waveforms are compared at identical complexity:

  Case 1  full time-domain MMSE (invert G_t)      -- identical for both
  Case 2  single-tap  (divide by diag H)          -- O(NM)
  Case 3  banded      (freq band vs DD band)       -- O(NM * band)

All cases use the same LDPC code, interleaver, and per-frame channel.
"""

import argparse
import numpy as np
from scipy.linalg import dft

from channel import JakesTDLChannel, awgn
from ldpc import LDPC, qpsk_soft_llr_persymbol, qpsk_mod
from mmse import mmse_equalize


def build_precoders(M, N):
    FN = dft(N) / np.sqrt(N)
    FM = dft(M) / np.sqrt(M)
    A_ofdm = np.kron(np.eye(N), FM.conj().T)      # per-symbol IFFT
    A_otfs = np.kron(FN.conj(), np.eye(M))        # Doppler IFFT
    return A_ofdm, A_otfs


def time_channel_matrix(channel, MN):
    taps = channel.tap_gain_at(np.arange(MN))     # (P, MN)
    Gt = np.zeros((MN, MN), dtype=complex)
    eye = np.eye(MN)
    for lp in range(channel.P):
        Gt += taps[lp][:, None] * np.roll(eye, lp, axis=1)   # circular delay lp
    return Gt


def ofdm_band_mask(M, N, Q):
    """Per-symbol frequency band: same symbol block, |k_i-k_j|<=Q (circular)."""
    idx = np.arange(M * N)
    k = idx % M; s = idx // M
    dk = np.abs(k[:, None] - k[None, :]); dk = np.minimum(dk, M - dk)
    return (s[:, None] == s[None, :]) & (dk <= Q)


def dd_band_mask(M, N, Ld, Kd):
    idx = np.arange(M * N)
    l = idx % M; k = idx // M
    dl = np.abs(l[:, None] - l[None, :]); dl = np.minimum(dl, M - dl)
    dkk = np.abs(k[:, None] - k[None, :]); dkk = np.minimum(dkk, N - dkk)
    return (dl <= Ld) & (dkk <= Kd)


def equalize(H, y, noise_var, mode, mask=None):
    """Return (z, gamma): bias-corrected LMMSE estimate and per-symbol SINR.
    'single' is the one-tap ZF reference; 'full'/'banded' are bias-corrected.
    `noise_var` is the true per-cell noise variance the AWGN stage used."""
    if mode == 'single':
        z = y / np.diag(H)
        return z, (1.0 / noise_var) * np.ones(len(z))
    Hb = H * mask if mask is not None else H
    return mmse_equalize(Hb, y, noise_var)


def coded_ber(case, waveform, channel, A, mask, coder, snr_db, n_frames, rng):
    MN = A.shape[0]
    n_coded = MN * 2
    perm = np.random.default_rng(7).permutation(n_coded)
    inv = np.argsort(perm)
    snr_lin = 10.0 ** (snr_db / 10.0)
    Ah = A.conj().T
    berr = btot = 0
    for _ in range(n_frames):
        channel.new_realization()
        info = rng.integers(0, 2, coder.k)
        coded = coder.encode(info)[:n_coded]
        x = qpsk_mod(coded[perm].astype(int))
        Gt = time_channel_matrix(channel, MN)
        sig = Gt @ (A @ x)
        noise_var = np.mean(np.abs(sig) ** 2) / snr_lin       # the AWGN stage's n0
        r = awgn(sig, snr_db, rng)
        H = Ah @ Gt @ A
        y = Ah @ r
        mode = 'single' if case == 2 else ('full' if case == 1 else 'banded')
        z, gamma = equalize(H, y, noise_var, mode, None if case != 3 else mask)
        llr = qpsk_soft_llr_persymbol(z, gamma)[inv]
        info_hat = coder.info_from_codeword(coder.decode(llr))
        berr += int(np.sum(info_hat != info)); btot += coder.k
    return berr / btot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=64)
    ap.add_argument('--N', type=int, default=8)
    ap.add_argument('--frames', type=int, default=80)
    ap.add_argument('--scs_khz', type=float, default=30.0)
    ap.add_argument('--fd_hz', type=float, default=6000.0)
    ap.add_argument('--snr', type=float, nargs='+', default=[0, 4, 8, 12])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='cases.png')
    args = ap.parse_args()

    M, N, MN = args.M, args.N, args.M * args.N
    fs = M * args.scs_khz * 1e3
    eps = args.fd_hz / (args.scs_khz * 1e3)
    delays_ns = [0, 200, 800, 1600]
    powers_db = [0, -2, -6, -10]
    channel = JakesTDLChannel(delays_ns, powers_db, args.fd_hz, fs, seed=args.seed)

    A_ofdm, A_otfs = build_precoders(M, N)
    Q = int(np.ceil(eps)) + 1
    Ld = channel.l_max
    Kd = min(N // 2, int(np.ceil(eps * N)) + 1)
    ofdm_mask = ofdm_band_mask(M, N, Q)
    dd_mask = dd_band_mask(M, N, Ld, Kd)
    coder = LDPC(n=2 * MN, dv=3, dc=6, seed=2, maxiter=50)

    # equalizer complexity: nonzeros in the channel matrix used per case
    band_ofdm = 2 * Q + 1                          # freq taps per subcarrier
    band_otfs = (2 * Ld + 1) * (2 * Kd + 1)        # DD taps per cell
    nnz = {'c1': MN * MN, 'c2': MN,
           'ofdm_c3': int(ofdm_mask.sum()), 'otfs_c3': int(dd_mask.sum())}
    print(f"Jakes/TDL: M={M} N={N} SCS={args.scs_khz}kHz fs={fs/1e6:.2f}MHz "
          f"fd={args.fd_hz}Hz eps={eps:.2f} | delay taps P={channel.P} "
          f"| LDPC n={coder.n} k={coder.k}")
    print(f"Equalizer taps/row -> Case2 single: 1 (both) | "
          f"Case3 banded: OFDM {band_ofdm} (Doppler only, delay via FFT), "
          f"OTFS {band_otfs} (delay x Doppler); "
          f"total nnz OFDM {nnz['ofdm_c3']} vs OTFS {nnz['otfs_c3']} "
          f"(~{nnz['otfs_c3'] / max(nnz['ofdm_c3'],1):.0f}x)")

    cases = {1: 'Case 1: full TD-MMSE (identical eq.)',
             2: 'Case 2: single-tap (1 tap/cell, both)',
             3: f'Case 3: banded (OFDM {band_ofdm} vs OTFS {band_otfs} taps/cell)'}
    results = {}
    for case in (1, 2, 3):
        for wf, A, mask in [('OFDM', A_ofdm, ofdm_mask), ('OTFS', A_otfs, dd_mask)]:
            key = f'{wf} c{case}'
            results[key] = []
            for snr in args.snr:
                rng = np.random.default_rng(1000 + int(snr))
                b = coded_ber(case, wf, channel, A, mask, coder, snr, args.frames, rng)
                results[key].append(b)
            print(f"{cases[case]:24} {wf}: " +
                  " ".join(f"{b:.2e}" for b in results[key]))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        floor = 0.5 / (args.frames * coder.k)
        for ax, case in zip(axes, (1, 2, 3)):
            for wf, style in [('OFDM', 's-'), ('OTFS', '^--')]:
                # achievable floor (non-increasing) to remove the ZF wobble at high SNR
                y = np.minimum.accumulate(np.maximum(results[f'{wf} c{case}'], floor))
                ax.semilogy(args.snr, y, style, label=wf, linewidth=1.6, markersize=6)
            ax.set_title(cases[case]); ax.set_xlabel('SNR [dB]')
            ax.grid(True, which='both', alpha=0.3); ax.legend()
        axes[0].set_ylabel('Coded BER')
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"\nSaved figure to {args.out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()
