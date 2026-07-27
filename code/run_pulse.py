"""Pulse shaping validation:
  (a) spectrum / mask -- plain OFDM (rectangular) vs filtered OFDM vs OTFS,
  (b) BER vs SNR -- plain OFDM vs filtered OFDM, both with the ICI-aware
      multitap equalizer, QPSK, 0..16 dB in 0.5 dB steps.

Point: an LTI pulse-shaping filter cuts out-of-band emission (meets a mask) and
adds only delay spread, which the FFT diagonalizes and the ZP guard absorbs; the
Doppler ICI is handled by the multitap equalizer regardless. So filtering costs a
few edge subcarriers but not BER on the data subcarriers.
"""

import argparse
import numpy as np
from scipy.linalg import dft

from channel import (DoublySelectiveChannel, EffectiveFilteredChannel,
                     design_lowpass)
from zpofdm import ZPOFDM
from ldpc import qpsk_mod, qpsk_soft_llr_persymbol, LDPC


def coded_ber(wf, channel, data_idx, coder, snr_db, n_frames, rng):
    M, N = wf.M, wf.n_sym
    nd = len(data_idx)
    n_coded = 2 * nd * N
    perm = np.random.default_rng(7).permutation(n_coded)
    inv = np.argsort(perm)
    snr_lin = 10.0 ** (snr_db / 10.0)
    berr = btot = 0
    for _ in range(n_frames):
        channel.new_realization()
        info = rng.integers(0, 2, coder.k)
        coded = coder.encode(info)[:n_coded]
        syms = qpsk_mod(coded[perm].astype(int)).reshape(nd, N, order='F')
        grid = np.zeros((M, N), dtype=complex)
        grid[data_idx] = syms
        z, gamma = wf.simulate(grid.flatten(order='F'), channel, snr_db, rng)
        zd = z.reshape(M, N, order='F')[data_idx].flatten(order='F')
        gd = gamma.reshape(M, N, order='F')[data_idx].flatten(order='F')
        llr = qpsk_soft_llr_persymbol(zd, gd)[inv]
        info_hat = coder.info_from_codeword(coder.decode(llr))
        berr += int(np.sum(info_hat != info)); btot += coder.k
    return berr / btot


def welch_psd(x, nfft=256):
    x = np.asarray(x); L = nfft; step = nfft // 2
    segs = [x[i:i + L] for i in range(0, len(x) - L, step)]
    w = np.hanning(L)
    P = np.zeros(L)
    for s in segs:
        P += np.abs(np.fft.fftshift(np.fft.fft(s * w))) ** 2
    P /= max(len(segs), 1)
    f = np.linspace(-0.5, 0.5, L, endpoint=False)
    return f, 10 * np.log10(P / P.max() + 1e-12)


def uncoded_ber(wf, channel, data_idx, snr_db, n_frames, rng):
    M, N = wf.M, wf.n_sym
    nd = len(data_idx)
    berr = btot = 0
    for _ in range(n_frames):
        channel.new_realization()
        bits = rng.integers(0, 2, 2 * nd * N)
        syms = qpsk_mod(bits).reshape(nd, N, order='F')
        grid = np.zeros((M, N), dtype=complex)
        grid[data_idx] = syms
        z, _ = wf.simulate(grid.flatten(order='F'), channel, snr_db, rng)
        Xd = z.reshape(M, N, order='F')[data_idx].flatten(order='F')
        bh = np.empty(2 * nd * N, int)
        bh[0::2] = (np.real(Xd) < 0); bh[1::2] = (np.imag(Xd) < 0)
        berr += int(np.sum(bh != bits)); btot += len(bits)
    return berr / btot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=64)
    ap.add_argument('--numdata', type=int, default=48)
    ap.add_argument('--N', type=int, default=4)
    ap.add_argument('--cp', type=int, default=16)
    ap.add_argument('--filt', type=int, default=12)
    ap.add_argument('--guard', type=int, default=4)
    ap.add_argument('--frames', type=int, default=50)
    ap.add_argument('--eps', type=float, default=0.15)
    ap.add_argument('--out', default='pulse.png')
    args = ap.parse_args()

    M, nd, N, cp = args.M, args.numdata, args.N, args.cp
    fs = 1e6; ts = 1 / fs
    numzeros = M - nd
    # DC-centered occupied band: data on low-frequency subcarriers (around index 0
    # and M-1), nulls at the band edge (around Nyquist, index M/2) where the
    # lowpass pulse-shaping filter rolls off.
    data_idx = np.concatenate([np.arange(0, (nd + 1) // 2),
                               np.arange(M - nd // 2, M)])
    dopplers = [0.0, args.eps * fs / M, -0.7 * args.eps * fs / M]
    channel = DoublySelectiveChannel([0, 2, 4], [0, -3, -8], dopplers, fs, seed=1)
    g = design_lowpass(nd + 2 * args.guard, M, args.filt)   # passband covers data + guard nulls
    ch_filt = EffectiveFilteredChannel(channel, g)          # eff. delay = 8 + filt - 1 < cp

    ofdm = ZPOFDM(M, N, cp, ts, Q=None, q_guard=1, equalizer='mmse')
    Q = ofdm._band_Q(channel)
    coder = LDPC(n=2 * nd * N, dv=3, dc=6, seed=2, maxiter=50)
    print(f"M={M} data={nd} cp={cp} N={N} eps={args.eps} Q={Q} filt={args.filt} "
          f"LDPC n={coder.n} k={coder.k}")

    # ---- (a) spectra (long signal for a smooth PSD) ----
    rng = np.random.default_rng(0)
    Np = 128
    gpsd = np.zeros((M, Np), dtype=complex)
    gpsd[data_idx] = qpsk_mod(rng.integers(0, 2, 2 * nd * Np)).reshape(nd, Np, order='F')
    xsym = np.fft.ifft(gpsd, axis=0) * np.sqrt(M)                 # per-symbol IFFT
    x_plain = np.vstack([xsym, np.zeros((cp, Np))]).flatten(order='F')   # +ZP guard
    x_filt = np.convolve(x_plain, g, mode='same')
    xdd = qpsk_mod(rng.integers(0, 2, 2 * M * Np)).reshape(M, Np, order='F')
    x_otfs = np.fft.ifft(xdd, axis=1, norm='ortho').flatten(order='F')
    fpl, Ppl = welch_psd(x_plain, 512); fof, Pof = welch_psd(x_otfs, 512); ffi, Pfi = welch_psd(x_filt, 512)

    # ---- (b) BER vs SNR ----
    snrs = np.arange(0, 12.01, 0.5)
    ber_plain, ber_filt = [], []
    for snr in snrs:
        r1 = np.random.default_rng(100 + int(2 * snr))
        ber_plain.append(coded_ber(ofdm, channel, data_idx, coder, snr, args.frames, r1))
        r2 = np.random.default_rng(100 + int(2 * snr))
        ber_filt.append(coded_ber(ofdm, ch_filt, data_idx, coder, snr, args.frames, r2))
    print("SNR  plain     filtered")
    for s, a, b in zip(snrs, ber_plain, ber_filt):
        if abs(s - round(s)) < 1e-9:
            print(f"{s:4.0f} {a:.2e}  {b:.2e}")

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        occ = nd / M / 2
        ax[0].plot(fpl, Ppl, label='OFDM (rectangular)', lw=1.2)
        ax[0].plot(fof, Pof, label='OTFS (rectangular)', lw=1.0, alpha=0.7)
        ax[0].plot(ffi, Pfi, label='OFDM filtered', lw=1.6)
        ax[0].axvspan(occ, 0.5, color='red', alpha=0.06)
        ax[0].axvspan(-0.5, -occ, color='red', alpha=0.06)
        ax[0].axhline(-40, color='k', ls=':', lw=0.8, label='mask (-40 dB)')
        ax[0].set_xlim(-0.5, 0.5); ax[0].set_ylim(-90, 3)
        ax[0].set_xlabel('Normalized frequency'); ax[0].set_ylabel('PSD [dB]')
        ax[0].set_title('Spectrum / out-of-band emission'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        yp = np.minimum.accumulate(np.maximum(ber_plain, 1e-5))
        yf = np.minimum.accumulate(np.maximum(ber_filt, 1e-5))
        ax[1].semilogy(snrs, yp, 's-', label='OFDM (rectangular)', ms=4)
        ax[1].semilogy(snrs, yf, 'o-', label='OFDM filtered', ms=4)
        ax[1].set_xlabel('SNR [dB]'); ax[1].set_ylabel('Coded BER')
        ax[1].set_title(f'QPSK LDPC BER, ICI-aware eq. (Q={Q}, $\\epsilon$={args.eps})')
        ax[1].grid(True, which='both', alpha=0.3); ax[1].legend()
        fig.tight_layout(); fig.savefig(args.out, dpi=130)
        print(f"\nSaved figure to {args.out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()
