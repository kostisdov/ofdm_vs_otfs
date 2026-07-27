"""Standard CP-OFDM with single-tap frequency-domain equalization.

This is the incumbent OFDM receiver: a cyclic prefix makes the channel appear
circular, so after the FFT each subcarrier is scaled by a single complex gain and
equalized by one complex division. Under a doubly dispersive channel the gain is
only the time-averaged frequency response; the intra-symbol variation leaks energy
across subcarriers (ICI) that the single tap cannot remove, which floors the BER.
The soft LLRs use the per-subcarrier SNR of the single-tap model (the ICI is
unmodeled, exactly as in a deployed receiver), so the floor appears naturally.
"""

import numpy as np
from scipy.linalg import dft
from channel import awgn


class CPOFDM:
    def __init__(self, M, n_sym, cp_len, ts):
        self.M = M
        self.n_sym = n_sym
        self.N = n_sym
        self.Lcp = cp_len
        self.ts = ts
        self.F = dft(M) / np.sqrt(M)
        self.Fh = self.F.conj().T

    def frame_num_qam(self):
        return self.M * self.n_sym

    def sym_len(self):
        return self.M + self.Lcp

    def frame_len(self):
        return self.sym_len() * self.n_sym

    def modulate(self, qam_symbols):
        X = np.asarray(qam_symbols).reshape(self.M, self.n_sym, order='F')
        x = np.fft.ifft(X, axis=0, norm='ortho')
        block = np.vstack([x[self.M - self.Lcp:], x])       # prepend cyclic prefix
        return block.flatten(order='F')

    def _diag_gain(self, channel, sym_index):
        """Single-tap per-subcarrier gains: the diagonal of the circular
        frequency-domain channel, i.e. the time-averaged frequency response."""
        M = self.M
        n_global = sym_index * (M + self.Lcp) + self.Lcp + np.arange(M)
        taps = channel.tap_gain_at(n_global)                # (P, M)
        Ht = np.zeros((M, M), dtype=complex)
        for p in range(channel.P):
            lp = int(channel.l[p])
            cols = (np.arange(M) - lp) % M
            Ht[np.arange(M), cols] += taps[p]               # circular channel
        return np.diag(self.F @ Ht @ self.Fh)               # per-subcarrier gain

    def demodulate(self, rx_time, channel, noise_var):
        M = self.M
        R = rx_time.reshape(M + self.Lcp, self.n_sym, order='F')
        z = np.zeros((M, self.n_sym), dtype=complex)
        g = np.zeros((M, self.n_sym), dtype=float)
        s2 = noise_var if noise_var else 1e-3
        for s in range(self.n_sym):
            Y = np.fft.fft(R[self.Lcp:, s], norm='ortho')   # drop CP, to freq
            hd = self._diag_gain(channel, s)
            p = np.abs(hd) ** 2
            mu = p / (p + s2)                               # single-tap MMSE gain
            z[:, s] = (hd.conj() * Y / (p + s2)) / mu       # unbiased estimate
            g[:, s] = p / s2                                # per-subcarrier SNR (ICI unmodeled)
        return z.flatten(order='F'), g.flatten(order='F')

    def simulate(self, qam_symbols, channel, snr_db, rng, est_channel=None):
        tx = self.modulate(qam_symbols)
        sig = channel.apply(tx)
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = np.mean(np.abs(sig) ** 2) / snr_lin
        rx = awgn(sig, snr_db, rng)
        csi = est_channel if est_channel is not None else channel
        return self.demodulate(rx, csi, noise_var)
