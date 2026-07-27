"""ZP-OFDM transmitter and perfect-CSI multi-tap (banded) ICI equalizer.

Per OFDM symbol the exact M x M frequency-domain ICI matrix H is assembled from
the true time-varying tap gains (Proposition 1 of the note),
    H = F @ H_t @ F^H,
truncated to a circular band of half-width Q, and inverted (ZF or MMSE). Q is
chosen from the normalized Doppler via Q = ceil(eps_max) + delta.
"""

import numpy as np
from scipy.linalg import dft
from channel import awgn
from mmse import mmse_equalize


class ZPOFDM:
    def __init__(self, M, n_sym, guard_len, ts, Q=None, q_guard=1, equalizer='mmse'):
        self.M = M
        self.n_sym = n_sym
        self.N = n_sym
        self.Lg = guard_len
        self.ts = ts
        self.Q_fixed = Q            # if None, set adaptively from Doppler
        self.q_guard = q_guard
        self.equalizer = equalizer
        self.F = dft(M) / np.sqrt(M)          # unitary DFT
        self.Fh = self.F.conj().T
        # circular frequency-distance matrix for banded masking
        idx = np.arange(M)
        d = np.abs(idx[:, None] - idx[None, :])
        self.circ_dist = np.minimum(d, M - d)

    # ---- geometry ----
    def frame_num_qam(self):
        return self.M * self.n_sym

    def sym_len(self):
        return self.M + self.Lg

    def frame_len(self):
        return self.sym_len() * self.n_sym

    # ---- transmit ----
    def modulate(self, qam_symbols):
        X = np.asarray(qam_symbols).reshape(self.M, self.n_sym, order='F')
        x = np.fft.ifft(X, axis=0, norm='ortho')            # unitary IFFT
        block = np.vstack([x, np.zeros((self.Lg, self.n_sym), dtype=complex)])
        return block.flatten(order='F')

    # ---- receive + equalize (perfect CSI) ----
    def _band_Q(self, channel):
        if self.Q_fixed is not None:
            return self.Q_fixed
        eps_max = np.max(np.abs(channel.nu)) * self.M * self.ts
        return int(np.ceil(eps_max)) + self.q_guard

    def _ici_matrix(self, channel, sym_index):
        M, Lg = self.M, self.Lg
        n_global = sym_index * (M + Lg) + np.arange(M)
        taps = channel.tap_gain_at(n_global)            # (P, M), true or estimated
        Ht = np.zeros((M, M), dtype=complex)
        for p in range(channel.P):
            lp = int(channel.l[p])
            cols = (np.arange(M) - lp) % M
            Ht[np.arange(M), cols] += taps[p]
        return self.F @ Ht @ self.Fh

    def demodulate(self, rx_time, channel, noise_var=None):
        """Banded multi-tap equalize. Returns (z, gamma): the unbiased LMMSE
        symbol estimate and the per-symbol SINR, both flattened column-major to
        match the transmit ordering, for bias-corrected soft decoding.
        `noise_var` is the true per-cell noise variance (n0), preserved through
        the unitary FFT, and is used to calibrate gamma."""
        M, Lg = self.M, self.Lg
        R = rx_time.reshape(M + Lg, self.n_sym, order='F')
        Q = self._band_Q(channel)
        mask = (self.circ_dist <= Q)
        l_max = channel.l_max
        z_out = np.zeros((M, self.n_sym), dtype=complex)
        g_out = np.zeros((M, self.n_sym), dtype=float)
        s2 = noise_var if noise_var else 1e-3
        for s in range(self.n_sym):
            col = R[:, s].copy()
            yM = col[:M].copy()
            if l_max > 0:                                   # ZP overlap-add
                yM[:l_max] += col[M:M + l_max]
            Y = np.fft.fft(yM, norm='ortho')
            Hq = self._ici_matrix(channel, s) * mask
            if self.equalizer == 'zf':
                z_out[:, s] = np.linalg.solve(Hq, Y)
                g_out[:, s] = 1.0 / s2                      # nominal fallback
            else:                                           # bias-corrected MMSE
                z_out[:, s], g_out[:, s] = mmse_equalize(Hq, Y, s2)
        return z_out.flatten(order='F'), g_out.flatten(order='F')

    def simulate(self, qam_symbols, channel, snr_db, rng, est_channel=None):
        """End-to-end: modulate -> true time-varying channel + AWGN -> ZP
        overlap-add + banded multi-tap equalize. The signal always propagates
        through the true `channel`; the equalizer uses `est_channel` if given
        (imperfect CSI), else the true channel (perfect CSI). Returns
        (z, gamma) for bias-corrected soft decoding."""
        tx = self.modulate(qam_symbols)
        sig = channel.apply(tx)
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = np.mean(np.abs(sig) ** 2) / snr_lin     # the AWGN stage's n0
        rx = awgn(sig, snr_db, rng)
        csi = est_channel if est_channel is not None else channel
        return self.demodulate(rx, csi, noise_var=noise_var)
