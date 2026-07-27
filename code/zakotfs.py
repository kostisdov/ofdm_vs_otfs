"""OTFS (reduced-CP, rectangular-pulse) transmitter and perfect-CSI LMMSE
delay-Doppler detector.

The transmit signal is the IFFT of the DD grid along the Doppler dimension
(S = X_dd @ F_N^H), serialized column-major with a single cyclic prefix -- the
"precoded single-carrier / IFFT-along-Doppler" view. Detection forms the exact
MN x MN delay-Doppler channel matrix H_dd = B G_t A and applies LMMSE. This is
the equal-footing baseline the OTFS critique targets; it reduces to standard
OTFS with rectangular pulses.
"""

import numpy as np
from scipy.linalg import dft
from channel import awgn
from mmse import mmse_equalize


class OTFS:
    def __init__(self, M, N, cp_len, ts):
        self.M = M                      # delay bins (per time-slot samples)
        self.N = N                      # Doppler bins (time-slots)
        self.Lcp = cp_len
        self.ts = ts
        self.MN = M * N
        FN = dft(N) / np.sqrt(N)        # unitary DFT (symmetric)
        IM = np.eye(M)
        self.A = np.kron(FN.conj(), IM)  # x_dd -> s_time  (IFFT along Doppler)
        self.B = np.kron(FN, IM)         # r_time -> y_dd  (FFT along Doppler)

    # ---- geometry ----
    def frame_num_qam(self):
        return self.MN

    def sym_len(self):
        return self.M            # time-slot stride (samples per Doppler symbol)

    def frame_len(self):
        return self.MN + self.Lcp

    # ---- transmit ----
    def modulate(self, qam_symbols):
        X_dd = np.asarray(qam_symbols).reshape(self.M, self.N, order='F')
        S = np.fft.ifft(X_dd, axis=1, norm='ortho')     # IFFT along Doppler
        s_time = S.flatten(order='F')
        cp = s_time[-self.Lcp:]
        return np.concatenate([cp, s_time])

    # ---- receive + LMMSE (perfect CSI) ----
    def _time_channel_matrix(self, channel):
        MN = self.MN
        taps = channel.tap_gain_at(np.arange(MN))        # (P, MN), true or estimated
        Gt = np.zeros((MN, MN), dtype=complex)
        eye = np.eye(MN)
        for p in range(channel.P):
            lp = int(channel.l[p])
            Gt += taps[p][:, None] * np.roll(eye, lp, axis=1)   # circular delay by l_p
        return Gt

    def _dd_band_mask(self, Ld, Kd):
        """2D delay-Doppler band mask on the vectorized (column-major) DD grid:
        keep couplings within +/-Ld in delay and +/-Kd in Doppler (circular).
        This is the DD-domain analogue of the banded frequency-domain mask used
        by the OFDM equalizer.
        """
        M, N = self.M, self.N
        idx = np.arange(self.MN)
        l = idx % M
        k = idx // M
        dl = np.abs(l[:, None] - l[None, :]); dl = np.minimum(dl, M - dl)
        dk = np.abs(k[:, None] - k[None, :]); dk = np.minimum(dk, N - dk)
        return (dl <= Ld) & (dk <= Kd)

    def simulate(self, qam_symbols, channel, snr_db, rng, dd_band=None, est_channel=None):
        """End-to-end circular OTFS block model: r = G_t s (true channel) + AWGN,
        then bias-corrected DD-domain LMMSE. dd_band=(Ld,Kd) truncates the DD
        channel to a delay-Doppler band (domain-specific reduced-complexity
        equalizer); dd_band=None is the full DD LMMSE, which uses the correct
        channel and is therefore a genuine lower bound on the banded one. The
        equalizer uses est_channel if given (imperfect CSI), else the true
        channel. Returns (z, gamma) for bias-corrected soft decoding.
        """
        X_dd = np.asarray(qam_symbols).reshape(self.M, self.N, order='F')
        S = np.fft.ifft(X_dd, axis=1, norm='ortho')
        s_time = S.flatten(order='F')                     # MN samples (core)
        Gt_true = self._time_channel_matrix(channel)
        sig = Gt_true @ s_time
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = np.mean(np.abs(sig) ** 2) / snr_lin   # the AWGN stage's n0
        r = awgn(sig, snr_db, rng)
        Gt_csi = self._time_channel_matrix(est_channel) if est_channel is not None else Gt_true
        H_dd = self.B @ Gt_csi @ self.A
        if dd_band is not None:
            H_dd = H_dd * self._dd_band_mask(dd_band[0], dd_band[1])
        y_dd = self.B @ r
        return mmse_equalize(H_dd, y_dd, noise_var)
