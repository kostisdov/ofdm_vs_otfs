"""Doubly-selective (time- and frequency-selective) tapped-delay-line channel.

A single realization is a set of P taps, each with an integer delay l_p, a
complex gain g_p, and a Doppler shift nu_p. The tap gain evolves in time as
    h_p(n) = g_p * exp(j 2 pi nu_p n Ts).
The same realization is applied identically to both waveforms so that the
ZP-OFDM vs Zak-OTFS comparison is on equal footing.
"""

import numpy as np


class DoublySelectiveChannel:
    def __init__(self, delays_samples, gains_db, dopplers_hz, fs, seed=None):
        self.l = np.asarray(delays_samples, dtype=int)
        self.P = len(self.l)
        g_lin = np.sqrt(10.0 ** (np.asarray(gains_db) / 10.0))
        self.g_lin = g_lin / np.sqrt(np.sum(g_lin ** 2))  # normalize total power
        self.nu = np.asarray(dopplers_hz, dtype=float)
        self.fs = float(fs)
        self.ts = 1.0 / self.fs
        self.rng = np.random.default_rng(seed)
        self.l_max = int(self.l.max())
        self._draw_gains()

    def _draw_gains(self):
        """Draw per-tap complex Rayleigh gains for a new frame realization."""
        phase = (self.rng.standard_normal(self.P) + 1j * self.rng.standard_normal(self.P)) / np.sqrt(2)
        self.g = self.g_lin * phase

    def new_realization(self):
        self._draw_gains()

    def tap_gain_timeseries(self, num_samples):
        """h_p(n) for n = 0..num_samples-1, shape (P, num_samples)."""
        return self.tap_gain_at(np.arange(num_samples))

    def tap_gain_at(self, n):
        """Exact time-varying tap gains at (global) sample indices n. (P, len)."""
        n = np.asarray(n)
        return self.g[:, None] * np.exp(1j * 2 * np.pi * self.nu[:, None] * n * self.ts)

    def apply(self, x):
        """Linear time-varying convolution: y(n) = sum_p h_p(n) x(n - l_p).

        x is a 1-D time-domain vector; output has the same length (trailing
        samples that would spill past the block are produced by zero-padding x,
        matching the ZP/overlap-add receiver convention).
        """
        x = np.asarray(x, dtype=complex)
        Nn = len(x)
        h = self.tap_gain_timeseries(Nn)
        y = np.zeros(Nn, dtype=complex)
        for p in range(self.P):
            lp = self.l[p]
            xd = np.concatenate([np.zeros(lp, dtype=complex), x[: Nn - lp]]) if lp > 0 else x
            y += h[p] * xd
        return y


class JakesTDLChannel:
    """Realistic tapped-delay-line channel with FRACTIONAL delays and a Jakes
    (Clarke) Doppler spectrum per physical path. Fractional delays are realized
    on the sample grid by (windowed) sinc interpolation, so each physical path
    spreads over several delay taps; each path fades as a sum-of-sinusoids with
    classical Doppler f_D. The resulting delay-Doppler channel is NOT sparse,
    which stresses any finite-complexity (banded) equalizer -- OFDM or OTFS.

    Exposes the same interface as DoublySelectiveChannel: l, P, tap_gain_at,
    apply, new_realization, nu (Doppler-spread proxy for band selection).
    """

    def __init__(self, delays_ns, powers_db, fd_hz, fs, n_sin=16, seed=None,
                 sinc_span=6):
        self.fs = float(fs)
        self.ts = 1.0 / self.fs
        self.tau = np.asarray(delays_ns, float) * 1e-9 * self.fs   # frac. delays [samples]
        p = 10.0 ** (np.asarray(powers_db, float) / 10.0)
        self.p = p / p.sum()
        self.fd = float(fd_hz)
        self.Q = len(self.tau)
        self.n_sin = n_sin
        self.rng = np.random.default_rng(seed)
        self.l_max = int(np.ceil(self.tau.max())) + sinc_span
        self.l = np.arange(self.l_max + 1)
        self.P = self.l_max + 1
        self.nu = np.array([self.fd])                 # for band-Q selection
        # windowed-sinc delay spread: contribution of path q to integer delay l
        L = self.l[:, None]
        hann = 0.5 * (1 + np.cos(np.pi * (L - self.tau[None, :]) / (sinc_span + 1)))
        hann = np.clip(hann, 0, 1)
        self.w = np.sinc(L - self.tau[None, :]) * hann * np.sqrt(self.p)[None, :]  # (P, Q)
        self._draw()

    def _draw(self):
        self.theta = self.rng.uniform(0, 2 * np.pi, (self.Q, self.n_sin))
        self.phi = self.rng.uniform(0, 2 * np.pi, (self.Q, self.n_sin))

    def new_realization(self):
        self._draw()

    def _path_gains(self, n):
        n = np.asarray(n, float)
        # g_q(n) = (1/sqrt(K)) sum_k exp(j(2pi f_D cos(theta_qk) n Ts + phi_qk))
        arg = (2 * np.pi * self.fd * np.cos(self.theta)[:, :, None] * (n[None, None, :] * self.ts)
               + self.phi[:, :, None])
        return np.exp(1j * arg).sum(axis=1) / np.sqrt(self.n_sin)     # (Q, len)

    def tap_gain_at(self, n):
        return self.w @ self._path_gains(n)          # (P, len) h_l(n)=sum_q w[l,q] g_q(n)

    def tap_gain_timeseries(self, num_samples):
        return self.tap_gain_at(np.arange(num_samples))

    def apply(self, x):
        x = np.asarray(x, dtype=complex)
        Nn = len(x)
        h = self.tap_gain_at(np.arange(Nn))          # (P, Nn)
        y = np.zeros(Nn, dtype=complex)
        for lp in range(self.P):
            if lp == 0:
                y += h[lp] * x
            else:
                y += h[lp] * np.concatenate([np.zeros(lp, complex), x[:Nn - lp]])
        return y


class EstimatedChannel:
    """Imperfect-CSI wrapper. The receiver observes each tap's slow-time gain
    (one noisy sample per OFDM/OTFS symbol, as from guard/embedded pilots) and
    reconstructs the intra-symbol variation by DFT (delay-Doppler) interpolation.
    Exposes the same l / P / tap_gain_at interface as the true channel, so the
    equalizers build H from the *estimate* while the signal propagates through
    the true channel. `pilot_snr_db` sets the per-observation pilot quality.
    """

    def __init__(self, true_chan, n_sym, sym_stride, pilot_snr_db, rng):
        self.l = true_chan.l
        self.P = true_chan.P
        self.l_max = true_chan.l_max
        self.ts = true_chan.ts
        self.nu = true_chan.nu       # Doppler spread is slow/well-estimated -> used only for band Q
        self.N = n_sym
        self.stride = sym_stride                       # samples between symbols
        # noisy slow-time observations of each tap gain (one per symbol)
        t_sym = np.arange(n_sym) * sym_stride
        a_true = true_chan.tap_gain_at(t_sym)          # (P, N)
        n0 = 10.0 ** (-pilot_snr_db / 10.0)
        noise = np.sqrt(n0 / 2) * (rng.standard_normal(a_true.shape)
                                   + 1j * rng.standard_normal(a_true.shape))
        a_hat = a_true + noise
        self.A = np.fft.fft(a_hat, axis=1)             # (P, N) Doppler coeffs

    def tap_gain_at(self, n):
        """Reconstruct h_p(n) at arbitrary sample indices by band-limited DFT
        interpolation of the noisy slow-time estimate. Signed frequency bins
        (fftfreq) are used so negative Dopplers interpolate correctly."""
        n = np.asarray(n, dtype=float)
        k = np.fft.fftfreq(self.N) * self.N            # signed bins: 0,1,..,-2,-1
        # h_p(n) = (1/N) sum_k A_p[k] exp(j2pi k (n/stride)/N)
        phase = np.exp(1j * 2 * np.pi * np.outer(k, n / self.stride) / self.N)  # (N, len)
        return (self.A @ phase) / self.N               # (P, len)


class EffectiveFilteredChannel:
    """Wraps a channel with an LTI transmit filter g (pulse shaping). The
    effective delay taps are the true taps convolved with g along delay:
    h_eff_m(n) = sum_k g[k] h_{m-k}(n). Being LTI, g adds delay spread but no
    ICI; the equalizer accounts for it through this effective channel.
    """

    def __init__(self, true_chan, g):
        self.true = true_chan
        self.g = np.asarray(g, dtype=complex)
        self.Lg = len(self.g)
        self.ts = true_chan.ts
        self.nu = true_chan.nu
        self.l_true = int(true_chan.l_max)
        self.l_max = self.l_true + self.Lg - 1
        self.l = np.arange(self.l_max + 1)
        self.P = self.l_max + 1

    def new_realization(self):
        self.true.new_realization()

    def tap_gain_at(self, n):
        n = np.asarray(n)
        taps = self.true.tap_gain_at(n)                  # (P_true, len)
        dense = np.zeros((self.l_true + 1, taps.shape[1]), dtype=complex)
        dense[np.asarray(self.true.l, int)] += taps      # place paths at delays
        eff = np.zeros((self.l_max + 1, taps.shape[1]), dtype=complex)
        for k in range(self.Lg):                         # convolve along delay with g
            eff[k:k + self.l_true + 1] += self.g[k] * dense
        return eff

    def tap_gain_timeseries(self, num):
        return self.tap_gain_at(np.arange(num))

    def apply(self, x):
        x = np.asarray(x, dtype=complex)
        Nn = len(x)
        h = self.tap_gain_at(np.arange(Nn))
        y = np.zeros(Nn, dtype=complex)
        for lp in range(self.P):
            y += h[lp] * (x if lp == 0 else np.concatenate([np.zeros(lp, complex), x[:Nn - lp]]))
        return y


def design_lowpass(num_loaded, fft_size, filt_len):
    """Hann-windowed sinc pulse-shaping filter matched to the occupied band."""
    fc = 0.5 * (num_loaded / fft_size) * 1.1             # one-sided cutoff (cyc/sample), 10% margin
    n = np.arange(filt_len) - (filt_len - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * n) * np.hanning(filt_len)
    return h / np.sum(h)


def awgn(x, snr_db, rng):
    """Add complex AWGN at the given SNR (measured on x)."""
    p = np.mean(np.abs(x) ** 2)
    n0 = p / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + noise
