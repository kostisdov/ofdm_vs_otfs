"""Bias-corrected LMMSE equalization with per-symbol SINR for soft decoding.

For the linear model y = H x + n, x ~ CN(0, I) (unit-energy QPSK), n ~ CN(0, s2 I)
with s2 = 1/snr_lin, the LMMSE estimate is

    x_hat = (H^H H + s2 I)^{-1} H^H y = W y,

which is *biased*: E[x_hat_i | x_i] = mu_i x_i with mu_i = (W H)_ii != 1. Feeding
x_hat straight into a fixed-SNR LLR (as if it were an unbiased AWGN observation)
mis-calibrates the soft information -- the harder the MMSE shrinks (large matrices,
low SNR), the worse the mismatch, which can let a truncated equalizer look better
than the full one. We remove the bias and hand the decoder the true per-symbol SINR.

Using the posterior error covariance Sigma = s2 (H^H H + s2 I)^{-1}, and the input
covariance being the identity, one has the exact identities

    mu_i    = 1 - Sigma_ii            (real, in (0,1])
    z_i     = x_hat_i / mu_i          (unbiased estimate, z_i = x_i + w_i)
    gamma_i = mu_i / Sigma_ii         (unbiased per-symbol SINR, Var(w_i) = 1/gamma_i)

`H` is whatever channel model the equalizer assumes (it may be banded); the LLRs
are then self-consistent with that model, so the full-channel equalizer -- which
uses the correct H -- is a genuine lower bound on the banded one.
"""

import numpy as np


def mmse_equalize(H, y, noise_var):
    """Return (z, gamma): unbiased LMMSE estimate and per-symbol SINR.

    `noise_var` is the true per-cell noise variance in y (the value the AWGN
    stage actually used, P_ch/snr_lin). Passing the nominal 1/snr_lin instead
    mis-scales gamma in deep-fade frames, so callers must supply the measured
    noise variance for the LLRs to be calibrated (mean|z-x|^2 == mean 1/gamma).
    """
    s2 = noise_var
    Hh = H.conj().T
    Ainv = np.linalg.inv(Hh @ H + s2 * np.eye(H.shape[0]))
    x_hat = Ainv @ (Hh @ y)
    sigma = s2 * np.real(np.diag(Ainv))          # Sigma_ii = posterior error var
    sigma = np.clip(sigma, 1e-12, 1.0 - 1e-12)
    mu = 1.0 - sigma                             # biased per-symbol gain
    z = x_hat / mu                               # unbiased estimate
    gamma = mu / sigma                           # unbiased per-symbol SINR
    return z, gamma
