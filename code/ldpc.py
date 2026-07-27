"""Self-contained regular (d_v, d_c) LDPC code with log-domain sum-product
belief-propagation decoding, and soft QPSK demapping. No external dependencies
beyond numpy -- used to test whether stronger (soft-decision) FEC lets an
ICI-aware banded-frequency OFDM receiver harvest the cross-symbol diversity and
approach OTFS.
"""

import numpy as np


def build_regular_ldpc(n, dv=3, dc=6, rng=None):
    """Random LDPC parity-check matrix H (m x n) with exact column weight dv
    and near-uniform row weight ~dc. Each column picks dv distinct rows,
    preferring the least-loaded rows to keep row weights balanced."""
    rng = rng or np.random.default_rng(0)
    m = n * dv // dc
    for _ in range(50):
        H = np.zeros((m, n), dtype=np.int8)
        rload = np.zeros(m, dtype=int)
        ok = True
        order = rng.permutation(n)
        for c in order:
            # candidate rows sorted by current load (+ jitter), take dv least loaded
            jitter = rng.random(m) * 0.5
            cand = np.argsort(rload + jitter)[:dv]
            if len(np.unique(cand)) < dv:
                ok = False; break
            H[cand, c] = 1
            rload[cand] += 1
        if ok and H.sum(1).min() > 0 and H.sum(0).min() > 0:
            return H
    raise RuntimeError("LDPC construction failed")


def h_to_g(H):
    """GF(2) reduce H to RREF; return generator G (k x n), the free (info)
    column indices, and pivot (parity) column indices. Rows of G span ker(H)."""
    H = (H.copy() % 2).astype(np.int8)
    m, n = H.shape
    pivots, r = [], 0
    is_piv = np.zeros(n, bool)
    for c in range(n):
        if r >= m:
            break
        rows = np.where(H[r:, c] == 1)[0]
        if len(rows) == 0:
            continue
        pr = r + rows[0]
        H[[r, pr]] = H[[pr, r]]
        for rr in range(m):
            if rr != r and H[rr, c]:
                H[rr] = (H[rr] + H[r]) % 2
        pivots.append(c); is_piv[c] = True; r += 1
    free = [c for c in range(n) if not is_piv[c]]
    k = len(free)
    G = np.zeros((k, n), dtype=np.int8)
    for i, f in enumerate(free):
        G[i, f] = 1
        for ri, pc in enumerate(pivots):
            if H[ri, f]:
                G[i, pc] = 1
    return G, np.array(free), np.array(pivots)


class LDPC:
    def __init__(self, n=512, dv=3, dc=6, seed=0, maxiter=50):
        self.H = build_regular_ldpc(n, dv, dc, np.random.default_rng(seed))
        self.G, self.free, self.pivots = h_to_g(self.H)
        self.n = n
        self.k = self.G.shape[0]
        self.maxiter = maxiter
        self.rate = self.k / self.n
        # rectangular support (row weight is exactly dc) for a vectorized BP
        m = self.H.shape[0]
        self.m = m
        self.chk_mat = np.array([np.where(self.H[r] == 1)[0] for r in range(m)])  # (m, dc)

    def encode(self, info_bits):
        return (np.asarray(info_bits) @ self.G) % 2

    def info_from_codeword(self, cw):
        return cw[self.free]

    def decode(self, llr):
        """Vectorized log-domain sum-product BP. llr[v]=log P(b=0)/P(b=1).
        Messages live on the rectangular edge array (m x dc). Returns the
        decoded n-bit codeword (hard)."""
        llr = np.asarray(llr, dtype=float)
        cm = self.chk_mat                       # (m, dc) variable index per edge
        M = llr[cm]                             # var->check messages, (m, dc)
        cw = (llr < 0).astype(np.int8)
        for _ in range(self.maxiter):
            T = np.tanh(np.clip(M / 2.0, -30, 30))
            prod = np.prod(T, axis=1, keepdims=True)
            Tsafe = np.where(np.abs(T) < 1e-12, 1e-12, T)
            others = np.clip(prod / Tsafe, -1 + 1e-12, 1 - 1e-12)
            E = 2.0 * np.arctanh(others)        # check->var messages, (m, dc)
            totalE = np.zeros(self.n)
            np.add.at(totalE, cm.ravel(), E.ravel())
            total = llr + totalE
            cw = (total < 0).astype(np.int8)
            if not np.any((self.H @ cw) % 2):
                return cw
            M = total[cm] - E                   # var->check update
        return cw


def qpsk_mod(bits):
    """Unit-power Gray QPSK matching qpsk_soft_llr: b0->real, b1->imag,
    bit 0 -> +1. Used on the LDPC soft path so mapping and demapping agree."""
    b = np.asarray(bits).reshape(-1, 2)
    return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)


def qpsk_soft_llr(x_hat, snr_lin):
    """Per-bit LLRs for unit-power Gray QPSK from equalized symbols.
    bit0<-real, bit1<-imag; L = log P(0)/P(1) = 2*sqrt(2)*snr_lin*component."""
    k = 2.0 * np.sqrt(2.0) * snr_lin
    llr = np.empty(2 * len(x_hat))
    llr[0::2] = k * np.real(x_hat)
    llr[1::2] = k * np.imag(x_hat)
    return llr


def qpsk_soft_llr_persymbol(z, gamma):
    """Bias-corrected per-bit LLRs from an unbiased estimate z and per-symbol
    SINR gamma (effective 1/N0). Same Gray mapping as qpsk_soft_llr, but each
    symbol is weighted by its own SINR instead of a single nominal SNR."""
    k = 2.0 * np.sqrt(2.0) * np.asarray(gamma)
    llr = np.empty(2 * len(z))
    llr[0::2] = k * np.real(z)
    llr[1::2] = k * np.imag(z)
    return llr
