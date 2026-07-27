# OFDM with Adaptive ICI-Aware Equalization for Doubly Dispersive Channels

Preprint + reproducible simulation code for a like-for-like comparison of **OFDM**
and **OTFS** on doubly dispersive (high-mobility) channels.

**Thesis.** The inter-carrier-interference (ICI) floor that doubly dispersive
channels impose on OFDM can be removed by a *banded, multi-tap frequency-domain
equalizer whose order is set adaptively from the delay–Doppler channel estimate*
— the band follows the Doppler (mobility), the coefficients follow the fast
fading. Under identical coding and interleaving, this receiver comes **within
about 1 dB of a delay–Doppler OTFS detector at roughly two orders of magnitude
lower equalizer complexity**. A matched-complexity study and an FDE-OTFS
experiment show that OTFS's edge comes from its *joint* equalizer, not from the
modulation-level spreading, and that the same diversity is recoverable by channel
coding over OFDM.

📄 **Paper:** [`paper/main.pdf`](paper/main.pdf)

---

## Repository layout

```
ofdm_vs_otfs/
├── paper/               LaTeX source, compiled PDF, and the four figures
│   ├── main.tex
│   ├── main.pdf
│   ├── cases_jakes.png      (Fig. 1)
│   ├── ber_eps03.png        (Fig. 2a)
│   ├── ber_eps12.png        (Fig. 2b)
│   └── pulse_shaping.png    (Fig. 3)
└── code/                Python simulation suite (numpy / scipy / matplotlib)
    ├── channel.py           doubly-selective + Jakes/3GPP-TDL channels, AWGN
    ├── mmse.py              bias-corrected LMMSE (unbiased estimate + per-symbol SINR)
    ├── ldpc.py              self-contained regular LDPC (BP) + QPSK soft LLRs
    ├── zpofdm.py            ZP-OFDM TX + banded ICI-aware equalizer
    ├── cpofdm.py            standard CP-OFDM + single-tap equalizer
    ├── zakotfs.py           OTFS TX + delay–Doppler LMMSE detector
    ├── compare_cases.py     matched-complexity precoding framework (+ Fig. 1)
    ├── run_waveforms.py     four-receiver comparison incl. FDE-OTFS (Fig. 2)
    ├── run_pulse.py         pulse shaping / spectral mask (Fig. 3)
    ├── run_ber.py           domain-specific BER sweep (imperfect CSI, conv/LDPC)
    ├── run_techniques.py    waveform-domain 3-technique BER sweep
    ├── qam.py, coding.py    QAM + convolutional-code helpers (commpy)
    └── requirements.txt
```

## Setup

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only `numpy`, `scipy`, and `matplotlib` are needed for the three paper figures;
`scikit-commpy` is used solely by `run_ber.py --code cc`.

## Reproducing the figures

Run from `code/`. These are Monte-Carlo simulations with dense linear algebra, so
each command takes on the order of **10–30 min** on a laptop.

```bash
# Fig. 1 — matched-complexity OFDM vs OTFS (Jakes/TDL, eps=0.2)
python compare_cases.py --frames 100 --snr 0 2 4 6 8 --out cases_jakes.png

# Fig. 2 — four receivers on realistic 3GPP-TDL-C, mild and severe Doppler
python run_waveforms.py --frames 150 --eps_max 0.3 --snr 0 1 2 3 4 5 6 8 --out ber_eps03.png
python run_waveforms.py --frames 150 --eps_max 1.2 --snr 0 1 2 3 4 5 6 8 --out ber_eps12.png

# Fig. 3 — pulse shaping / spectral containment
python run_pulse.py --frames 60 --out pulse_shaping.png
```

Copy the resulting PNGs into `paper/` and rebuild the PDF:

```bash
cd ../paper && pdflatex main.tex && pdflatex main.tex
```

## Method notes

- **Adaptive band.** The equalizer half-width is `Q = ceil(f_D / Δf) + Δ`, read
  from the Doppler support of the delay–Doppler channel estimate. It sizes the
  band to the mobility while the coefficients track the fast fading.
- **Bias-corrected soft decoding.** All receivers use `mmse.py`: each equalizer
  output is de-biased and weighted by its own post-equalization SINR, so the
  full-channel LMMSE is a genuine lower bound on any banded/mismatched version.
- **Matched-complexity comparison.** `compare_cases.py` / `run_waveforms.py`
  write every waveform as a unitary precoding of the same circular time-domain
  signal (`r = Gt · A · x`), so OFDM, FDE-OTFS, and DD-OTFS differ only in the
  precoder and the equalizer band — an apples-to-apples cost/performance test.

## Citation

```bibtex
@unpublished{dovelos_ofdm_ici_2026,
  author = {Konstantinos Dovelos},
  title  = {{OFDM} with Adaptive {ICI}-Aware Equalization for Doubly Dispersive Channels},
  note   = {Preprint},
  year   = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
