# Decomposing Label-Free Test-Time Adaptation for Encrypted QUIC Traffic Classification Under Abrupt Temporal Drift

<!-- Zenodo: create a release on GitHub, link the repo in Zenodo, then paste
     the badge Markdown Zenodo gives you, replacing the two placeholders below.
     Use the CONCEPT DOI (always resolves to the latest version), not a
     version-specific DOI. -->
[![DOI](https://zenodo.org/badge/xxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxx)

Code, pinned environment, and complete experimental record for a
single-author IEEE Letters manuscript studying what test-time
adaptation (TTA) actually recovers when an encrypted-traffic
classifier meets a real, documented distribution shift.

**Author:** Praveen Hegde (IEEE Senior Member) ·

## TL;DR

A public QUIC classifier (MM-CESNET-V2, trained on week W-2022-44 of
CESNET-QUIC22) loses **22.65 accuracy points** when Google changed its
TLS certificates mid-week W-45. Under a leakage-clean protocol (all
hyperparameters frozen on W-46 before W-47 is touched):

| Condition (steps = 50) | Recovery | % of gap |
|---|---|---|
| BN-statistics recalibration only (no gradients) | **+2.43 ± 0.15 p** | 10.7% |
| + filtered entropy gradients (q = 0.5), headline | **+3.06 ± 0.27 p** | 13.5% |
| + unfiltered entropy gradients (q = 1.0) | **+1.32 ± 0.53 p** | 5.8% |
| Two-phase schedule (post-hoc observation) | **+4.31 ± 0.49 p** | 19.0% |

Findings: recalibration alone carries ~4/5 of the recovery; entropy
filtering gates the **sign** of the gradient contribution (+0.62 p
filtered vs −1.12 p unfiltered, relative to stats-only); no
class-collapse signature (macro-F1 flat, all 102 classes retained);
recovery is nearly insensitive to batch ordering.

## The self-audit (read this if you run TTA experiments)

An early version of this pipeline probed accuracy every 50 steps. The
probe called `model.eval()` and never restored training mode, so from
step 51 onward BN statistics froze and a **different algorithm** ran,
an undocumented two-phase hybrid created by the measurement itself. It
inflated the headline by over a point, corrupted hyperparameter
selection, and manufactured a spurious ordering effect. Bit-level
provenance tracking (frozen accuracies are exact dyadic rationals)
made the distortion detectable, diagnosable, and correctable. The full
mechanism, the falsification test that confirmed it, and every
superseded number with its cause are recorded in
[`results/RESULTS.md`](results/RESULTS.md) (Sections 7–8) and
[`CHANGELOG.md`](CHANGELOG.md).

Practical rules that fell out of this: evaluate once, at the final
step; never let an evaluation touch a model that will continue
adapting; pin your environment (`requirements-lock.txt`); TTA changes
model state by design, so measurements become interventions very
easily.

## Repository layout

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md                 # correction record (summary)
├── requirements.txt             # loose, human-readable
├── requirements-lock.txt        # exact pins; canonical environment
├── scripts/                     # numbered pipeline, run in order below
│   ├── 02_errorbars.py          # headline: tune on W-46, report on W-47
│   ├── 03_mechanism_errorbars.py# unfiltered (q=1.0) control
│   ├── 04_bnstats_control.py    # stats-only matched control
│   ├── 05_collapse_check.py     # macro-F1 / class-collapse diagnostics
│   ├── 06_inperiod_reference.py # self-measured gap denominator
│   ├── 07_w45_depth_probe.py    # intra-week drift onset trace
│   ├── 08_leakage_demo.py       # falsification test for the audit finding
│   ├── 09_hybrid_schedule.py    # deliberate two-phase reimplementation
│   ├── 10_oracle_ceiling.py     # labeled-oracle ceiling (matched capacity)
│   ├── 11_switchpoint_probe.py  # switch-point sensitivity (tuning week only)
│   ├── 12_filtered100_errorbars.py # Table II symmetry fill (K=3 at 100 steps)
│   └── tta_guards.py            # state-audit guards: guarded_eval, anchors
├── legacy/                      # superseded scripts, kept as evidence
│   └── (state-mutating-probe versions; do not use)
├── diagnostics/                 # determinism audits, repro-unit diag
├── results/
│   ├── RESULTS.md               # CANONICAL numbers + correction record
│   ├── raw/                     # clean console logs + JSON checkpoints
│   └── superseded/              # pre-correction artifacts, marked
└── manuscript/                  # LaTeX source (IEEEtran), modular sections
    ├── main.tex
    ├── sections/*.tex
    └── figures/                 # standalone TikZ/pgfplots figure sources + PDFs
```

## Reproducing the results

**Requirements.** Python 3.12, CPU only (every number in the paper was
produced on a 16 GB Windows laptop with no GPU). Install exact pins:

```bash
python -m venv tent-env
tent-env/Scripts/activate        # Windows; use bin/activate elsewhere
pip install -r requirements-lock.txt
```

**Data and weights.** CESNET-QUIC22 (size S) downloads on first use
via `cesnet-datazoo` into `./data/`; MM-CESNET-V2 W-44 weights via
`cesnet-models` into `./models/`. Both are public.

**Run order and cost** (from repo root, all resumable via JSON
checkpoints):

| Step | Command | ~Time (CPU) |
|---|---|---|
| Denominator | `python scripts/06_inperiod_reference.py --size S` | 25 min |
| Drift onset | `python scripts/07_w45_depth_probe.py --size S` | 25 min |
| Headline | `python scripts/02_errorbars.py --size S --K 5` | 3.5 h |
| Unfiltered control | `python scripts/03_mechanism_errorbars.py --size S --K 3` | 55 min |
| Stats-only control | `python scripts/04_bnstats_control.py --size S --K 3` | 65 min |
| Collapse check | `python scripts/05_collapse_check.py --size S --quant 0.5` | 30 min |
| Leakage demo | `python scripts/08_leakage_demo.py --size S` | 40 min |
| Two-phase | `python scripts/09_hybrid_schedule.py --size S --K 3` | 1.7 h |
| Oracle ceiling | `python scripts/10_oracle_ceiling.py --size S --K 3` | 1.5 h |
| Switch probe | `python scripts/11_switchpoint_probe.py --size S` | 35 min |
| Table II fill | `python scripts/12_filtered100_errorbars.py --size S --K 3` | 1.7 h |

Determinism note: the pipeline is bit-deterministic within a process;
"seeds" vary the one genuine stochastic factor, the adaptation batch
ordering (`numpy.random.default_rng(1000*window + k)`). Expected
values, including bit-level anchors, are documented in
`results/RESULTS.md` §8 and asserted by scripts 08–09 at runtime.

## Manuscript

`manuscript/` contains the IEEEtran source targeting IEEE
Communications Letters (4 pages compiled). Figures are TikZ/pgfplots
drawn from the measured values; standalone versions for production are
in `manuscript/figures/`.

## Citation

If you use this code or the findings, please cite the paper (BibTeX
will be added upon acceptance) and see `CITATION.cff` for the
repository itself.

## License

MIT (see `LICENSE`).