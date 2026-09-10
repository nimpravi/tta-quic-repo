# Decomposing Label-Free Test-Time Adaptation for Encrypted QUIC Traffic Classification Under Abrupt Temporal Drift

<!-- Zenodo: create a release on GitHub, link the repo in Zenodo, then paste
     the badge Markdown Zenodo gives you, replacing the two placeholders below.
     Use the CONCEPT DOI (always resolves to the latest version), not a
     version-specific DOI. -->
[![DOI](https://zenodo.org/badge/1290349685.svg)](https://doi.org/10.5281/zenodo.21210653)

Code, pinned environment, and complete experimental record for a
single-author IEEE Letters manuscript studying what test-time
adaptation (TTA) actually recovers when an encrypted-traffic
classifier meets a real, documented distribution shift.

**Author:** Praveen Hegde (IEEE Senior Member) ·

## TL;DR

## TL;DR

A public QUIC classifier (MM-CESNET-V2, trained on week W-2022-44 of
CESNET-QUIC22) loses **22.65 accuracy points** when Google changed the TLS
certificates of its services during W-2022-45. This repository measures what
label-free test-time adaptation recovers, what it costs, who pays that cost,
and how it compares to the alternative an operator actually has.

Under a leakage-clean protocol with every hyperparameter frozen on W-2022-46
before W-2022-47 is touched:

| Condition (episodic, 50 steps) | Recovery | % of gap |
|---|---|---|
| BN-statistics recalibration only | **+2.43 ± 0.15 p** | 10.7% |
| + filtered entropy gradients (q = 0.5) | **+3.06 ± 0.27 p** | 13.5% |
| + unfiltered entropy gradients (q = 1.0) | **+1.32 ± 0.53 p** | 5.8% |

**That +3.06 is a net of two large opposing effects.** Split by a class
partition built from pre-report weeks and hash-locked before the report week
was touched, it is a gain of **+6.87 points on drifted classes** and a loss
of **2.79 points on classes that did not drift**. Below roughly **29 percent**
drift prevalence the method is net harmful overall, and prevalence cannot be
measured without the labels the method exists to avoid needing.

**The loss is a tail, not a tax.** Per-class counts show recalibration
imposing a broad cost (median large undrifted class loses 1.22 points) and
entropy filtering removing most of it (median 0.18) while leaving one
catastrophic case. `instagram`, with 0.2 points of drift, falls from
**0.9721 to 0.7548** accuracy on 52,866 flows and accounts for **85 percent**
of the filtered method's net loss on undrifted traffic. Excluding it, that
loss is 0.46 points. No collapse diagnostic detects any of this: aggregate
accuracy rises, macro-F1 stays flat, and all 102 classes remain predicted.

**Supervised retraining on stale labels does three to five times better.**
Labels 1, 3 or 7 days old recover **+11.07 to +14.72 points** at every delay
tested, with three to seven times less collateral damage, and retraining only
the classifier leaves undrifted traffic untouched (+0.11). A pre-registered
kill rule fired on this comparison. The operational quantity is not label
age, which costs at most 1.5 points across a week, but the interval between
drift onset and the first post-drift labels.

**It is not an artifact of the protocol.** Strictly causal streaming costs
0.18 points against the transductive headline; the result replicates across
seven days at +3.04 ± 0.24; and the partition threshold is not load-bearing,
with the sign of the transfer and the operating point above break-even
holding at every cut from 5 to 20 points.

Full numbers, kill-rule outcomes and the correction record:
[`results/RESULTS.md`](results/RESULTS.md).

---

## The self-audit (read this if you run TTA experiments)

An early version of this pipeline probed accuracy every 50 steps. The probe
called `model.eval()` and never restored training mode, so from step 51
onward BN statistics froze and a **different algorithm** ran, an undocumented
two-phase hybrid created by the measurement itself. It inflated the headline
by over a point, corrupted hyperparameter selection, and manufactured a
spurious ordering effect. Bit-level provenance tracking made the distortion
detectable, diagnosable and correctable. Every superseded number is retained
with its cause attached in [`results/RESULTS.md`](results/RESULTS.md)
sections 7 and 8.

`scripts/tta_guards.py` ships the safeguard as infrastructure: every
evaluation in scripts 02 to 21 is wrapped in `guarded_eval`, which snapshots
each module's training flag and a checksum of every BN buffer and parameter
before the evaluation and asserts them unchanged after, and recorded
bit-level anchors are asserted at runtime. Rerunning script 06 under the
guards reproduces the recorded accuracies bit for bit.

Practical rules that fell out of this: evaluate once, at the final step;
never let an evaluation touch a model that will continue adapting; pin your
environment; and check what your headline number is a net of, because a
collapse check will not tell you.

**Known limitation.** The pipeline is bit-deterministic within a process for
every condition except the full-network fine-tune of Experiment B, which is
the only one that puts the whole model in training mode and therefore
activates its three dropout layers. Torch's global RNG is not seeded. Two
independent runs of that condition differ by 0.014 to 0.133 points; both are
released.

---

## Verification

```bash
python scripts/21_verify_all.py
```

Regenerates every number in the manuscript from the released artifacts and
compares each against the value recorded in `results/RESULTS.md`. It loads no
model, reads no dataset, and runs in seconds. Current state: **231 checks
passed, 0 failed, 0 artifacts missing.**

A failure means an artifact and the record disagree; the script cannot say
which is wrong. It re-runs no experiment, so it detects an inconsistent
record, not a wrong experiment. It also names, in its own output, the three
artifacts it does not parse, so that "verification passed" can never be read
as "everything was checked".

Two claims that used to rest on hand inspection are now checked here: the
dropout modules behind the determinism exception in RESULTS.md 7.6, and the
identity of the module the `head` capacity retrains.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md                       # correction record, v1 to v4 (summary)
├── requirements.txt                   # loose, human-readable
├── requirements-lock.txt              # exact environment that produced the results
│
├── PREREGISTRATION_switchpoint.md     # governs the switch-point selection
│                                      #   SHA-256 4ebd14fb…712e45, locked BEFORE its run
├── PREREGISTRATION_streaming_delayed_label.md
│                                      # governs experiments A to E
│                                      #   section 0.1 records which families it
│                                      #   preceded and which it did not
│
├── scripts/
│   │   # v1 to v3 pipeline
│   ├── 02_errorbars.py                # headline: tune on W-46, report on W-47
│   ├── 03_mechanism_errorbars.py      # unfiltered (q=1.0) control
│   ├── 04_bnstats_control.py          # stats-only matched control
│   ├── 05_collapse_check.py           # macro-F1 and class-collapse diagnostics
│   ├── 06_inperiod_reference.py       # self-measured gap denominator
│   ├── 07_w45_depth_probe.py          # intra-week drift onset trace (Fig. 2)
│   ├── 08_leakage_demo.py             # falsification test for the audit finding
│   ├── 09_hybrid_schedule.py          # deliberate two-phase reimplementation
│   ├── 10_oracle_ceiling.py           # matched-capacity labeled reference
│   ├── 11_switchpoint_probe.py        # coarse probe, superseded by the selection
│   ├── 11_switchpoint_select.py       # PRE-REGISTERED selection; kill rule A fired
│   ├── 12_filtered100_errorbars.py    # 100-step error bars
│   │
│   │   # v4: experiments A to E, the post-hoc analysis, and verification
│   ├── 13_window_trend_analysis.py    # per-window trend, reads raw JSON only
│   ├── 14_stream_order_audit.py       # A0: ordering, period lengths, day map
│   ├── 15_acrossday_replication.py    # Experiment E: seven days
│   ├── 16_streaming.py                # Experiment A: --tune then --report
│   ├── 17_delayed_label.py            # Experiment B: --tune then --report
│   ├── 18_nondrifted_control.py       # Experiment C: --c1, --partition, --c2
│   ├── 19_w46_stability_reference.py  # Experiment D: matched stability reference
│   ├── 20_delayed_label_partition.py  # post-hoc partition (NOT pre-registered)
│   ├── 21_verify_all.py               # regenerates every paper number from raw
│   └── tta_guards.py                  # guarded_eval, assert_anchor; used by 02-21
│
├── legacy/                            # superseded scripts, kept as evidence
│   └── (state-mutating-probe versions; do not use)
├── diagnostics/                       # determinism audits, repro-unit diagnostics
│
├── results/
│   ├── RESULTS.md                     # CANONICAL numbers and correction record
│   ├── raw/                           # console logs and JSON checkpoints (below)
│   └── superseded/                    # pre-correction artifacts, marked with cause
│
└── (not tracked)
    ├── data/                          # CESNET-QUIC22, downloaded on first use
    ├── models/                        # MM-CESNET-V2 weights, downloaded on first use
    └── manuscript/                    # LaTeX source; excluded by .gitignore
```

### What is in `results/raw/`

Grouped by the section of `results/RESULTS.md` each one supports.

| Section | Artifacts |
|---|---|
| 1, gap and denominator | `inperiod_reference.json` |
| 2 and 3, decomposition and step count | `errorbars_progress.json`, `bnstats_progress_steps50.json`, `mechanism_progress_steps50.json`, `filtered100_progress.json` |
| 4, class-level behaviour | `collapse_check_q0.5_steps50.json` |
| 5, ordering and period lengths | `streaming_order_audit.json`, `w45_depth_probe.json` |
| 6, Experiment A streaming | `streaming_config.json`, `streaming_results.json` |
| 7 and 8, audit and provenance | `leakage_demo.json`, `determinism_audit_run.txt` |
| 9, labeled reference | `oracle_matched_progress.json` |
| 10, switch-point selection | `switchpoint_select.json`, `switchpoint_probe.json`, `hybrid_progress.json` |
| 11, Experiment D | `w46_stability_reference.json` |
| 12, Experiment E | `acrossday_progress.json` |
| 13, Experiment B delayed-label | `delayed_label_config.json`, `delayed_label_progress.json` |
| 14, Experiment C non-drifted cost | `nondrifted_c1_progress.json`, `class_partition.json`, `class_partition.sha256`, `nondrifted_c2_progress.json` |
| 15, post-hoc partition | `delayed_label_partition_progress.json` |

Console logs for each run are in the same directory with matching names.

### Two conventions worth knowing before you read the tree

**Files that gate other files.** `streaming_config.json` and
`delayed_label_config.json` are written by the `--tune` mode of their scripts
and required by `--report`, so the report week cannot be touched before the
configuration is selected on the tuning week. `class_partition.sha256` must
exist and match `class_partition.json` before
`18_nondrifted_control.py --c2` will run, so the class partition cannot be
adjusted after a report-week number exists. These are mechanical, not
conventions of good behaviour.

**Nothing is deleted.** Superseded results stay in `results/superseded/` and
superseded scripts stay in `legacy/`, each with the cause of its supersession
recorded in `results/RESULTS.md` section 7. The two switch-point scripts are
both present for the same reason: `11_switchpoint_probe.py` is the coarse
probe that motivated the selection, and `11_switchpoint_select.py` is the
pre-registered selection that superseded it.

## Reproducing the results

**Verifying without running anything.** Every number in the paper is
regenerated from the released raw artifacts by:

```bash
python scripts/21_verify_all.py
```

That script loads no model, reads no dataset, needs no GPU, and finishes in
seconds. It reports 145 checks passed, 0 failed, 0 artifacts missing, and it
lists by name the four artifacts it does not parse, so a pass cannot be
mistaken for a complete check.

**Requirements.** Python 3.12, CPU. Every number in the paper was produced on
a Windows 11 laptop with no GPU.

```bash
python -m venv tent-env
tent-env/Scripts/activate        # Windows; use bin/activate elsewhere
pip install -r requirements-lock.txt
```

`requirements-lock.txt` is the exact environment that produced the results,
captured from it, with the interpreter and platform recorded in its header.
`requirements.txt` is a looser, human-readable alternative.

**Forcing CPU.** The scripts select CUDA automatically when it is available.
The released numbers were produced on CPU, so to reproduce them exactly on a
machine with a GPU:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/02_errorbars.py --size S --K 5     # Linux/macOS
set CUDA_VISIBLE_DEVICES=  && python scripts/02_errorbars.py --size S --K 5   # Windows
```

**Data and weights.** CESNET-QUIC22 (size S) downloads on first use via
`cesnet-datazoo` into `./data/`; MM-CESNET-V2 W-44 weights via
`cesnet-models` into `./models/`. Both are public.

**Run order and cost** (from the repository root; all resumable via JSON or
torch checkpoints). Timings are measured, not estimated.

| Step | Command | ~Time (CPU) |
|---|---|---|
| Denominator | `python scripts/06_inperiod_reference.py --size S` | 25 min |
| Drift onset | `python scripts/07_w45_depth_probe.py --size S` | 25 min |
| Headline | `python scripts/02_errorbars.py --size S --K 5` | 3.5 h |
| Unfiltered control | `python scripts/03_mechanism_errorbars.py --size S --K 3` | 55 min |
| Stats-only control | `python scripts/04_bnstats_control.py --size S --K 3` | 65 min |
| Collapse check | `python scripts/05_collapse_check.py --size S --quant 0.5` | 30 min |
| Leakage demo | `python scripts/08_leakage_demo.py --size S` | 40 min |
| Two-phase (rejected) | `python scripts/09_hybrid_schedule.py --size S --K 3` | 1.7 h |
| Labeled reference | `python scripts/10_oracle_ceiling.py --size S --K 3` | 1.5 h |
| Switch probe | `python scripts/11_switchpoint_probe.py --size S` | 35 min |
| Switch selection | `python scripts/11_switchpoint_select.py --size S` | 2.5 h |
| Table II fill | `python scripts/12_filtered100_errorbars.py --size S --K 3` | 1.7 h |
| Window trend | `python scripts/13_window_trend_analysis.py` | seconds |
| A0 ordering audit | `python scripts/14_stream_order_audit.py --size S` | 5 min |
| Across-day (E) | `python scripts/15_acrossday_replication.py --size S --K 3` | 3.6 h |
| Streaming tune (A) | `python scripts/16_streaming.py --tune --size S` | 2.7 h |
| Streaming report (A) | `python scripts/16_streaming.py --report --size S` | 5.5 h |
| Delayed-label tune (B) | `python scripts/17_delayed_label.py --tune --size S` | ~4 h |
| Delayed-label report (B) | `python scripts/17_delayed_label.py --report --size S` | ~8 h |
| Non-drifted control (C1) | `python scripts/18_nondrifted_control.py --c1 --size S --K 3` | 1.6 h |
| Class partition | `python scripts/18_nondrifted_control.py --partition --size S` | 28 min |
| Partition report (C2) | `python scripts/18_nondrifted_control.py --c2 --size S --K 3` | ~1 h |
| Stability reference (D) | `python scripts/19_w46_stability_reference.py --size S` | 15 min |
| Post-hoc partition | `python scripts/20_delayed_label_partition.py --size S --combined` | 4.5 h |
| Capacity sizes | `python scripts/count_params.py` | seconds |
| Threshold sweep | `python scripts/22_threshold_sweep.py` | seconds |
| Verify everything | `python scripts/21_verify_all.py` | seconds |

Two steps are gated deliberately and will refuse to run out of order.
`16_streaming.py --report` and `17_delayed_label.py --report` require the
configuration file written by their own `--tune` mode, so the report week
cannot be touched before the configuration is selected on the tuning week.
`18_nondrifted_control.py --c2` requires `class_partition.sha256` to exist
and match `class_partition.json`, so the class partition cannot be adjusted
after a report-week number exists.

Between `--partition` and `--c2` you must hash `class_partition.json` into
`class_partition.sha256` and commit both. `--c2` verifies the digest and
refuses to run otherwise, so the partition cannot be adjusted after a
report-week number exists.

`22_threshold_sweep.py` writes `threshold_sweep.json` into the working
directory. Move it into `results/raw/` before committing, or the verifier
will read the fresh copy in preference to the committed one and the two can
drift apart silently.

**Determinism.** The pipeline is bit-deterministic within a process except as
noted in the self-audit section. "Seeds" vary the one genuine stochastic
factor, the adaptation batch ordering
(`numpy.random.default_rng(1000*window + k)`). Expected values, including
bit-level anchors, are documented in `results/RESULTS.md` and asserted at
runtime.

---

## Manuscript and pre-registrations

Two pre-registration documents govern the study and are committed with their
SHA-256 recorded:

- `PREREGISTRATION_switchpoint.md`, hash-locked **before** the switch-point
  selection it governs.
- `PREREGISTRATION_streaming_delayed_label.md`, hash-locked before
  Experiments B, C and D, and after A and E. Its section 0.1 records that
  asymmetry explicitly and states what may and may not be claimed for each
  family.

Kill rules in both documents fired and are reported as such: the switch-point
schedule was rejected on ordering instability before the report week was
consulted, and the delayed-label comparison triggered the framing change
recorded in `results/RESULTS.md` section 13.

The manuscript source is not tracked in this repository. See `.gitignore`.

## Citation

If you use this code or the findings, please cite the paper (BibTeX
will be added upon acceptance) and see `CITATION.cff` for the
repository itself.

## License

MIT (see `LICENSE`).