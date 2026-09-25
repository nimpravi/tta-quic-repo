# Repository structure to commit

The tree below is the full repository as it should stand for the current
manuscript. Items marked **NEW** are not in the layout the present README
describes; items marked **STALE** exist in that layout and no longer match
reality.

The single largest gap: the committed record stops at script 22 and
`RESULTS.md` v6, while the manuscript now rests on scripts 23 to 30 and a
second verification script. The manuscript's footnote promises "two
verification scripts". Only one is described anywhere in the repository.

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── .gitignore
├── requirements.txt
├── requirements-lock.txt
│
├── PREREGISTRATION_switchpoint.md
├── PREREGISTRATION_streaming_delayed_label.md
├── ADDENDUM_partition_threshold.md
├── ADDENDUM_predrift_labels.md                      # NEW to the layout
├── ADDENDUM_source_day_sweep.md                     # NEW to the layout
├── ADDENDUM_replacement_fraction.md                 # NEW, does not yet exist
│
├── scripts/
│   ├── tta_guards.py
│   ├── count_params.py                              # NEW to the layout
│   │
│   │   # v1 to v3 pipeline
│   ├── 02_errorbars.py
│   ├── 03_mechanism_errorbars.py
│   ├── 04_bnstats_control.py
│   ├── 05_collapse_check.py
│   ├── 06_inperiod_reference.py
│   ├── 07_w45_depth_probe.py
│   ├── 08_leakage_demo.py
│   ├── 09_hybrid_schedule.py
│   ├── 10_oracle_ceiling.py
│   ├── 11_switchpoint_probe.py
│   ├── 11_switchpoint_select.py
│   ├── 12_filtered100_errorbars.py
│   │
│   │   # v4: experiments A to E, post-hoc analysis, verification
│   ├── 13_window_trend_analysis.py
│   ├── 14_stream_order_audit.py
│   ├── 15_acrossday_replication.py
│   ├── 16_streaming.py
│   ├── 17_delayed_label.py
│   ├── 18_nondrifted_control.py
│   ├── 19_w46_stability_reference.py
│   ├── 20_delayed_label_partition.py
│   ├── 21_verify_all.py
│   │
│   │   # v6: partition threshold robustness
│   ├── 22_threshold_sweep.py                        # NEW to the layout
│   │
│   │   # v7: the work behind the current manuscript
│   ├── 23_source_pool_characterization.py           # NEW
│   ├── 24_momentum_displacement.py                  # NEW
│   ├── 25_prevalence_sweep.py                       # NEW
│   ├── 26_inference_cost.py                         # NEW
│   ├── 27_perclass_momentum.py                      # NEW
│   ├── 28_verify_revision.py                        # NEW, second verifier
│   ├── 29_recalibration_control.py                  # NEW
│   └── 30_steps_replacement.py                      # NEW
│
├── legacy/
│   └── (state-mutating-probe versions; do not use)
│
├── diagnostics/
│   ├── diag_loader_determinism.py
│   └── (determinism audits, repro-unit diagnostics)
│
├── results/
│   ├── RESULTS.md
│   ├── raw/
│   │   │   # v1 to v3
│   │   ├── inperiod_reference.json
│   │   ├── errorbars_progress.json
│   │   ├── bnstats_progress_steps50.json
│   │   ├── mechanism_progress_steps50.json
│   │   ├── filtered100_progress.json
│   │   ├── collapse_check_q0.5_steps50.json
│   │   ├── w45_depth_probe.json
│   │   ├── leakage_demo.json
│   │   ├── oracle_matched_progress.json
│   │   ├── hybrid_progress.json
│   │   ├── switchpoint_select.json
│   │   ├── switchpoint_probe.json
│   │   ├── determinism_audit_run.txt
│   │   │
│   │   │   # v4
│   │   ├── streaming_order_audit.json
│   │   ├── streaming_config.json
│   │   ├── streaming_results.json
│   │   ├── delayed_label_config.json
│   │   ├── delayed_label_progress.json
│   │   ├── nondrifted_c1_progress.json
│   │   ├── nondrifted_c2_progress.json
│   │   ├── class_partition.json
│   │   ├── class_partition.sha256
│   │   ├── w46_stability_reference.json
│   │   ├── acrossday_progress.json
│   │   ├── delayed_label_partition_progress.json
│   │   │
│   │   │   # v5 and v6
│   │   ├── params_count.json                        # NEW to the layout
│   │   ├── threshold_sweep.json                     # NEW to the layout
│   │   │
│   │   │   # v7
│   │   ├── predrift_label_progress.json             # NEW
│   │   ├── delayed_label_partition_K3_progress.json # NEW
│   │   ├── source_pool_characterization.json        # NEW
│   │   ├── momentum_displacement_progress.json      # NEW
│   │   ├── prevalence_sweep_progress.json           # NEW
│   │   ├── prevalence_sweep.json                    # NEW
│   │   ├── perclass_momentum_progress.json          # NEW
│   │   ├── perclass_momentum.json                   # NEW
│   │   ├── steps_replacement_progress.json          # NEW
│   │   ├── steps_replacement.json                   # NEW
│   │   ├── recalibration_control.json               # NEW
│   │   ├── inference_cost.json                      # NEW
│   │   └── (console logs, matching names, .txt)
│   │
│   └── superseded/
│
└── (not tracked, see .gitignore)
    ├── data/
    ├── models/
    └── manuscript/
```

## `.gitignore`

The README references a `.gitignore` that the layout never lists. Commit it:

```gitignore
data/
models/
manuscript/
__pycache__/
*.pyc
tent-env/
.venv/
*.aux
*.log
*.out
*.bbl
*.blg
```

## What each new script produced, for the raw-artifact table

| Script | Artifact | What it supports in the manuscript |
|---|---|---|
| 23 | `source_pool_characterization.json` | per-day frozen accuracy and entropy; the drift-onset dates in Section II-A |
| 24 | `momentum_displacement_progress.json` | Delta_A and Delta_U per condition and momentum; most of Table II |
| 25 | `prevalence_sweep_progress.json`, `prevalence_sweep.json` | the measured break-even, Section III-C |
| 26 | `inference_cost.json` | the 3.17x latency ratio and throughput, Section III-D |
| 27 | `perclass_momentum_progress.json`, `perclass_momentum.json` | the service column of Table II and all of Fig. 2 |
| 28 | (no artifact; reads the others) | the second verification script |
| 29 | `recalibration_control.json` | the pre-drift recalibration control |
| 30 | `steps_replacement_progress.json`, `steps_replacement.json` | the two fixed-rho pairs |

## Commit sequence

The ordering claims in the record are only as good as the history that backs
them, so commit in this order and do not squash.

1. **Resolve the pre-registration lock first** (see `REPO_MD_CHANGES.md`,
   item P-1). Nothing else should be committed until Section 10 of
   `PREREGISTRATION_streaming_delayed_label.md` is either filled from the
   existing history or its status corrected.
2. Scripts 23 to 30, with no artifacts. A reader can then see the code that
   produced the records before the records appear.
3. `ADDENDUM_replacement_fraction.md`, if you write it (item A-3).
4. The v7 artifacts under `results/raw/`.
5. `RESULTS.md` v7, `CHANGELOG.md` v7, `README.md`.
6. Tag and release, so Zenodo mints a version DOI under the existing concept
   DOI.

## Two mechanical checks before pushing

```bash
# every artifact named in RESULTS.md exists
grep -o '`[a-z0-9_]*\.json`' results/RESULTS.md | tr -d '`' | sort -u | \
  while read f; do [ -f "results/raw/$f" ] || echo "MISSING: $f"; done

# both verifiers pass
python scripts/21_verify_all.py
python scripts/28_verify_revision.py
```
