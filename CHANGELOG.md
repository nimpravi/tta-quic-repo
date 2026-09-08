# CHANGELOG

A summary of what changed between versions of the experimental record, and
why. This file is a summary: the full mechanism of every correction, with the
evidence for it, is in [`results/RESULTS.md`](results/RESULTS.md) section 7.


---

## v4 (2026-09-08)

Five pre-registered experiment families, one post-hoc analysis, and a
verification pass. The study's conclusion changed.

**Added, under `PREREGISTRATION_streaming_delayed_label.md`:**

- **Experiment A, streaming.** Strictly causal, prediction before adaptation,
  no buffering. Recovers +2.88 points on the three report windows and +2.63
  over the full week of 6,607,244 flows, against the episodic +3.06. Buffering
  the whole window buys 0.18 points; buffering one batch has negative value.
- **Experiment B, delayed-label retraining.** Supervised retraining on labels
  1, 3 and 7 days old recovers +11.07 to +14.72 points. **Kill rule B-K1
  fired at every delay**, so the paper no longer presents label-free
  adaptation as an operational recommendation. There is no crossover delay
  within the range the dataset can test.
- **Experiment C, cost on non-drifted traffic.** The +3.06 headline is the net
  of a gain of about +6.9 points on drifted classes and a loss of 2.5 to 5.5
  points on classes that did not drift. Below roughly 29 percent drift
  prevalence, adaptation is net harmful. Kill rule C3 fired.
- **Experiment D, matched stability reference.** The 0.12 ordering-stability
  ceiling used in the v3 switch-point selection was calibrated against a
  reference measured on a different window size. A matched ceiling would have
  been 0.18. The rejection survives its own correction.
- **Experiment E, across-day replication.** The headline holds across seven
  days: +3.04 ± 0.24 against the single-day +3.06.
- **Post-hoc partition analysis**, labeled as not pre-registered: the
  collateral cost on non-drifted traffic is specific to label-free
  adaptation. Supervised retraining gains three times more on drifted traffic
  and does three to seven times less damage.

**Added to the repository:**

- Scripts 13 to 21, including `21_verify_all.py`, which regenerates every
  number in the paper from the released artifacts and reports 145 checks
  passed, 0 failed, 0 artifacts missing. It needs no dataset, model or GPU.
- `requirements-lock.txt`. The README, `RESULTS.md`, the manuscript and the
  cover letter had all referenced a pinned environment file that did not
  exist, while `requirements.txt` left `cesnet-datazoo` and `cesnet-models`
  unconstrained. Those two govern which flows are selected and how, so the
  environment was not in fact pinned. It is now.
- `class_partition.json` and its SHA-256, locked before any report-week
  per-partition number was computed.

**Fixed in the recorded pipeline:**

- `tta_guards.py` is now imported and used by scripts 02 to 12. It had
  existed since v2 but was called by nothing, while the manuscript claimed
  every evaluation was wrapped in it. Verified by rerunning script 06 and
  confirming its output bit-identical to the record.
- `06_inperiod_reference.py` printed the retracted +4.34 headline.
- `07_w45_depth_probe.py` annotated flow counts with the train batch size
  (256) instead of the test batch size (2048), understating every printed
  flow range by a factor of eight. Accuracies were unaffected.
- `count_available_batches()` reported its own cap as a measurement, so every
  console printed `n_avail=650` as though W-2022-47 held 650 batches. It holds
  3,227. The function now says when the cap binds.

**Statements from v3 that are now wrong** (detail in `RESULTS.md` 7.7):

- The labeled oracle was described as an unattainable upper bound. Delayed
  labels beat it. It is a matched-capacity reference.
- The three evaluation windows were described as slices of a single week.
  They are slices of a single day, 20221121, spanning 11.19, 4.46 and 8.12
  hours.
- Entropy filtering was described as gating the sign of the gradient
  contribution. It makes recovery on drifted traffic slightly worse; its
  entire benefit is limiting damage to traffic that did not drift.
- "Roughly four fifths" for the recalibration share is one of three values:
  79 percent, 73 percent or 82 percent depending on the experiment.
- The W-2022-46 tuning figure of +2.68 is a single natural-order draw; the
  K=5 mean on the identical window is +2.42.
- The two-phase standard deviation was recorded as 0.49 (ddof=1) while every
  other figure in the record is ddof=0. The ddof=0 value is 0.46.

**Known limitation of determinism**, new in v4: the pipeline is
bit-deterministic within a process for every condition except the
full-network fine-tune of Experiment B, which is the only one that puts the
whole model in training mode and so activates its three dropout layers.
Torch's global RNG is not seeded. Two independent runs of that condition
differ by 0.014 to 0.133 points against a 14-point effect; both are released
and reported as an independent replication rather than one being chosen.

---

## v3 (2026-07-11)

- **Pre-registered switch-point selection** (`PREREGISTRATION_switchpoint.md`,
  SHA-256 `4ebd14fb…712e45`, hash-locked before the run). Kill rule A fired at
  selection time on the tuning week: the selected switch point's ordering
  standard deviation was 0.27 against a pre-committed ceiling of 0.12. Per
  the pre-registration, **no W-2022-47 confirmatory run was performed**.
- The two-phase schedule was reclassified from "post-hoc observation, future
  work" to **rejected candidate**. The +4.31 figure recorded in v2.1 is a
  favorable single-ordering draw from a high-variance procedure, not a stable
  capability. Switch points 62 and 75 each contained a single ordering that
  drove accuracy below the frozen baseline.
- The coarse W-2022-46 switch-point probe from v2.1 was superseded: it used
  one natural ordering and no error bars, so it could not see the instability
  the seeded selection exposed.

## v2.1 (2026-07-05)

- **Matched-capacity labeled oracle** added: +11.55 ± 0.38, 51.0 percent of
  the gap, giving the label-free headline a supervised reference.
- Filtered adaptation at 100 steps measured with error bars, replacing a
  single-ordering table cell.
- Coarse switch-point probe on the tuning week, and a W-2022-47 depth trace
  that showed trendless scatter where W-2022-45 declines, ruling out
  within-week composition as the driver of the W-2045 trend.

## v2 (date not recorded in the repository)

The forensic audit and the corrections that followed it.

- **Found:** the pipeline's mid-trajectory accuracy probe called
  `model.eval()` and never restored training mode. For 100-step
  configurations, steps 51 onward therefore ran a different algorithm, an
  undocumented two-phase hybrid created by the measurement itself.
- **Headline retracted** from +4.34 to +3.06, and the reported percent-of-gap
  from 31.4 to 13.5.
- **Denominator corrected** from the published whole-week figure of about
  0.867 to a self-measured pre-shift reference of 0.9552, after a depth probe
  established that the whole-week value mixes traffic from before and after
  the mid-week event.
- **Hyperparameter selection corrected**: the old grid compared pure-mode
  50-step rows against hybrid-mode 100-step rows and so picked the wrong step
  count. Honest final-step tuning selects 50 steps.
- **Ordering claim retracted**: the reported contrast between natural and
  shuffled adaptation order was cross-vintage arithmetic comparing runs of two
  different algorithms. No ordering effect exists in the clean pipeline.
- Bit-level trajectory reconstruction confirmed the mechanism and showed no
  test-label leakage was involved at the final configuration.

## v1 (date not recorded in the repository)

Original results. Later found to have been produced under the state-mutating
evaluation probe described in v2, and superseded in full. The affected
artifacts are retained under `results/superseded/` with their cause attached
rather than deleted.
