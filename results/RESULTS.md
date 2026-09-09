# RESULTS: Label-Free Test-Time Adaptation Under a Documented Drift Event in
# Encrypted QUIC Traffic Classification

Version 5. This file supersedes all previous versions. Every number in
Sections 1 to 17 was produced by the pipeline in one pinned environment
(`requirements-lock.txt`: Python 3.12, torch 2.12.1, numpy 2.5.0,
scikit-learn 1.9.0, cesnet-datazoo 0.2.0, cesnet-models 0.4.1; CPU), and
every number is regenerated from the released raw artifacts by
`scripts/21_verify_all.py`, which reports 189 checks passed, 0 failed, 0
artifacts missing. Section 18 records what that script does not cover.

**What changed in v4.** Five new experiment families (A, B, C, D, E) and one
post-hoc analysis were run under
`PREREGISTRATION_streaming_delayed_label.md`. Their outcome changes the
study's conclusion. The headline recovery of +3.06 points is now known to be
the net of a large gain on drifted traffic and a large loss on traffic that
did not drift; supervised retraining on labels up to a week old recovers four
to five times more; and the pre-registered kill rule B-K1 fired, which
obliges the framing change recorded in Section 13. Several statements in v3
are corrected in Section 7.

**What changed in v5.** No experiment was run and no measured value changed.
Four descriptions that were true but incomplete are made precise, and three
facts derived from released artifacts are added: the parameter counts of the
three retraining capacities (Section 13), the definition used for the
break-even prevalence (Section 14.4), and the sensitivity of the class
partition to its threshold (Section 14.1). Section 7.8 records the
clarifications. The additions are **not** covered by
`scripts/21_verify_all.py`; Section 18 says so.

---

## 0. Protocol summary

- **Dataset / model:** CESNET-QUIC22 (size S, `ALL_KNOWN`, 102 classes);
  MM-CESNET-V2 pretrained on W-2022-44 (published weights and transforms).
- **Windows:** the test loader yields 2048-flow batches (`test_batch_size`
  defaults to 2048; the `batch_size=256` in the scripts applies to train
  loaders only). An evaluation window is 200 consecutive batches, that is
  409,600 flows; three disjoint windows per week at batch offsets 0, 200, 400.
- **Leakage-clean tuning:** all hyperparameters selected on W-2022-46, then
  frozen. Episodic config: **lr 1e-3, 50 steps, entropy quantile 0.5**.
- **Adaptation (TENT-style):** BN affine parameters only; BN modules in train
  mode (momentum 0.1) so running statistics also update; Adam; per-batch
  entropy-quantile filtering; episodic and transductive unless stated.
  Evaluation exactly once, after the final step.
- **Error bars:** the pipeline is bit-deterministic within a process **except
  for the full-network fine-tune condition** (Section 7.6). "Seeds" vary the
  adaptation batch ordering, `numpy.random.default_rng(1000*window + k)`.
- **State auditing:** every evaluation in scripts 02 to 21 is wrapped in
  `guarded_eval` (strict), which proves the model came back unchanged, and
  recorded bit-level anchors are asserted at runtime.

## 1. Drift magnitude and the gap denominator

| Week | Per-window accuracy | Mean |
|---|---|---|
| W-45 (early week, pre-onset) | 0.9559 / 0.9507 / 0.9590 | **0.9552 ± 0.0034** |
| W-46 | 0.7233 / 0.7253 / 0.7451 | 0.7312 ± 0.0099 |
| W-47 (report week) | 0.7224 / 0.7243 / 0.7395 | **0.7287** |

**Gap = 22.65 points**, both terms self-measured under the identical
protocol. Recovery in points is primary; percent-of-gap is secondary.

## 2. Episodic decomposition at the frozen configuration (50 steps)

| Condition | Recovery | % of gap | Per-window | K |
|---|---|---|---|---|
| BN-statistics recalibration only | **+2.43 ± 0.15** | 10.7% | +2.64 / +2.35 / +2.31 | 3 |
| + filtered entropy gradients (q=0.5) | **+3.06 ± 0.27** | 13.5% | +3.42 / +2.95 / +2.81 | 5 |
| + unfiltered entropy gradients (q=1.0) | **+1.32 ± 0.53** | 5.8% | +1.99 / +1.18 / +0.78 | 3 |

Recalibration alone accounts for most of the label-free recovery. Filtered
gradients add +0.62 points over recalibration; unfiltered gradients subtract
1.12. **Section 14 shows what that gradient term is actually doing, and it is
not what v3 claimed.**

## 3. Step-count sensitivity

| Condition | 50 steps | 100 steps |
|---|---|---|
| Stats-only | +2.43 ± 0.15 | +2.23 ± 0.15 |
| Filtered (q=0.5) | +3.06 ± 0.27 | +2.96 ± 0.27 |
| Unfiltered (q=1.0) | +1.32 ± 0.53 | +0.69 ± 0.37 |

Recalibration saturates by step 50; prolonged entropy minimization on moving
statistics buys nothing or harms.

## 4. No prediction collapse

All 102 classes remain predicted after adaptation, macro-F1 is flat (mean
-0.09 points on a ~0.80 base), and prediction-distribution entropy rises
slightly. Accuracy is not being bought by collapsing onto head classes. This
diagnostic is necessary but, as Section 14 shows, far from sufficient: a
method can pass every collapse check while destroying accuracy on a large
identifiable subset of traffic.

## 5. Where the report windows actually are (A0, script 14)

Measured directly against `TIME_FIRST` and `TIME_LAST`, with no model loaded.

- W-2022-46 holds 5,044,543 known-class flows in 2,464 batches; W-2022-47
  holds 6,607,244 in 3,227. Both span exactly 168.0 hours. The `n_avail=650`
  in earlier consoles is the cap passed to `count_available_batches`, not a
  measurement.
- **The three report windows are one day, not one week.** Batches 0 to 599 of
  W-2022-47 fall entirely inside 20221121, covering 99.4% of that day and
  18.6% of the week. Because the windows hold equal flow counts and traffic
  volume varies over the day, they span 11.19, 4.46 and 8.12 hours
  respectively, covering Monday 21 November almost end to end.
- **Stream ordering.** The stream is approximately ordered by flow export
  time with local jitter. Per-batch median time is non-decreasing across
  every consecutive batch pair in both periods (zero inversions).
  Displacement under a stable global sort by `TIME_LAST` is median 38 and 45
  positions, 99.9th percentile 1,574 and 1,721, maximum 2,428 and 2,414,
  against a batch of 2,048. Flow-level non-decreasing fractions are 0.5047
  and 0.5036 on `TIME_FIRST`, 0.5075 and 0.5060 on `TIME_LAST`; a fraction
  near one half is the signature of small symmetric jitter and does not
  distinguish light jitter from a shuffle, which is why displacement is the
  reported criterion. Batch composition is what a globally export-sorted
  stream would give.

## 6. Experiment A: streaming, strictly causal (script 16)

Configuration selected on W-2022-46 only (lr 1e-4, q 0.5) and frozen before
W-2022-47 was touched. In the causal conditions the prediction for batch b
uses only information from batches strictly before b.

| Condition | w1 | w2 | w3 | 3-window | full week |
|---|---|---|---|---|---|
| frozen | .7224 | .7243 | .7395 | .7287 | .7363 |
| causal-filtered | .7538 | .7539 | .7649 | .7575 | .7625 |
| causal-stats | .7499 | .7463 | .7608 | .7523 | n/a |
| batch-transductive filtered | .7494 | .7532 | .7641 | .7556 | n/a |
| reset every 200 batches | .7538 | .7527 | .7656 | .7573 | n/a |

Recovery in points: causal-filtered **+2.88** on the three windows and
**+2.63** over the full week of 6,607,244 flows, against the episodic
**+3.06**.

1. **Strict causality costs 0.18 points, about 6%.** The transductive
   protocol holds 409,600 flows in memory before classifying any of them,
   which Section 5 shows is 4.5 to 11.2 hours of traffic. That buffering buys
   almost nothing.
2. **Buffering one batch has negative value.** Batch-transductive
   normalization scores 0.20 points below strictly causal.
3. **Continuous adaptation does not collapse over a week.** Running accuracy
   rises monotonically across 3,227 batches and every one of the seven days
   is positive. Resetting every 200 batches changes nothing (+2.86 vs +2.88).

Streaming has exactly one arrival order, so it has no ordering error bars;
the reported spread is across windows and days.

## 7. Correction record

### 7.1 to 7.5 (unchanged from v3)
The state-mutating accuracy probe, the corrupted hyperparameter selection,
the retracted ordering claim, the superseded denominator, and the
clean-but-stale results are recorded as in v3 and are not restated here.

### 7.6 Determinism has one exception (new in v4)
The claim that the pipeline is bit-deterministic within a process holds for
every condition except the **full-network fine-tune** of Experiment B. That
condition is the only one that calls `m.train()` on the whole model, which
activates the architecture's three dropout modules
(`cnn_global_pooling.3`, `mlp_flowstats.8`, `mlp_shared.3`). Dropout draws
from torch's global RNG, which the pipeline does not seed. Since v5 that
module listing is recorded in `params_count.json` and checked by
`scripts/21_verify_all.py`, so this exception rests on the architecture
rather than on a one-time inspection. Two independent
runs of that condition (scripts 17 and 20) differ by 0.014 to 0.133 points
against a 14-point effect. The recorded values are **not** revised; both runs
are released and the pair is reported as an accidental independent
replication. Every other condition is bit-identical across the two runs.

### 7.7 Statements from v3 that are now wrong (new in v4)
- The matched-capacity labeled oracle was described as "an unattainable upper
  bound". It is neither. Delayed-label retraining at one-day-old labels
  exceeds it (Section 13). It is a matched-capacity, matched-step supervised
  reference and nothing more.
- The three evaluation windows were described as slices of a single week
  demonstrating within-week stability. They are slices of a single day
  (Section 5). Experiment E supplies the across-day evidence.
- Entropy filtering was described as gating the sign of the gradient
  contribution. Section 14 shows filtering makes recovery on drifted traffic
  slightly worse and that its entire benefit is limiting damage to traffic
  that did not drift.
- The "roughly four fifths" attribution to recalibration is one of three
  values: 79% on the recorded single day, 73% across Experiment E's seven
  days, 82% in streaming. Report the range or name the experiment.
- The W-2022-46 tuning figure of +2.68 is a single natural-order draw. The
  K=5 mean on the identical window is +2.42 (Section 11), and the natural
  order sits outside the shuffled range.
- The two-phase standard deviation is recorded in v3 Section 6 as 0.49, which
  is the ddof=1 value; every other standard deviation in this file is ddof=0.
  The ddof=0 value is **0.46**. This file uses 0.46.

### 7.8 Descriptions from v4 that were incomplete (new in v5)
No measured value changes here. Each item is a description that was true as
far as it went and misleading if read closely.

- The `head` capacity was described as "BN stats frozen". It retrains the
  final classification layer with the backbone in evaluation mode and
  touches no normalization at all, neither the affine parameters nor the
  statistics. Section 13 now states what each capacity updates, with
  parameter counts.
- Section 15 attributed the +0.11 on unaffected traffic to "freezing the
  statistics". The correct statement is that `head` never moves them.
  Consequently `head` versus `matched` is a two-variable contrast, and the
  single-variable isolation of the statistics is `src-stats` versus frozen.
- Section 8 item 3 said the post-hoc partition units "bit-reproduce
  Experiment B". They reproduce the corresponding k=0 units. The Section 13
  entries for the K=3 conditions are means and differ by up to 0.48 points.
- Section 14.4 gave a break-even prevalence without saying which of three
  reasonable definitions produced it. It is the mean of the per-window
  values.

## 8. Provenance chain

1. Frozen W-47 accuracies are bit-identical across all six Table I and II
   artifacts, across Experiments A, B, C and E, and across the post-hoc
   analysis.
2. Experiment E's 20221121 units use the same seeds as Table I window 1 and
   bit-reproduce it: +3.405 / +3.460 / +3.332.
3. Each post-hoc partition unit bit-reproduces the **corresponding k=0
   unit** of Experiment B for every condition except the full fine-tune
   (Section 7.6). It does not reproduce the Section 13 table entries for
   `head`, `matched`, `src-stats` and `src-tent`, because those are K=3
   means while the partition run holds only k=0; the two differ by up to
   0.48 points (`src-stats`, delta=7). This is bookkeeping, not
   disagreement.
4. The class partition reconstructs Table I exactly from its two parts
   (Section 14).
5. `PREREGISTRATION_switchpoint.md` was hash-locked before its run.
   `PREREGISTRATION_streaming_delayed_label.md` was hash-locked before
   Experiments B, C and D but after A and E; its Section 0.1 records that
   asymmetry, and the manuscript uses the weaker wording for A and E.
6. `scripts/21_verify_all.py` regenerates every number above from the
   released artifacts: 189 checks, 0 failures.

## 9. Matched-capacity labeled reference

BN-affine fine-tune with ground-truth cross-entropy on the same windows, 50
steps, K=3: **+11.55 ± 0.38** (per-window +12.02 / +11.52 / +11.10), 51.0% of
the gap. Reported as a matched-capacity reference, **not** as an upper bound
(Section 7.7).

## 10. Pre-registered switch-point selection: NEGATIVE

Hash-locked before the run
(SHA-256 `4ebd14fbe8b721e9bb86683febd724b4ba7b08fb8a2d7c83a7444c37f9712e45`).

| Switch | Mean | Order-std | Per-ordering |
|---|---|---|---|
| **25** | **+5.11** | **0.27** | +4.71 / +4.86 / +5.38 / +5.24 / +5.37 |
| 37 | +3.98 | 0.30 | +4.40 / +3.95 / +3.47 / +3.95 / +4.11 |
| 50 | +3.46 | 0.18 | +3.69 / +3.62 / +3.20 / +3.34 / +3.43 |
| 62 | +2.45 | 1.88 | **-1.28** / +3.37 / +3.10 / +3.31 / +3.75 |
| 75 | +2.23 | 1.72 | +2.39 / **-1.13** / +3.21 / +3.27 / +3.40 |

Kill rule A fired at selection time (0.27 > 0.12 ceiling); the W-47
confirmatory was never run. The two-phase schedule is a rejected candidate.

## 11. Experiment D: matched stability reference (script 19)

Pure filtered adaptation on the identical W-2022-46 60-batch window with the
identical K=5 seeds: **+2.42, order-std 0.09** (+2.47 / +2.40 / +2.39 /
+2.57 / +2.29).

The 0.12 ceiling in Section 10 was set at twice 0.06, a value measured on
W-2022-47 200-batch windows, while the switch-point standard deviations were
measured on this 60-batch window. Those are not comparable: with 50 steps on
60 batches an ordering consumes 50 of 60, while on 200 batches it consumes 50
of 200. **A matched ceiling would have been 0.18, not 0.12.** The selected
switch point's 0.27 still exceeds 0.18, so **the rejection survives its own
correction**. The recorded outcome of Section 10 is not revised, because the
selection was executed as pre-registered; only the calibration argument is
corrected, and the rejection is re-justified on the evidence that does not
depend on the ceiling: single orderings at switch points 62 and 75 drove
accuracy below the frozen baseline.

## 12. Experiment E: across-day replication (script 15)

Frozen configuration, one 200-batch window per day of W-2022-47, K=3, no
tuning. The 20221126 window extends 18 batches into 20221127 because that day
holds only 182 batches; this was pre-registered and is flagged.

| Day | Frozen | Stats | Filtered |
|---|---|---|---|
| 20221121 | 0.7224 | +2.64 ± 0.02 | +3.40 ± 0.05 |
| 20221122 | 0.7253 | +2.44 ± 0.02 | +3.29 ± 0.12 |
| 20221123 | 0.7296 | +2.18 ± 0.12 | +3.15 ± 0.10 |
| 20221124 | 0.7389 | +1.77 ± 0.16 | +2.97 ± 0.08 |
| 20221125 | 0.7301 | +2.10 ± 0.08 | +2.99 ± 0.21 |
| 20221126 | 0.7597 | +2.16 ± 0.04 | +2.69 ± 0.03 |
| 20221127 | 0.7511 | +2.23 ± 0.04 | +2.79 ± 0.10 |

**Filtered: +3.04 ± 0.24 across seven days**, against the single-day headline
+3.06. Stats-only: +2.22 ± 0.25. No day falls below the frozen baseline. The
across-day standard deviation (0.24) is essentially the same as the
within-day across-window figure (0.26), so the recorded error bar was the
right size while being the wrong quantity.

Limitation: all seven windows are anchored at each day's first batch and
therefore sample the same part of the diurnal cycle. The per-day streaming
figures in Section 6 cover whole days and complement this.

## 13. Experiment B: delayed-label retraining (script 17). B-K1 FIRED.

Evaluation on the three report windows of 20221121. For a delay of delta
days, the retraining data is the labelled traffic of (evaluation day minus
delta): 20221120, 20221118 and 20221114 respectively, **all inside the tuning
week. No W-2022-47 label ever enters training.** All configurations tuned
inside W-2022-46 and frozen before W-2022-47 was touched.

**What each capacity actually updates** (counts from `scripts/count_params.py`
on the released weights; model total 2,261,653 parameters):

| Capacity | Updated by gradients | Parameters | BN running statistics |
|---|---|---|---|
| `head` | the final `Linear` module, named `classifier`, backbone in `eval()` | 61,302 (2.71%) | **not touched at all** |
| `matched` | BN affine `{gamma, beta}`, 12 BN modules | 6,400 (0.28%) | move (momentum 0.1) |
| `full` | every parameter | 2,261,653 (100%) | move (momentum 0.1) |

`head` is **not** a BN-affine condition with frozen statistics. It changes no
normalization of any kind; the backbone stays in evaluation mode. `matched`
is the capacity-matched comparison because it is exactly the parameter set
TTA adapts. The 6,412 BN running buffers are 2 x 3,200 channel statistics
plus 12 `num_batches_tracked` counters; they are moved by `matched` and
`full` but never gradient-updated.

Note the inversion: `matched` recovers more overall (+13.00) than `head`
(+11.19) while updating one tenth as many parameters, and it is the one that
damages the classes that did not drift.

| Condition | d=1 | d=3 | d=7 | mean | % of gap |
|---|---|---|---|---|---|
| src-stats (no labels) | +2.21 | +2.74 | +2.16 | +2.37 | 10.5% |
| src-tent (no labels) | +2.84 | +2.92 | +2.66 | +2.80 | 12.4% |
| head (labels, classifier layer only) | +11.07 | +11.28 | +11.21 | +11.19 | 49.4% |
| matched (labels, BN affine + stats) | +13.55 | +13.43 | +12.04 | +13.00 | 57.4% |
| full (labels, all parameters) | +14.20 | +14.72 | +13.51 | +14.14 | 62.4% |
| matched + TTA | +12.11 | +11.95 | +10.77 | +11.61 | 51.3% |
| head + TTA | +8.42 | +8.21 | +8.12 | +8.25 | 36.4% |
| full + TTA | +13.52 | +13.50 | +12.54 | +13.19 | 58.2% |

**B-K1 fired**, at every delta, by 11.66 points against a 2.00-point
threshold. Per the pre-registration the paper does not present label-free
adaptation as an operational recommendation.

1. **There is no crossover delta.** The pre-registration asked for one. At
   every delay the dataset can test, the worst supervised baseline (+11.07)
   exceeds the best label-free result anywhere in this study (+3.06) by more
   than three times. The absence is the finding.
2. **Label age costs almost nothing.** One-day-old to week-old labels costs
   1.51 points for matched, 0.69 for full, and nothing for head. The
   operational quantity is not a rolling delay but the interval between drift
   onset and the arrival of the first post-drift labels.
3. **The label-free ceiling is the mechanism, not the data.** Giving the
   label-free method a full day of recent unlabelled traffic instead of the
   current window changes nothing (+2.37 and +2.80 against +2.43 and +3.06).
4. **Stacking adaptation on a retrained model always hurts.** B-K4 fired nine
   times out of nine. Section 14 gives the mechanism.

Reported limitations: all supervised capacities selected steps=100, the grid
maximum, and none had turned over, so these are a **lower bound** on the
baselines. The label-free configuration was tuned on a grid that offered 50
and 100 steps and chose 50 because 100 was worse, so the step budget is not
an unfair advantage. Delta is confounded with source day-of-week: 20221120 is
a Sunday, 20221118 a Friday, 20221114 the previous Monday, and the Sunday
pool held 198 batches rather than 200.

## 14. Experiment C: what adaptation costs traffic that did not drift

### 14.1 The class partition (hash-locked before any report-week number)

Built from W-2022-45 and W-2022-46 only: a class is affected if its frozen
per-class recall on W-2022-46 is more than 10 points below its recall on the
pre-shift windows of W-2022-45. **29 affected classes, 73 unaffected, 1
undetermined** (`_unknown`, zero support). No minimum support was
pre-registered and none is applied; per-class support is released so thin
evidence is visible, and three affected classes rest on fewer than 100 flows
in one period.

The rule referenced no provider names and returned a set of which **21 of 29
are Google services** (`google-fonts` 0.963 to 0.184, `google-gstatic` 0.955
to 0.206, `google-www` 0.968 to 0.385, `youtube` 0.948 to 0.686). This is
independent, data-driven corroboration of the documented certificate change.
Three non-Google classes also show large drops and are noted rather than
explained: `dns-doh` (0.994 to 0.592 on 37,000 flows), `garmin` (0.926 to
0.251), `adavoid` (0.896 to 0.246).

**Threshold sensitivity (added in v5, from `class_partition.json`).** The
10-point cut was pre-registered, and it does not fall in an empty region of
the drop distribution. Loosening it barely matters: at 5 points the affected
set grows to 34 classes but the affected share of W-2022-46 flows moves only
from 60.7% to 60.9%, because the five added classes carry 2,680 flows
between them. Tightening it matters a great deal: at 15 points the set falls
to 21 classes and the affected share falls to 47.9%, and at 20 points to 18
classes and 41.5%. The swing is dominated by one class, `google-ads`, which
drops 10.6 points, just over the cut, and carries 94,577 flows, 7.7% of the
week. **The decomposition has not been recomputed at any other threshold.**
Doing so is the natural robustness check and would have to be declared
post-hoc.

### 14.2 The headline is a net of two large opposing effects

| Window | Condition | Affected | Unaffected | Net | Recorded |
|---|---|---|---|---|---|
| 1 | stats | +7.45 | -5.50 | +2.64 | +2.64 |
| 1 | filtered | +7.22 | -3.07 | +3.40 | +3.42 |
| 2 | stats | +6.76 | -4.64 | +2.35 | +2.35 |
| 2 | filtered | +6.39 | -2.50 | +2.95 | +2.95 |
| 3 | stats | +7.33 | -4.52 | +2.31 | +2.31 |
| 3 | filtered | +6.99 | -2.80 | +2.85 | +2.81 |

Affected traffic is 62.9%, 61.3% and 57.6% of the three windows. The
support-weighted combination reconstructs Table I exactly. **Adaptation gains
about seven points on drifted traffic and destroys 2.5 to 5.5 points on
traffic that did not drift. The published +3.06 is what is left after those
cancel.** No collapse check in Section 4 detects this.

### 14.3 Entropy filtering is damage limitation, not recovery

Splitting the gradient term by partition:

| Window | On affected | On unaffected | Net |
|---|---|---|---|
| 1 | -0.23 | +2.43 | +0.76 |
| 2 | -0.36 | +2.14 | +0.61 |
| 3 | -0.33 | +1.72 | +0.53 |
| mean | **-0.31** | **+2.10** | **+0.63** |

Filtering makes recovery on drifted traffic slightly worse in all three
windows. Its entire benefit is limiting collateral damage.

### 14.4 Break-even drift prevalence

Solving f* = -d_unaffected / (d_affected - d_unaffected) for the affected
fraction at which adaptation nets zero, **per window, then averaging the
three**: **28.8% for filtered, 40.4% for stats-only.** The per-window values
are 29.8 / 28.1 / 28.6 and 42.5 / 40.7 / 38.1. The definition matters at the
second decimal: solving once from the pooled means gives 28.9% and 40.5%,
and support-weighting gives 28.9% and 40.4%. This file uses the mean of the
per-window values throughout. Below roughly 29% drifted traffic, filtered
adaptation makes the classifier worse overall. These windows sit at 58 to
63%. An operator cannot measure that fraction without the labels the method
exists to avoid needing.

### 14.5 This also explains the per-window trend

The affected fraction falls across the three windows (62.9, 61.3, 57.6)
because they are different times of one Monday. Frozen accuracy rises and
recovery falls in step. The per-window pattern in Table I is diurnal
composition of drifted traffic, measured rather than inferred.

### 14.6 Period-level control (C1)

Adaptation applied to periods where nothing drifted:

| Period | Window | Frozen | Stats | Filtered |
|---|---|---|---|---|
| W-2022-45 | 1 | 0.9559 | -0.09 ± 0.00 | -0.32 ± 0.06 |
| W-2022-45 | 2 | 0.9507 | -0.07 ± 0.01 | -0.29 ± 0.01 |
| W-2022-45 | 3 | 0.9590 | -0.04 ± 0.01 | -0.30 ± 0.01 |
| W-2022-44 | 1 | 0.9497 | -0.26 ± 0.04 | -0.28 ± 0.01 |
| W-2022-44 | 2 | 0.9548 | -0.54 ± 0.29 | -0.55 ± 0.31 |
| W-2022-44 | 3 | 0.9617 | -0.07 ± 0.02 | -0.30 ± 0.05 |

W-2022-44 is the week the public weights were trained on, so its test flows
overlap the model's training data and its frozen accuracy is inflated by
memorization; it is reported with that caveat and is not a clean control.

**C1 and C2 answer different questions and must be reported as such.** When
the whole window is clean, adaptation costs 0.03 to 0.32 points. When clean
and drifted traffic share the adaptation batches, the same classes lose 2.5
to 5.5 points. The second is the deployment case.

**C3 fired.** In C1 at -0.54 and -0.55, both on W-2022-44 window 2 and both
driven by one ordering (k=2, -0.95 and -0.98 against roughly -0.30 for the
other two), which is evidence that a single unrepresentative draw of
adaptation batches can cost a point even on undrifted traffic. In C2 at
-5.50. Per the pre-registration this cost is stated in the abstract.

## 15. Post-hoc: the same partition applied to the supervised conditions

**NOT PRE-REGISTERED.** Added after Section 14 made the question unavoidable.
It decomposes numbers that already exist and revises no kill-rule verdict.
K=1, declared.

| Condition | Δ affected | Δ unaffected | Labels? | BN stats |
|---|---|---|---|---|
| head (classifier only) | +18.30 | **+0.11** | yes | never touched |
| matched | +22.13 | -0.91 | yes | moved |
| full | +23.61 | -0.40 | yes | moved |
| matched + TTA | +20.25 | -1.69 | yes | moved |
| src-stats | +7.11 | -5.25 | **no** | moved |
| src-tent | +6.79 | -3.50 | **no** | moved |
| stats-only (current window) | +7.18 | -4.89 | **no** | moved |
| filtered (current window) | +6.87 | -2.79 | **no** | moved |

**The collateral damage is specific to label-free adaptation.** Supervised
retraining gains three times more on drifted traffic and does three to seven
times less damage to undrifted traffic. The condition that never moves the
statistics does no damage at all.

**The mechanism.** Moving BN statistics toward a drifted mixture is what
helps drifted classes and what harms everything else. The clean isolation is
`src-stats`, which moves the statistics with no gradients anywhere: +7.11 on
affected, -5.25 on the rest. The gradient term is a compensator for that
harm: labelled gradients on the same moving statistics nearly eliminate it
(-0.91) while tripling the gain, and entropy gradients remove about half of
it (-4.89 to -2.79) at the cost of slightly less gain. `head` never creates
the displacement in the first place, since it touches no normalization, so
there is nothing to compensate for (+0.11). This one reading accounts for the
decomposition in Section 2, the role of filtering in Section 14.3, the
label-free ceiling in Section 13, and why supervision wins.

Note that `head` versus `matched` is **not** a single-variable contrast: they
differ both in which parameters receive gradients and in whether the
statistics move. Only `src-stats` versus frozen isolates the statistics.

**And it explains B-K4.** Stacking adaptation on a retrained model costs 1.88
points on affected traffic and 0.78 on unaffected: entropy minimization pulls
the model back toward its own confident predictions, undoing the supervised
fit on exactly the classes supervision had corrected.

## 16. Reading for an operator

1. Adaptation is not free on traffic that did not drift. Below roughly 29%
   drift prevalence it is net harmful, and prevalence cannot be measured
   without labels.
2. If any labelled post-drift traffic exists, retrain. Week-old labels are
   nearly as good as day-old ones and four to five times better than
   label-free adaptation.
3. Do not stack label-free adaptation on top of a retrained model.
4. If nothing is labelled yet, statistic recalibration is the cheapest and
   lowest-variance option, and the ceiling is about three points of a
   22.65-point gap regardless of how much unlabelled traffic is supplied.

## 17. Raw artifacts

- **A0 / ordering:** `streaming_order_audit.json`
- **Experiment A:** `streaming_config.json`, `streaming_results.json`
- **Experiment B:** `delayed_label_config.json`, `delayed_label_progress.json`
- **Experiment C:** `nondrifted_c1_progress.json`, `class_partition.json`,
  `class_partition.sha256`, `nondrifted_c2_progress.json`
- **Experiment D:** `w46_stability_reference.json`
- **Experiment E:** `acrossday_progress.json`
- **Post-hoc:** `delayed_label_partition_progress.json`
- **Capacity sizes (new in v5):** `params_count.json`, written by
  `scripts/count_params.py` from the released weights
- **v2.1 and earlier (unchanged):** `errorbars_progress.json`,
  `bnstats_progress_steps50.json`, `mechanism_progress_steps50.json`,
  `filtered100_progress.json`, `oracle_matched_progress.json`,
  `hybrid_progress.json`, `switchpoint_select.json`, `switchpoint_probe.json`,
  `inperiod_reference.json`, `w45_depth_probe.json`,
  `collapse_check_q0.5_steps50.json`, `leakage_demo.json`, plus the marked
  superseded set.

## 18. Verification and its limits

`scripts/21_verify_all.py` regenerates every number in Sections 1 to 15 from
the released artifacts and compares each against the value recorded here:
**189 checks passed, 0 failed, 0 artifacts missing.** It loads no model and
reads no dataset, and runs in seconds.

The v5 additions are outside that coverage: the parameter counts in Section
13 come from `scripts/count_params.py` against the released weights, and the
threshold-sensitivity figures in Section 14.1 are computed from
`class_partition.json`. Neither is checked by `21_verify_all.py`, and adding
checks for them would be a small, worthwhile change.

A failure would mean the artifact and this file disagree; the script cannot
say which is wrong. It re-runs no experiment, so it detects an inconsistent
record and not a wrong experiment. Three artifacts are outside its coverage
and remain verified only by hand: `leakage_demo.json`,
`w45_depth_probe.json`, and `switchpoint_probe.json`.
`w45_depth_probe.json` backs the drift-onset figure, which is the only
figure in the manuscript whose numbers this script does not regenerate.
`collapse_check_q0.5_steps50.json` moved inside coverage in v5.
