# Pre-registration: streaming evaluation, delayed-label retraining baselines,
# and non-drifted-traffic controls

Project: Decomposing Label-Free Test-Time Adaptation for Encrypted QUIC
Traffic Classification Under Abrupt Temporal Drift.

Status: DRAFT, not yet locked. This document becomes binding only when it is
committed to the repository and its SHA-256 is recorded in `RESULTS.md`.
Nothing below may be edited after that commit. Any change requires a new
pre-registration document with its own hash, and both remain in the record.

Author: Praveen Hegde.
Date drafted: [FILL BEFORE LOCKING].
Supersedes nothing. Extends the protocol fixed in
`PREREGISTRATION_switchpoint.md`
(SHA-256 `4ebd14fbe8b721e9bb86683febd724b4ba7b08fb8a2d7c83a7444c37f9712e45`).

---

## 0. Scope and the single governing rule

Five experiment families are pre-registered here:

- **A. Streaming (chronological-order) evaluation.** What the method delivers
  when flows are classified as they arrive, with no buffering of the future.
- **B. Delayed-label retraining baselines.** What supervised retraining
  delivers when labels arrive N days late, which is the alternative a network
  operator actually has.
- **C. Non-drifted-traffic controls.** What adaptation costs when nothing has
  drifted.
- **D. Matched stability reference on the tuning week.** A repair to the
  reference value behind the stability ceiling used in the switch-point
  selection.
- **E. Across-day replication.** Seven days instead of one, at the already
  frozen configuration.

The governing rule is unchanged and absolute: **every hyperparameter, every
selection, and every kill rule is fixed on W-2022-46 (tuning) or earlier, and
W-2022-47 (report) is consulted exactly once per registered condition, after
this document is hash-locked.** There is no discretionary choice left for
after the report-week numbers exist. Where a result could be read two ways,
the reading is written down here, in advance, in Section 6.

## 0.1 Provenance of this document, and what it can and cannot claim

This section exists because the honest value of a pre-registration is exactly
the verifiable ordering between writing it and running the experiments, and
that ordering is not uniform across the families below. It is recorded here
rather than left for a reader to reconstruct.

**Written before the runs, committed after.** The designs for Experiments A
and E were written before those experiments were executed, but this document
was not committed and hash-locked until after they had run. There is
therefore no third-party-verifiable timestamp establishing that the design
preceded the results for those two families. The claim available for them is
"designed in advance and reported in full, including the criterion that
failed", not "hash-locked before the report week was consulted". The
manuscript uses the weaker wording for Experiments A and E. The stronger
claim remains available only for the switch-point selection in
`PREREGISTRATION_switchpoint.md`, which was hash-locked before its run.

**Genuinely prospective at lock time.** At the moment this document is
committed, Experiment B has produced no result of any kind on W-2022-47 and
no tuned configuration: only two five-step plumbing smokes on W-2022-46,
neither of which wrote a configuration file. Experiments C and D have not
been run at all. For B, C and D the hash lock is meaningful in the same sense
as the switch-point pre-registration, and kill rule B-K1 in particular is
committed before any number it could be applied to exists.

**One design change made in response to data, recorded here.** The two
label-free controls in Experiment B, B-src-stats and B-src-tent, were added
after the second plumbing smoke showed that the supervised capacities could
not be interpreted without them: a model retrained on the previous day also
recalibrates its batch-normalization statistics on that day, which costs no
labels, and the original design had no way to separate the two
contributions. The smoke that prompted this ran on W-2022-46 only, with five
gradient steps on 10,240 flows, and produced no W-2022-47 number. No kill
rule was altered, weakened, or added. The controls can only make the
supervised comparison harder to read favorably, never easier, since a large
label-free control subtracts from what the labels can be said to buy.

**What follows.** No criterion, kill rule, grid, or condition in this
document may be changed after the commit recorded in Section 10. Where a
family's ordering is weaker, the manuscript says so in the text rather than
describing the whole study with the strongest wording that applies to any
part of it.

## 1. Facts measured before locking

These were measured, not assumed. The measuring script (`14_stream_order_audit.py`)
loads no model, performs no adaptation, and consults no label, so nothing below
touched the report week in the sense the protocol cares about. Raw output:
`streaming_order_audit.json`.

1. CESNET-QUIC22 covers 31 October to 27 November 2022 and contains exactly
   four weekly periods, W-2022-44 through W-2022-47. There is no data after
   W-2022-47. A delayed-label design needing "labels that arrive after the
   report week" is impossible and is not used. The designs below take labels
   from the past relative to the traffic being classified.
2. `DatasetConfig` accepts an explicit `test_dates` list checked against
   `dataset.available_dates`, so periods can be defined at single-day
   granularity. W-2022-47 is 20221121 to 20221127; W-2022-46 is 20221114 to
   20221120.
3. `test_batch_size` defaults to 2048 and is not overridden anywhere in the
   pipeline. A batch is 2048 flows; a 200-batch window is 409,600 flows.
   Measured batch sizes are 2048 throughout with one short final batch per
   period (319 in W-2022-46, 396 in W-2022-47).
4. **True period lengths.** W-2022-46 holds 5,044,543 known-class flows in
   2,464 batches; W-2022-47 holds 6,607,244 in 3,227. Both span exactly
   168.0 hours. The `n_avail=650` recorded in the existing consoles is the
   cap passed to `count_available_batches`, not a measurement.
5. **The three report windows are one day, not one week.** Batches 0 to 599
   of W-2022-47 fall entirely inside 20221121, covering 1,228,800 of that
   day's 1,236,267 flows, which is 99.4 percent of the day and 18.6 percent
   of the week. Because the windows are equal in flow count and traffic
   volume varies over the day, they are unequal in time: window 1 spans
   11.19 hours, window 2 spans 4.46, window 3 spans 8.12, together covering
   Monday 21 November almost end to end in the capture site's clock. They
   are three different times of day, not three comparable samples. The
   W-2022-46 tuning window of 60 batches falls in the overnight hours of
   20221114.
6. **Stream ordering.** The test dataloader iterates a `SequentialSampler`
   over indices sorted by (table, row), that is by date and then position
   within the day's table. Measured against wall clock, the stream is
   approximately ordered by flow export time (`TIME_LAST`) with local
   jitter, not sorted exactly by either time:
     - Per-batch median time is non-decreasing across every consecutive
       batch pair in both periods. Zero inversions.
     - Displacement, the number of positions a flow would move under a
       stable global sort by `TIME_LAST`: median 38 and 45, p99.9 1,574 and
       1,721, maximum 2,428 and 2,414. A batch is 2,048 positions, so the
       99.9th percentile is under one batch and the maximum is 1.2 batches.
       Flows displaced beyond one batch are 5.2e-06 and 5.2e-05 of the
       stream, that is 26 and 341 flows.
     - The flow-level non-decreasing fraction is 0.5047 and 0.5036 on
       `TIME_FIRST`, 0.5075 and 0.5060 on `TIME_LAST`. A fraction near one
       half is the signature of small symmetric jitter, since jitter larger
       than the typical inter-arrival gap inverts about half of adjacent
       pairs. It does not distinguish light local jitter from a full
       shuffle; displacement does, and a full shuffle would give a
       displacement near n/3, five orders of magnitude larger than measured.
   The operative conclusion: **batch composition is what a globally
   export-sorted stream would give**, so consuming the loader in its natural
   order is consuming traffic in arrival order at the resolution the
   experiments operate at.
7. Timestamps carry sub-second precision (5,044,526 of 5,044,543 flows and
   6,607,221 of 6,607,244). Flow durations are median 0.21 s, p90 25 s,
   p99.9 305 s, maximum 363 s.
8. In the episodic protocol, 50 adaptation steps consume 50 distinct batches
   of the window's 200 (`order[s % len(order)]` with `s < 50`), that is
   102,400 of the window's 409,600 flows, while evaluation covers all 200.
   This is a property of the measured protocol and is reported as such.

## 2. Definitions fixed here

- **Flow (data point).** One QUIC flow, the unit the classifier scores: its
  per-packet information sequence, its flow statistics, and its packet
  histograms, after the published MM-CESNET-V2 input transforms.
- **Batch.** 2048 consecutive flows as delivered by the test dataloader.
- **Adaptation step.** One forward pass over one batch, plus, in gradient
  conditions, one backward pass and one Adam update of the batch-normalization
  affine parameters. In statistics-only conditions a step is the forward pass
  alone, which still updates the running statistics because the modules are in
  training mode.
- **Window.** 200 consecutive batches, 409,600 flows.
- **Episodic protocol.** The protocol of the current results: adapt on a
  window, then evaluate on that same window, then reset to the pretrained
  weights.
- **Streaming protocol.** Defined in Section 3.
- **Recovery.** Adapted accuracy minus frozen accuracy on the identical set of
  flows, in accuracy points. Primary quantity throughout, as before.

## 3. Experiment A: streaming evaluation

### A0. Ordering and length verification: COMPLETE

Run before this document was locked, with no model, no adaptation and no
label. Outcome recorded in Section 1 items 4 to 7 and in
`streaming_order_audit.json`.

**Criteria, original and final, both recorded.** The criterion first drafted
here was a flow-level `TIME_FIRST` non-decreasing fraction of at least 0.99.
The measurement showed that this statistic cannot distinguish light local
jitter from a full shuffle, because both sit near one half, and that it
therefore tests a property (flow-exact sorting) that the streaming
experiments neither have nor need. It was replaced, **after seeing the
ordering statistics and before any adaptation or accuracy number of any kind
existed for the streaming conditions**, by two criteria that test the
property the design depends on:

- **A0-1, batch order.** Per-batch median time non-decreasing across at
  least 0.999 of consecutive batch pairs. **Measured: 1.000000 in both
  periods, zero inversions. PASS.**
- **A0-2, displacement.** 99.9th percentile displacement under a stable
  global sort by `TIME_LAST` below one batch, 2,048 positions.
  **Measured: 1,574 (W-2022-46) and 1,721 (W-2022-47). PASS.**

The original 0.99 flow-level criterion **FAILED** (0.5047 and 0.5036) and is
reported in the manuscript as a measured property of the data alongside the
displacement result, not suppressed.

The substitution is recorded rather than quietly made because it is the one
discretionary decision in this document. Its justification does not depend on
any outcome of Experiments A to E: it rests on the mechanical fact that the
non-decreasing fraction is uninformative about local jitter, which is
demonstrable without reference to any model. No further criterion in this
document may be restated once the lock in Section 10 is recorded.

### A1. Conditions

All on the frozen adaptation parameter set (batch-normalization affine
parameters and running statistics, momentum 0.1). The model is never given a
label in any A condition.

- **A-frozen.** No adaptation. Reference for every streaming number.
- **A-causal-filtered.** For each batch b in chronological order: predict on b
  with the model in evaluation mode, so the prediction uses only running
  statistics accumulated from batches strictly before b; record the
  predictions; then take one filtered adaptation step on b. Accuracy is
  computed over the recorded predictions only. This is the strictly causal
  condition and it is the headline streaming number.
- **A-causal-stats.** As A-causal-filtered with the forward pass only, no
  optimizer.
- **A-batchtrans-filtered.** As A-causal-filtered except the prediction for
  batch b is taken with the batch-normalization modules in training mode, so
  batch b's own statistics normalize batch b. This requires buffering 2048
  flows before classifying any of them and is reported as such. The difference
  from A-causal-filtered is the measured value of that buffer.
- **A-reset-200.** A-causal-filtered with the model reset to the pretrained
  weights every 200 batches, bounding how far the adapted state can travel.

The episodic headline (+3.06 points) is carried into the same table
unchanged, labeled as requiring the full window in memory before any flow in
it is classified.

### A2. What is reported

Accuracy and recovery for each condition over: (i) each of the three windows
that define the existing Table I, batches 0 to 199, 200 to 399, 400 to 599,
so the streaming numbers sit directly beside the episodic ones on identical
flows, with each window's measured wall-clock span stated (11.19, 4.46 and
8.12 hours); and (ii) the whole of W-2022-47, all 3,227 batches, for the two
conditions that carry the claim, A-frozen and A-causal-filtered. The
remaining three conditions are reported on the three report windows only.
This split is a compute allocation fixed here in advance, not a choice made
after seeing which conditions looked better.

No condition is reported as a mean that hides a window below the frozen
baseline; per-window values are always shown.

Streaming has exactly one arrival order, so there are no seeded orderings and
no ordering error bars. The reported spread is across windows. Because the
three Table I windows are consecutive segments of a single day (Section 1
item 5), that spread characterizes variation across the diurnal cycle of one
Monday and nothing wider. Experiment E supplies the across-day replication.

### A3. Tuning, on W-2022-46 only

Grid: learning rate in {1e-4, 1e-3} crossed with entropy quantile in
{0.5, 1.0}, four configurations. Selection on the first 600 batches of
W-2022-46 by highest streaming accuracy in the A-causal-filtered condition.
Ties within 0.05 points break toward the smaller learning rate, the more
conservative choice. The selected configuration is written into `RESULTS.md`
before any W-2022-47 streaming run.

The reset period is **not** a tuned hyperparameter. A-reset-200 is a reported
condition evaluated at the selected learning rate and quantile, because its
purpose is to show what bounding the adapted state costs or buys, not to be
chosen for its score.

### A4. Kill rules

- **A-K1.** If the selected streaming configuration does not exceed the frozen
  model on W-2022-46, the streaming conditions are still run once on
  W-2022-47 and reported, because their purpose is to describe deployable
  behavior rather than to promote a method. What is killed in that case is the
  claim, not the measurement: the paper then states that under strict
  causality label-free adaptation does not deliver a usable recovery on this
  event, and the abstract says so.
- **A-K2.** If any of the three report windows shows A-causal-filtered below
  A-frozen, the method is described as capable of harming a window, and no
  aggregate is presented without that fact adjacent to it.
- **A-K3.** No streaming configuration is selected, tuned, or filtered using a
  W-2022-47 number. If a W-2022-47 streaming result prompts an idea for a
  better configuration, that idea is future work and is labeled as untested.

## 4. Experiment B: delayed-label retraining baselines

### B1. Structure

Labels are assumed to arrive with a fixed delay. When traffic on day d is
being classified, the most recent labeled traffic available is day d minus
delta. Delta in {1, 3, 7} days, all three reported, none selected post hoc.
For every evaluation day in W-2022-47, day d minus delta is a real date in the
dataset for all three delta values.

The retrained model for day d is fit on labeled flows from day d minus delta
only. It never sees any flow from day d. It is then evaluated on the same
W-2022-47 windows used everywhere else in the study, with the same evaluation
code path.

### B2. Capacities

- **B-matched.** Batch-normalization affine parameters and statistics only,
  labeled cross-entropy, same optimizer, same step count and the same number
  of flows as the label-free method consumes (50 steps, 102,400 flows). This
  is the capacity-matched and data-matched comparison.
- **B-full.** All parameters trainable, labeled cross-entropy. This is what an
  operator would in fact do, and the paper's comparison is not honest without
  it. Step count and learning rate tuned on W-2022-46 as in B4. Reported at
  K=1 if compute requires, and the K is stated.
- **B-head.** Final classification layer only, labeled cross-entropy. Cheap,
  and the common operational shortcut.
- **B-combined.** B-matched or B-full followed by label-free adaptation on the
  current unlabeled window. This is the only condition in the study that could
  become a deployment recommendation, and it is registered here so that it
  cannot be presented as a discovery made after the fact.

Two **label-free controls on the same source day**, added for the reason
recorded in Section 0.1. They receive the same pool, the same step budget and
the same seeded orderings as the supervised capacities, and no labels at all:

- **B-src-stats.** Batch-normalization statistics recalibrated on the source
  day by forward passes only. No gradients, no labels.
- **B-src-tent.** Filtered entropy adaptation on the source day, at the tuned
  learning rate and the frozen quantile. Label-free.

These are not delayed-label baselines and are excluded from kill rule B-K1,
which continues to compare only the supervised capacities against label-free
filtered adaptation. They are reported in the same table, in their own
family, and the quantity "best supervised minus best label-free control" is
reported explicitly as what supervision buys over recalibration on the same
delayed data. If that quantity is small, that is the finding, and it is
reported as prominently as the raw supervised number.

The existing transductive labeled oracle (+11.55 points, labels for the very
window being classified) is retained and relabeled explicitly as an
unattainable upper bound, not a baseline.

### B3. What is reported

A single table on W-2022-47, all conditions on identical flows: frozen;
label-free stats-only; label-free filtered; B-matched, B-full, B-head at each
delta; B-combined; transductive oracle. Accuracy points over frozen, per
window and pooled.

### B4. Tuning, on W-2022-46 only

The same delayed-label structure is available inside W-2022-46 and the days
preceding it. Learning rate in {1e-4, 1e-3}, steps in {50, 100} for B-matched
and B-head; learning rate in {1e-5, 1e-4} and steps in {50, 100} for B-full,
the lower range reflecting full-network fine-tuning. Selection by highest
W-2022-46 accuracy at delta = 1, then frozen for all delta and for
W-2022-47. Deliberately, the delayed-label baselines are tuned to their own
advantage, so that the comparison cannot be accused of crippling them.

### B5. Kill rules, fixed before any W-2022-47 delayed-label number exists

- **B-K1, the one that matters.** If any delayed-label baseline at any delta
  in {1, 3, 7} exceeds the label-free filtered recovery on W-2022-47 by more
  than 2.0 points, the paper does not present label-free adaptation as an
  operational recommendation. The framing changes, in the abstract and the
  conclusion, to the following, which is written here so it cannot be softened
  later: label-free test-time adaptation is a stopgap whose value is confined
  to the interval between the onset of drift and the arrival of labels, and
  this study measures how small that value is. The crossover delta is reported
  explicitly.
- **B-K2.** If delayed-label retraining fails to beat the frozen model at some
  delta, that is reported as a finding, not dropped. A baseline that does not
  work is evidence about the problem.
- **B-K3.** No delta is selected after the fact. All three are reported in the
  same table whatever they show.
- **B-K4.** If B-combined does not beat its own B component, no combined
  recommendation is made.

## 5. Experiment C: cost of adaptation on non-drifted traffic

### C1. Period-level control

Run the frozen label-free configuration (filtered, and stats-only) on three
windows of traffic that did not experience the shift: the earliest three
windows of W-2022-45, which are the study's pre-shift reference period, and
three windows of W-2022-44, the week the public weights were trained on.
K=3 seeded orderings, identical protocol to Table I.

Recovery here is expected to be near zero or negative. Whatever it is, it is
the cost an operator pays for running adaptation continuously when nothing has
drifted, and it is reported with the same prominence as the positive result.

Note on independence: the frozen accuracies of these windows are already
recorded and are unaffected by adaptation, so using the reference period as a
test period does not contaminate the gap denominator.

### C2. Class-partitioned control within W-2022-47

The primary partition is defined on the tuning week, with no reference to
W-2022-47 and no reliance on any external list. On W-2022-46, compute the
frozen model's per-class recall and compare it against the frozen model's
per-class recall on the pre-shift reference windows of W-2022-45. Classes
whose recall dropped by more than 10 points form the **affected** set; the
rest form the **unaffected** set. The two sets are written to
`class_partition.json` and their SHA-256 is recorded in `RESULTS.md`
**before** any W-2022-47 per-partition number is computed.

Then, on W-2022-47, report the change in accuracy produced by adaptation
separately on the affected and unaffected sets, for the filtered and
stats-only conditions.

A secondary, weaker check may be reported: the same split constructed from
service provider names in the dataset servicemap. It is secondary because it
depends on an external attribution of which services changed certificates,
which this study does not independently verify.

### C3. Kill rule

- **C-K1.** If adaptation reduces accuracy on the unaffected set by more than
  0.5 points, that cost is stated in the abstract, not only in the limitations
  section.

## 6. Experiment D: matched stability reference on the tuning week

The switch-point selection recorded in `PREREGISTRATION_switchpoint.md` used a
stability ceiling of 0.12 points, described as roughly twice the pure method's
0.06 points. That 0.06 was measured on W-2022-47 windows of 200 batches, while
the switch-point ordering standard deviations were measured on a W-2022-46
window of 60 batches. Those two window sizes do not produce comparable
ordering variance: with 50 steps on a 60-batch window an ordering consumes 50
of 60 batches, whereas on a 200-batch window it consumes 50 of 200, so the
subsets that different orderings select overlap far more in the first case.

Experiment D supplies the missing matched reference: pure filtered adaptation
at the frozen configuration, on the identical W-2022-46 60-batch window, with
the identical K=5 seeded orderings. It is inexpensive.

Pre-committed reporting, whichever way it falls:

- **D-R1.** If the pure method's W-2022-46 ordering standard deviation is at or
  below 0.06 points, the 0.12 ceiling was calibrated correctly and the
  switch-point rejection stands as recorded.
- **D-R2.** If it is above 0.06 points, the ceiling was stricter than intended.
  The rejection of the two-phase schedule is then re-justified in the
  manuscript on the evidence that does not depend on the ceiling, namely the
  single orderings at switch points 62 and 75 that drove accuracy below the
  frozen baseline, and the mis-calibration is stated in the text. The
  numerical outcome of the original selection is not revised, because it was
  executed as pre-registered; only its interpretation is corrected.

Experiment D touches no report-week data.

## 7. Experiment E: across-day replication of the existing results

A0 established that the recorded results rest on a single day of traffic
(Section 1 item 5). The manuscript currently describes the three windows as
slices of a single week demonstrating within-week stability. That is wrong,
and correcting it narrows the evidence rather than widening it, so the
correction is paired with the cheapest available strengthening.

**Design.** Repeat the frozen-configuration conditions (frozen baseline,
stats-only, filtered) on one 200-batch window drawn from each of the seven
days of W-2022-47, anchored at the day's first batch as given by the day map
in `streaming_order_audit.json`: batches 0, 603, 1233, 1856, 2418, 2823 and
3004. K=3 seeded orderings. No tuning and no configuration choice: the
configuration is the one already frozen on W-2022-46.

The window size is held at 200 batches so that every window is the 409,600
flows of Table I and the numbers are directly comparable. Six of the seven
days are long enough to contain such a window. **20221126 is not**: it holds
182 batches, so its window necessarily extends 18 batches, 9 percent of the
window, into 20221127. This is recorded here rather than discovered later,
and the affected window is labeled in the results table. The alternative,
shrinking every window to 182 batches, would break comparability with the
existing record and was rejected for that reason.

**What it buys.** Seven days rather than one, so the across-window standard
deviation becomes an estimate of variation across days rather than across
hours of one Monday, and the claim in the manuscript's uncertainty paragraph
becomes true instead of aspirational.

**Kill rules.**
- **E-K1.** All seven days are reported. No day is dropped for any reason,
  including a day that behaves unlike the others.
- **E-K2.** If the recovery on any day falls below zero, the method is
  described as capable of harming a day of traffic, in the abstract.
- **E-K3.** If the across-day standard deviation exceeds the across-window
  standard deviation already reported (0.26 points) by more than a factor of
  three, the headline is restated with the across-day error bar, and the
  single-day figure is retained only as a per-day detail.
- **E-K4.** The three original windows remain the primary Table I result, so
  that the existing record is not quietly replaced by a more favorable one.
  Experiment E is reported as an addition, whichever direction it moves the
  numbers.

## 8. Environment, provenance, and guards

- Same pinned environment as the recorded results (`requirements-lock.txt`).
  Any package change invalidates comparability and requires the whole
  affected family to be rerun.
- Every new script asserts, at runtime, the recorded bit-level anchors for
  frozen accuracy on the windows it touches: 0.72239013671875,
  0.72425537109375, 0.73946044921875 on the three W-2022-47 windows, and
  0.7534749348958333 on the W-2022-46 60-batch tuning window. A mismatch
  aborts the run.
- Every evaluation call in every new script is wrapped in `guarded_eval` from
  `scripts/tta_guards.py`, in strict mode. Before any experiment under this
  document is run, the existing scripts 02 to 12 are brought under the same
  wrapper, so that the released artifact matches what the manuscript claims
  about it.
- Raw per-unit outputs are written to JSON checkpoints, one file per
  experiment family, and are released.

## 9. Reporting commitments

- Recovery in accuracy points remains primary; percent-of-gap remains
  secondary.
- Every number that enters the manuscript is reproduced by a released script
  from a released raw artifact.
- Negative and null outcomes from any family above are reported at the same
  length and in the same place as positive ones.
- If the outcome of Experiment B triggers B-K1, the manuscript's contribution
  claim is rewritten before anything else in it is revised.

## 10. Lock

Convention, matching `PREREGISTRATION_switchpoint.md`: the hash below is the
SHA-256 of this file as committed in the content commit named below, that is
before these lines were filled in. A second commit records the hash. Both
commits are given so that the ordering is checkable by anyone.

SHA-256 of this file at lock time: [PASTE HASH FROM STEP 4]
Content commit (the locked file):  [PASTE COMMIT FROM STEP 4]
Recording commit (these lines):    [PASTE AFTER STEP 6]
Pushed to the public repository at: [PASTE DATE AND REMOTE]
