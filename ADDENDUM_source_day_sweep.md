# Addendum: the source-day sweep (post-hoc, declared)

Status: **draft, to be hash-locked before the run.** A follow-up to
`ADDENDUM_predrift_labels.md`, prompted by an unplanned observation in that
run. It revises nothing in it: P1 to P4 were evaluated on 20221107 over the
supervised capacities and their verdicts stand.

---

## 1. What prompted it

Adapting label-free on 20221107 recovered $+6.01$ points on the report
windows, against $+3.06$ for the same method applied to the report windows
themselves and $+2.66$ to $+2.92$ for post-drift source days. Three seeds,
range $[5.80, 6.19]$.

Two facts make this worth chasing rather than reporting:

- On the same day, supervised retraining with true labels recovers $+0.96$.
  Beating ground truth with self-generated targets on identical data is an
  anomaly.
- It is not monotone in anything obvious. W-2022-44 gives $+0.86$,
  W-2022-45 gives $+6.01$, W-2022-46 gives $+2.80$.

A harness fault is ruled out: the same code path reproduced the post-drift
values that match Experiment B, and the frozen anchor passed bit-for-bit.

## 2. What to run

`src-tent` and `src-stats` only, K=1, on the remaining days of W-2022-45:
20221108 through 20221113. Six days, twelve units. From the recorded
per-unit times, roughly 26 minutes per day and under three hours in total.

K=1 is deliberate. The seed range at 20221107 was 0.39 points wide against an
effect of 3 points, so a single ordering separates the hypotheses below. Days
that matter can be re-run at K=3 afterwards.

This is a screening sweep. It adds no supervised condition and therefore
cannot bear on P1 to P3.

## 3. Why this is a natural experiment, not just a replication

The drift onset falls **inside** W-2022-45. The depth probe shows the frozen
model at roughly $0.95$ early in that week decaying to $0.89$ by its end, so
source days later in the week are progressively more drifted. The sweep
therefore varies source cleanliness while holding everything else fixed.

## 4. Pre-committed readings, fixed before any number is seen

**S1.** If recovery is near $+6$ on the early days and declines across the
week as the source becomes more drifted, that is a dose-response in source
cleanliness. The manuscript's claim that more unlabeled traffic does not
raise the ceiling is then **wrong, not merely narrow**, and the paper reports
the sweep as a finding: label-free adaptation is limited by the cleanliness
of the traffic it adapts on, not by the amount.

**S2.** If 20221107 is isolated and the other W-2022-45 days sit near the
post-drift $+2.8$, then 20221107 is an anomaly. The ceiling claim survives,
narrowed to post-drift sources, and the anomaly is reported as unexplained
rather than built on.

**S3.** If every W-2022-45 day sits near $+6$ with no gradient across the
week, cleanliness is not the explanation. Report as unexplained, state that
no mechanism is offered, and do not build on it.

**S4.** Nothing here revises a pre-registered number, the B-K1 verdict, or
P1 to P4.

## 5. What is expected, recorded now

S1, with recovery declining across W-2022-45 as the source day approaches the
onset.

The reasoning: entropy minimization pushes the model toward its own
confident predictions. On clean traffic the confident predictions are mostly
correct, so the gradient behaves like self-training on correct targets. On
drifted traffic the confident predictions include confidently wrong ones on
the affected classes, so the same gradient reinforces error. This is the same
account already in the manuscript for why filtering helps, carried one step
further to the source data rather than the sample selection.

**If S2 holds instead, the prediction is wrong and the record says so.** The
prediction is written here so that a decline cannot be claimed afterwards as
something anticipated, and so that an isolated outlier cannot be quietly
dropped.

## 6. What is not being run, and why

No supervised condition. The question is about the label-free control, and
adding supervised cells would invite reading them against P1 to P3, which
were settled on 20221107 and are not reopened.
