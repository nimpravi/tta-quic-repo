# Addendum: the pre-drift label condition (post-hoc, declared)

Status: **draft, to be hash-locked before the run.** Not part of
`PREREGISTRATION_streaming_delayed_label.md`. Prompted by the observation
that every label source in Experiment B post-dates the drift.Declared
post-hoc wherever it is reported.

---

## 1. Why this exists

Experiment B retrains on 20221120, 20221118 and 20221114. All three fall in
W-2022-46, after the certificate change during W-2022-45. Experiment B
therefore measures what label-free adaptation is worth **once post-drift
labels exist**, and the kill rule B-K1 fired on that comparison.

That is not the premise the method rests on. The argument for test-time
adaptation is that **no post-drift labels exist yet**. The interval between
drift onset and the first labeled post-drift traffic is the only regime in
which an operator has nothing else to use, and the study has never measured
it. As it stands a reader can conclude that label-free adaptation is
dominated in general, when what was shown is that it is dominated once the
drift has been labeled.

## 2. What to run

The same capacities, the same frozen configuration, the same three report
windows of 20221121, retrained on days that precede the drift.

| Source | Week | Why this day |
|---|---|---|
| **20221107** | W-2022-45 | Monday, before the onset the depth probe locates later in that week, and day-of-week matched to the Monday evaluation day. **This is the informative cell.** |
| 20221031 | W-2022-44 | Unambiguously pre-drift, but it is the week the public weights were trained on, so its result is inflated by memorization and is reported with that caveat rather than as a clean condition. |

Capacities: `src-stats`, `src-tent`, `head`, `matched`, `full`. K=3 except
`full` at K=1, matching Experiment B's convention. Roughly 26 units.

**Configuration is reused, not retuned.** The per-capacity learning rates and
step counts frozen for `--report` are applied unchanged. Retuning per source
day is a selection freedom the pre-registration does not grant, and the
conservative reading is that a configuration tuned to help a post-drift
source is not being handicapped here. A tuning sweep on the pre-drift source remains
available as a follow-up and would be declared separately.

**Isolation.** The run writes `predrift_label_progress.json`, a new artifact.
`delayed_label_progress.json` is not touched, `DELTAS` is not modified, and
the B-K1 verdict is not recomputed, so the 231 existing checks stay valid.

**Anchor.** The run must reproduce the frozen window accuracies
0.72239013671875 / 0.72425537109375 / 0.73946044921875 bit-for-bit. If it
does not, nothing from it is reported until the discrepancy is explained.

## 3. Decision rules, fixed before any number is seen

Evaluated on the 20221107 source. The W-2022-44 source informs nothing on its
own because of memorization.

**P1.** If the best pre-drift capacity exceeds the label-free filtered result
(+3.06) by more than 2.00 points, the same margin B-K1 used, then label-free
adaptation is dominated even before post-drift labels exist. The paper's
remaining operational claim does not survive and is withdrawn, not softened.

**P2.** If pre-drift retraining fails to beat the frozen model at every
capacity, then the window in which label-free adaptation is the best
available option is exactly the interval between drift onset and the first
post-drift labels. The paper states that as its operational scope, in the
abstract.

**P3.** If the result falls between zero and +3.06, the value is reported as
measured. Label-free adaptation retains an advantage in the pre-label
interval, smaller than the current framing implies, and the framing is
adjusted to the measured size.

**P4.** No pre-registered number is revised. Experiment B's table, its kill
rule, and every value verified by `21_verify_all.py` stand unchanged.

## 4. What is expected, recorded now so it cannot be claimed afterward

Near zero, possibly negative, at every capacity.

The reasoning: pre-drift labels teach the pre-drift feature distribution,
which is what the frozen model already encodes, so there is little for the
classifier to learn that it does not already know. For `matched` and `full`
the retrain additionally moves the BN running statistics toward pre-drift
traffic, which is the wrong direction for W-2022-47 and should cost rather
than help. `src-stats` on a pre-drift day is the cleanest case of this and
should land at or just below zero.

If instead the pre-drift condition recovers materially, the mechanism account
in the manuscript is wrong in a way that matters, because it would mean the
gain does not require post-drift information at all. **Writing the prediction
down before the run is what makes either outcome informative.**

## 5. How it is reported

One row group in Table II or a short paragraph in Section III-C, labeled
post-hoc, with the W-2022-44 caveat attached to its own line. If P2 holds,
the abstract's closing clause becomes a measured statement rather than an
inference: the operational case for label-free adaptation is confined to the
interval before the first post-drift labels, and that interval is the only
one in which it is not dominated.
