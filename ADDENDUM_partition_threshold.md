# Addendum: partition threshold robustness (post-hoc, declared)

Status: **draft, to be hash-locked before the run.** This is not part of
`PREREGISTRATION_streaming_delayed_label.md`; it is a post-hoc robustness
analysis, prompted by a sensitivity check on `class_partition.json` performed
after v4 was complete. It is declared as post-hoc wherever it is reported.

SHA-256 of this file at lock time: 97494a2e4216eee60aeb7700dd2ae5c4818524550fee8f289562ab2a25c6142c
---

## 1. Why this exists

The class partition uses a pre-registered rule: affected if per-class recall
on W-2022-46 falls more than 10 points below W-2022-45. Every claim in
Section 14 of RESULTS.md rests on it. A sensitivity check on the released
artifact shows the cut does not fall in an empty region of the drop
distribution:

| Threshold | Affected classes | Affected share of W-46 flows |
|---|---|---|
| 0.05 | 34 | 60.9% |
| **0.10 (pre-registered)** | **29** | **60.7%** |
| 0.15 | 21 | 47.9% |
| 0.20 | 18 | 41.5% |

Loosening the cut is nearly free. Tightening it is not, and the swing is
dominated by one class: `google-ads` drops 10.6 points, half a point above
the cut, and carries 94,577 flows, 7.7% of the week.

The decomposition has never been computed at any threshold but 10 points,
so the sensitivity of the *conclusion*, as opposed to the sensitivity of the
partition, is currently unknown.

## 2. The better experiment

The obvious run is "repeat C2 at a 15-point threshold". That answers one
question and leaves the same objection available at every other cut.

Instead, record **per-class correct and total counts** on the report windows,
for the frozen model and for each adapted seed. Every threshold then becomes
an offline arithmetic exercise, permanently, with no further model runs. The
artifact stops being tied to one partition rule.

It also makes a stronger analysis possible. With per-class outcomes, the
binary partition can be replaced by a continuous one: plot each class's
accuracy change under adaptation against its measured drift magnitude
(`recall_W45 - recall_W46`). If the mechanism in the manuscript is right,
that relationship is increasing and crosses zero somewhere, and the
threshold question dissolves into a measured dose-response curve. A reviewer
cannot object to a threshold that is no longer load-bearing.

## 3. What to run

One re-run of Experiment C's C2 stage only. Nothing else is touched.

- 3 report windows of 20221121, the same batch offsets as the record.
- Frozen evaluation, then `stats` and `filtered` at the frozen episodic
  configuration, K=3, the same seeds as the record.
- 18 adaptation units plus 3 frozen evaluations.

Order of hours on CPU, extrapolating from the recorded claim that one
50-step window adaptation "completes in minutes". **That is an estimate from
the record, not a measurement.**

**Required change to `18_nondrifted_control.py`:** in the `--c2` path, in
addition to the existing `affected` and `unaffected` aggregates, write
`per_class` for every unit: for each class id, the number of flows of that
class in the window and the number classified correctly. The existing fields
must be kept unchanged so that `21_verify_all.py` continues to pass and the
new artifact remains a strict superset of the old one.

**Anchor:** the run must reproduce the recorded frozen window accuracies
0.72239013671875 / 0.72425537109375 / 0.73946044921875 bit-for-bit, and
recomputing the 10-point partition from `per_class` must return
+6.87 / -2.79 and +7.18 / -4.89. If either fails, the run is not comparable
with the record and nothing from it is reported until the discrepancy is
explained.

## 4. Decision rules, fixed before any new number is seen

These are written now, in advance, for the same reason the earlier kill
rules were.

**R1, sign.** If at any swept threshold in {0.05, 0.10, 0.15, 0.20} the
filtered change on unaffected traffic is greater than or equal to zero, the
sign of the transfer is threshold-dependent, and the abstract must say so
rather than stating the loss unqualified.

**R2, magnitude.** If the filtered loss on unaffected traffic at the
15-point cut is smaller in magnitude than 1.00 point, the paper reports the
loss as a range across thresholds, keeping the pre-registered 10-point value
as the headline and giving the range in the results.

**R3, operating point.** If at any swept threshold the windows' affected
fraction falls below the break-even computed at that same threshold, the
claim that these windows sit above break-even is threshold-dependent and is
stated as such.

**R4, mechanism.** If the rank correlation between per-class drift magnitude
and per-class accuracy change under adaptation is not positive, the
mechanism account is not supported by class-level evidence and is weakened
accordingly, whatever the aggregate tables show.

**R5, no substitution.** Whatever the sweep shows, the pre-registered
10-point results stand as the primary analysis. The sweep is reported as
post-hoc robustness. It never replaces Table II, and no recorded number is
revised on its basis.

## 5. What is already expected, recorded now so it cannot be claimed later

Tightening the cut moves eight classes with 10 to 15 point drops out of the
affected group and into the unaffected group. Those classes did drift, and
adaptation probably helps them, so the measured loss on the unaffected group
should shrink as the threshold tightens. If that happens it is not evidence
that the harm was overstated; it is evidence that a stricter cut makes the
unaffected group a less pure control. **Writing this down in advance is what
distinguishes it from a rationalization written afterward.** If the loss
instead grows, or does not move, the prediction was wrong and the record
says so.
