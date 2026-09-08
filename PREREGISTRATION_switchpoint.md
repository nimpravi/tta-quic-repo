# Pre-Registration: Switch-Point Tuning of the Two-Phase TTA Schedule

**Status:** FROZEN. Thresholds agreed by the author before any W-2022-47
evaluation of a tuned switch point exists: kill rule B magnitude floor =
+3.66 p; kill rules A/C ordering-stability ceiling = 0.12 p. This document
is committed to the repository (and its hash recorded) prior to running the
confirmatory step. Nothing below may change after the first W-47
confirmatory number is observed.

**Author:** Praveen Hegde. **Dataset/model:** CESNET-QUIC22 (size S,
ALL_KNOWN, 102 classes), frozen MM-CESNET-V2 (public W-2022-44 weights).
**Environment:** pinned `requirements-lock.txt` (Python 3.12, torch
2.12.1, numpy 2.5.0, sklearn 1.9.0, cesnet-datazoo 0.2.0, cesnet-models
0.4.1; CPU). Same windowing, leakage discipline, and state-guard
infrastructure as the existing pipeline.

---

## 1. Question and claim under test

**Claim (to be confirmed or killed):** Under leakage-clean, pre-registered
selection, a two-phase filtered-TENT schedule (early filtered adaptation
with BN statistics updating, then an explicit statistics freeze, then
filtered affine-only adaptation) recovers materially more accuracy on the
report week than the pure single-phase filtered-TENT headline
(+3.06 ± 0.27 p), **and** does so with ordering stability comparable to the
pure method.

If confirmed, the two-phase schedule is promoted from a post-hoc
observation to a pre-registered method and becomes the paper's positive
result. If not, it stays a post-hoc observation and the paper is submitted
as the decomposition (Option A), with this pre-registration recorded as
the reason.

## 2. Selection phase — TUNING WEEK W-2022-46 ONLY

W-2022-47 is **not touched** during selection. Selection uses a modified
script 11 (`11_switchpoint_select.py`) on W-46.

**Grid:** switch point s ∈ {25, 37, 50, 62, 75}, total steps = 100,
lr = 1e-3, q = 0.5, BN momentum = 0.1. (Finer than the current
{25,50,75} probe; required so the selected point is not forced onto the
accidental value of 50.)

**Orderings at selection:** K = 5 seeded orderings per switch point on the
W-46 window, `rng = numpy.random.default_rng(1000*0 + k)` for
k = 0..4 (window index 0, matching the existing seeding convention).
This is a change from the current natural-order-only probe and exists so
that the ordering-stability kill rule (§4) can be evaluated on W-46
without touching W-47.

**Selection rule (deterministic, no discretion):** among the five switch
points, select the one with the highest **pooled mean recovery on W-46**.
Ties (within 0.05 p) broken toward the **earlier** switch point (cheaper,
and consistent with the "freeze early" reading). The selected switch point
s\* and its W-46 pooled mean ± std and within-window order-std are recorded
in this file before any W-47 run.

## 3. Confirmatory phase — REPORT WEEK W-2022-47, ONCE

Using **only** s\* selected above, run the two-phase schedule on W-47 via
script 09 (`--switch s*`), K = 5 orderings × 3 windows, final-step
evaluation only, state guards active. This is the single confirmatory
measurement. It is run **once**. No re-selection, no second look, no
switch-point adjustment after seeing W-47.

The k=0 provenance anchors in script 09 (per-window +4.73 / +4.01 / +3.71
at switch=50) remain assertions **only if s\* = 50**; for any other s\*
the anchor check is disabled (there is no archived trajectory to match),
and provenance rests on within-run bit-determinism and the state-guard
checksums instead. This is noted so a disabled anchor is not later
mistaken for a broken one.

## 4. Kill rules — COMMITTED BEFORE RUNNING W-47

Both must pass for promotion. Evaluated in this order.

**Kill rule A (stability, evaluated on W-46 at SELECTION time — before W-47
is touched at all):** if the selected s\* has within-window order-std on
W-46 greater than **0.12 p** (i.e., more than ~2× the pure method's 0.06 p
and near the 0.19 p instability already seen for switch=50), the schedule
is declared too ordering-sensitive to headline. STOP. Do not run the W-47
confirmatory. Submit Option A. Record the failure here.

**Kill rule B (magnitude, evaluated on W-47 confirmatory):** if the
confirmatory pooled mean on W-47 is below **+3.66 p** (i.e., the lift over
the +3.06 pure headline is under +0.60 p), the schedule does not earn a
headline. It stays a post-hoc observation. Submit Option A. Record the
number here.

**Kill rule C (stability holds on W-47):** if the confirmatory within-window
order-std on W-47 exceeds **0.12 p**, or the previously-seen unexplained
window-1 natural-order interaction (+5.62 p class) reappears in any
seeded ordering, the schedule is not stable enough to headline regardless
of magnitude. Stays a post-hoc observation. Submit Option A. Record here.

**Promotion requires A and B and C to all pass.** Any single failure →
Option A. There is no partial-credit path and no "close enough" override.

## 5. What gets written either way

- **Promotion (A∧B∧C pass):** two-phase becomes a pre-registered row in
  Table I; §II gains a short switch-point-selection paragraph; §III-E is
  reworked from "accident found by a probe" to "method selected under the
  same leakage-clean discipline, whose switch point the audit happened to
  surface." The audit narrative is COMPRESSED, not deleted — it remains
  the provenance story. Page budget: promotion costs ~0.4–0.5 p; bought
  back by compressing §III-E and, if needed, trimming the W-47 control
  trace detail in §III-A. Target stays 5 pages.
- **No promotion (any kill fires):** paper is the Option-A decomposition,
  unchanged in substance. This pre-registration is cited in the repo as
  the record of a fairly-run, honestly-reported negative selection
  outcome, which is itself evidence of method integrity.

## 6. Provenance of this document

Committed to the repo before the W-47 confirmatory run; its SHA-256 is
recorded in the run log so the freeze is verifiable. The confirmatory
script's output (script 09 with `--switch s*`) is the only new W-47 number
generated under this protocol.
