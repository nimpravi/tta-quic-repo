# RESULTS: Test-Time Adaptation Under Abrupt Temporal Drift in Encrypted QUIC Traffic Classification

This file supersedes all previous
versions. Every number in Sections 1–6 was produced by the final-evaluation
pipeline in one pinned environment (`requirements-lock.txt`: Python 3.12,
torch 2.12.1, numpy 2.5.0, scikit-learn 1.9.0, cesnet-datazoo 0.2.0,
cesnet-models 0.4.1; CPU). Section 7 records superseded results and the
mechanisms of their corruption. Section 8 records the provenance chain.

The two-phase schedule of §6 was submitted to a
pre-registered switch-point selection (§10) and **REJECTED** on the
pre-committed ordering-stability rule. §6 is retained but reclassified from
"post-hoc observation / future work" to **rejected candidate**. The coarse
W-46 switch-point probe in §9 is superseded by the pre-registered selection
in §10.

---

## 0. Protocol summary

- **Dataset / model:** CESNET-QUIC22 (size S, `ALL_KNOWN`, 102 classes);
  MM-CESNET-V2 pretrained on W-2022-44 (published weights + transforms).
- **Windows:** test loader yields 2048-sample batches (the `batch_size=256`
  in the scripts is overridden by datazoo for test loaders). Each evaluation
  window = 200 consecutive batches = **409,600 flows**; three disjoint
  windows per week (skip 0 / 200 / 400). The loader is time-ordered
  (established by the W-45 depth probe, §5).
- **Leakage-clean tuning:** all hyperparameters selected on W-2022-46
  (60-batch window, deterministic natural-order adaptation, final-step
  evaluation), then frozen. W-2022-47 (report week) played no part.
  Frozen config: **lr = 1e-3, steps = 50, entropy quantile = 0.5**
  (W-46 recovery +2.68p).
- **Adaptation (TENT-style):** BN affine parameters only; BN modules in
  train mode (momentum 0.1) so running statistics also update; Adam;
  per-batch entropy-quantile filtering; transductive/episodic (adapt
  label-free on the same window that is then evaluated). **Evaluation
  exactly once, after the final step** (no mid-trajectory probes; see §7.1
  for why this matters).
- **Error bars:** the pipeline is bit-deterministic within a process, so
  "seeds" vary the one genuine stochastic factor: the adaptation batch
  ordering, `numpy.random.default_rng(1000*window + k)`. K orderings ×
  3 windows; pooled mean ± std reported, per-window means as secondary.

## 1. Drift magnitude and the gap denominator

Frozen W-44 model, identical windowing protocol throughout:

| Week | Per-window accuracy | Mean |
|---|---|---|
| W-45 (early week, pre-onset) | 0.9559 / 0.9507 / 0.9590 | **0.9552 ± 0.0034** |
| W-46 | 0.7233 / 0.7253 / 0.7451 | 0.7312 ± 0.0099 |
| W-47 (report week) | 0.7224 / 0.7243 / 0.7395 | **0.7287** |

**Gap = 0.9552 − 0.7287 = 22.65 points**, both terms self-measured under
the identical protocol. The previously used denominator (published
in-period figure ~0.867) is a whole-week W-45 mixture of pre- and
post-onset traffic (§5) and understates the true pre-shift gap by ~9
points. Percent-of-gap figures below use 22.65p; recovery in points is
primary throughout.

## 2. Main decomposition (all at the frozen config, steps = 50)

| Condition | Recovery (points) | % of gap | Per-window | Order-std |
|---|---|---|---|---|
| BN-statistics recalibration only (no gradients) | **+2.43 ± 0.15** | 10.7% | +2.64 / +2.35 / +2.31 | 0.03p |
| + filtered entropy gradients (q = 0.5), **headline** | **+3.06 ± 0.27** | 13.5% | +3.42 / +2.95 / +2.81 | 0.06p |
| + unfiltered entropy gradients (q = 1.0) | **+1.32 ± 0.53** | 5.8% | +1.99 / +1.18 / +0.78 | 0.14p |

(Headline: K=5×3=15 orderings; controls: K=3×3=9. Std is over pooled
orderings; conservative mean−1sd headline = +2.79p.)

**Findings:**

1. **Recalibration does most of the work.** BN-statistic recalibration
   alone recovers +2.43p, i.e. ~79% of the headline recovery, with the
   lowest variance of any condition and no gradient computation.
2. **Entropy filtering gates the SIGN of the gradient contribution.**
   Relative to the stats-only control, filtered gradients add **+0.62p**;
   unfiltered gradients subtract **−1.12p** (destroying ~46% of the
   recalibration benefit). The correct claim is not "filtering is
   load-bearing for recovery" (recovery exists without any gradients) but
   "filtering determines whether gradient adaptation helps or harms."
3. **Ordering of conditions is per-window consistent** (filtered >
   stats-only > unfiltered in all three windows individually). Welch
   tests on pooled orderings support all three pairwise contrasts
   (filtered vs stats t=6.99, p≈5e-7; stats vs unfiltered t=5.78,
   p≈2e-4; filtered vs unfiltered t=8.76, p≈4e-6), with the caveat that
   orderings within a window are not independent samples; the per-window
   consistency is the primary evidence.
4. **Unfiltered damage grows with steps** (secondary, pure-mode evidence
   across configs): vs matched stats-only controls, unfiltered gradients
   cost −1.12p at 50 steps and −1.54p at 100 steps (+0.69 ± 0.37 vs
   +2.23 ± 0.15). Pure filtered TENT is also slightly worse at 100 steps
   (~+2.98 at k=0) than at 50 (+3.06), consistent with the honest tuning
   choice of 50.

## 3. No class-collapse (headline condition, k=0 orderings)

| Window | Δaccuracy | ΔmacroF1 | Distinct preds | Pred-entropy Δ |
|---|---|---|---|---|
| 1 | +3.41p | −0.41p | 102 → 102 | +0.010 |
| 2 | +2.94p | +0.08p | 102 → 102 | +0.003 |
| 3 | +2.97p | +0.07p | 102 → 102 | +0.014 |

Macro-F1 is flat (mean −0.09p on a ~0.80 base); all 102 classes remain
predicted; prediction-distribution entropy rises slightly (concentration
would lower it). Recovery is a coherent redistribution concentrated in a
small set of head classes (largest predicted-mass gains: classes 19, 59,
57, 55; largest losses: 101, 61, 62, 66), consistent with re-sorting of
the service family affected by the W-45 event rather than tail-class
abandonment. The manuscript reports macro-F1 alongside accuracy.

## 4. Ordering (retraction and replacement)

**Retracted:** the previous claim that randomized adaptation order
stabilizes recovery (shuffled range 0.62p vs natural range 2.76p). Both
numbers are cross-vintage artifacts (§7.3): they compared runs of two
different algorithms.

**Replacement (clean pipeline):** recovery is essentially insensitive to
adaptation order. Within-window order-std is 0.06p (headline), 0.03p
(stats-only), 0.14p (unfiltered). Pure-mode natural-order runs at
steps=100 (+2.86 / +2.86, windows 2–3) match shuffled k=0 values
(+2.83 / +2.76) within noise. Across-window heterogeneity (std 0.26p)
dominates ordering variance and is the honest uncertainty.

## 5. Intra-week drift onset (W-45 depth probe; time-ordered loader)

Frozen-model accuracy vs position in the W-45 test stream (60-batch
windows; offsets in batches; flows = offset × 2048):

| Offset | 0 | 200 | 400 | 700 | 1000 | 1400 | 1800 |
|---|---|---|---|---|---|---|---|
| Acc | 0.966 | 0.949 | 0.955 | 0.948 | 0.914 | 0.892 | 0.908 |

Stable ~0.95–0.97 through offset ~700, then declining, consistent with a
mid-week onset of the documented Google TLS-certificate change (Luxemburk
et al., TMA 2023; CESNET Scientific Data 2024) and with time-ordered
test loading. This (a) validates early-W-45 as the pre-shift reference,
(b) explains the published ~0.867 as a whole-week pre/post mixture, and
(c) provides the drift figure. The full cliff to ~0.73 lies beyond the
walked depth; not required for the reference definition.

## 6. REJECTED CANDIDATE: two-phase "hybrid" schedule

Deliberate reimplementation (script 09) of the schedule discovered during
the forensic audit (§7.1): 50 steps standard filtered TENT (stats
updating), explicit stats-freeze (`m.eval()`), then 50 further steps of
filtered affine-only adaptation. Final-step evaluation; k=0 units
bit-reproduce the reconstructed trajectories (anchors matched).

| Condition | Recovery | % of gap | Per-window | Order-std |
|---|---|---|---|---|
| Hybrid (50+50, switch untuned) | **+4.31 ± 0.49** | 19.0% | +4.90 / +4.00 / +4.02 | 0.19p |

Lift over the pure-50 headline: **+1.25p** (Welch t=7.00, p≈2e-5;
per-window consistent). Interpretation consistent with §2: recalibrate
statistics early, then freeze them; filtered affine refinement continues
to help on frozen statistics, while continued statistic updates past
convergence slightly hurt.

**STATUS (v3): REJECTED.** The switch point (50) was an artifact of the
discovery and was never tuned, and the schedule showed elevated ordering
sensitivity (0.19p, ~3x the pure method's 0.06p) plus an unexplained
natural-order interaction (+5.62p, window 1, single run). Rather than
adopt it on the strength of an appealing number, the schedule was put
through a **pre-registered switch-point selection (§10)**. It **failed
the pre-committed ordering-stability kill rule at selection time on the
tuning week**, and per the pre-registration **no W-47 confirmatory run
was performed**. The +4.31p figure in the table above is therefore a
favorable single-configuration draw from a high-variance procedure, not
a stable capability. This is NOT the paper's method, and it is not
offered as a method at all. See §10.

## 7. Correction record (superseded results and mechanisms)

### 7.1 Superseded headline (+4.66; +4.34 ± 0.36 "31.4% of gap")

The pipeline's mid-trajectory accuracy probe (every 50 steps)
called `model.eval()` and never restored train mode. For steps=100
configs, steps 51–100 therefore ran a *different algorithm* (affine-only
adaptation on frozen BN statistics), an undocumented two-phase hybrid
created by the measurement itself. The archived headline numbers are
real measurements of that hybrid (verified: reconstructed k=0
trajectories reproduce archived per-window envelopes, and window 3's
archived minimum +3.71 exactly; see §8). They are superseded because the
algorithm was undocumented, its schedule accidental, and its config
selected by a corrupted tuning comparison (§7.2). **No test-label
checkpoint leakage occurred at the final config** (best-over-checkpoints
equals final-step there; leakage-demo inflation = +0.00p in all windows).
The deliberate reimplementation is reported honestly as §6.

### 7.2 Corrupted hyperparameter selection (steps=100 → 50)

In the old tuning grid, 50-step rows were pure-mode (probe at step 50 =
final evaluation) while 100-step rows were hybrid-mode (probe fired
mid-trajectory). The grid therefore compared different algorithms:
pure-50 (+2.68p) vs hybrid-100 (+3.88p), selecting steps=100. Honest
final-eval tuning selects steps=50 (+2.68p, bit-identical to the old
50-step row, confirming the shared pure-mode code path). Additionally,
the old `best = max(frozen, …)` clamp reported degrading configs
(lr=5e-3 rows) as exactly +0.00p, masking damage.

### 7.3 Retracted ordering claim (§4 of v1)

"Natural range 2.76p" mixed one hybrid-mode window (+5.62) with two
pure-mode windows (+2.86/+2.86): 5.62 − 2.86 = 2.76. "Shuffled range
0.62p" was all-hybrid (4.74 − 4.12). Both sides of the contrast are
cross-vintage arithmetic; no ordering effect exists in the clean
pipeline (§4).

### 7.4 Superseded denominator (0.867 → 0.9552)

See §1 and §5. Points-primary reporting is unaffected; %-of-gap figures
changed from 31.4% to 13.5% (headline).

### 7.5 Clean-but-stale results

The archived quant=1.0 mechanism run (+0.69 ± 0.37, steps=100) came from
a probe-free script and is uncontaminated; superseded only because the
frozen config changed to steps=50. It remains cited in §2 finding 4.

## 8. Provenance chain (why the final numbers are trusted)

1. Frozen accuracies are bit-identical (dyadic rationals over 409,600
   samples) across scripts 02–09, both step configs, and the 
   archive: data, windows, weights, and eval path never changed.
2. The clean steps=50 adapted accuracies are bit-identical between the
   headline run (02) and the collapse check (05), which recomputes
   accuracy from raw predictions through an independent code path.
3. The leakage demo (08) reproduced, from single reconstructed
   trajectories, the archived per-window K=8 envelopes, including
   window 3's archived minimum to printed precision, while its acc@50
   values are bit-identical to the clean steps=50 runs, tying old and
   new vintages to one verified mechanism.
4. The hybrid run's (09) k=0 anchors reproduce the demo trajectories to
   ≤0.01p, giving §6 the same provenance standard.
5. The old tuning grid's 50-step rows are bit-identical to the clean
   grid's, confirming the corruption analysis of §7.2.
6. Environment pinned in `requirements-lock.txt`; all numbers from one
   environment; within-process bit-determinism audited (see
   diagnostics/).
7. **(v3)** The pre-registered selection (§10) ran in the same pinned
   environment: its W-46 frozen accuracy is bit-identical
   (`0.7534749349`) across all 25 units, confirming the frozen model was
   never perturbed across the switch-point grid and that its recoveries
   are comparable to §2. The pre-registration was hash-locked
   (SHA-256 `4ebd14fb…712e45`) and committed **before** the run, so the
   "selected before the report week was touched" claim is verifiable
   rather than asserted. The kill rule fired at selection time; **no
   W-47 number was generated**, which is itself the strongest available
   guarantee that the report week did not inform the rejection.

## 9. Calibration and robustness experiments (v2.1, 2026-07-05)

All under the pinned environment, final-eval, same protocol.

- **Matched-capacity labeled oracle** (script 10, K=3): BN-affine
  fine-tune with ground-truth CE, 50 steps: **+11.55 ± 0.38 p**
  (per-window +12.02/+11.52/+11.10) = 51.0% of the 22.65p gap. The
  label-free headline is therefore **26.5% of the matched ceiling**;
  even supervision recovers only ~half the gap at this capacity.
- **Filtered TENT, steps=100, K=3** (script 12): **+2.96 ± 0.27 p**
  (per-window +3.34/+2.82/+2.73; k=0 units reproduce the earlier K=1
  values +3.35/+2.83/+2.76 exactly). Replaces the K=1 Table II cell;
  filtered gradient effect at 100 steps becomes +0.73 p.
- **Switch-point probe, W-46 only** (script 11, natural order):
  freeze@25 +3.69 p, freeze@50 +3.88 p, freeze@75 +2.76 p, pure-100
  +0.46 p (pure-50 reference +2.68 p). W-47 untouched.
  **SUPERSEDED by §10.** This probe used a single natural ordering and no
  error bars, so it could not see the ordering instability that the
  pre-registered selection (§10, K=5 orderings) later exposed. Its
  reading ("early-to-mid freezing is what matters") is consistent with
  the §10 means but misleading in isolation: the early switch points that
  look best by mean are precisely the least stable. Retained as the
  motivating observation, not as evidence.
- **W-47 depth trace** (script 07 --week W-2022-47): trendless scatter
  0.709--0.752 (range 0.042, no monotone component) at the same
  offsets where W-45 declines monotonically, ruling out within-week
  composition as the driver of the W-45 trend.

## 10. Pre-registered switch-point selection (v3, 2026-07-11): NEGATIVE

**Pre-registration:** `PREREGISTRATION_switchpoint.md`, committed to the
repository and hash-locked **before any W-47 evaluation of a tuned switch
point existed**. SHA-256:
`4ebd14fbe8b721e9bb86683febd724b4ba7b08fb8a2d7c83a7444c37f9712e45`.
Script: `11_switchpoint_select.py`. Raw: `switchpoint_select.json`.

**Design (all fixed in advance, no discretion):**
- Selection on **W-2022-46 only**. W-2022-47 not touched.
- Grid: switch ∈ {25, 37, 50, 62, 75} of 100 total steps (finer than the
  §9 probe, so the selected point is not forced onto the accidental 50).
- **K = 5 seeded orderings** per switch point, `default_rng(1000*0 + k)`,
  so ordering stability is measurable **at selection time**, the only way
  the stability rule can fire without consulting the report week.
- lr = 1e-3, q = 0.5, BN momentum = 0.1 (frozen config, unchanged).
- **Selection rule:** highest pooled mean recovery on W-46; ties within
  0.05p break toward the **earlier** (cheaper) switch point.

**Kill rules, committed before the run:**
- **A (stability, on W-46 at selection):** if the selected switch point's
  within-window order-std > **0.12p** (~2x the pure method's 0.06p),
  reject outright and **do not run the W-47 confirmatory**.
- **B (magnitude, on W-47, only if A passes):** confirmatory recovery must
  exceed **+3.66p** (> +0.60p over the pure +3.06p headline) to promote.
- **C (stability on W-47):** order-std > 0.12p, or reappearance of the
  unexplained +5.62p window-1 interaction, disqualifies regardless of mean.
- Promotion required **A ∧ B ∧ C**. No partial credit, no override.

**Environment check:** W-46 frozen accuracy = **0.7535** across all 25
units, bit-identical (`0.7534749349`), matching the recorded baseline.
Environment consistent with the rest of the pipeline; recoveries are
comparable to §2 numbers.

**Selection outcome (W-2022-46, K=5, recovery in points):**

| Switch | Mean | Order-std | Per-ordering |
|---|---|---|---|
| **25** | **+5.11** | **0.27** | +4.71 / +4.86 / +5.38 / +5.24 / +5.37 |
| 37 | +3.98 | 0.30 | +4.40 / +3.95 / +3.47 / +3.95 / +4.11 |
| 50 | +3.46 | 0.18 | +3.69 / +3.62 / +3.20 / +3.34 / +3.43 |
| 62 | +2.45 | 1.88 | **−1.28** / +3.37 / +3.10 / +3.31 / +3.75 |
| 75 | +2.23 | 1.72 | +2.39 / **−1.13** / +3.21 / +3.27 / +3.40 |

(pure-50 reference on the same clean W-46 grid: +2.68p)

**Deterministic selection → s\* = 25** (highest pooled mean, +5.11p).

**KILL RULE A: FAIL.** s\* = 25 has order-std **0.27p > 0.12p ceiling**.
Per the pre-registration: **STOP. The W-47 confirmatory was NOT run.** The
two-phase schedule is a **rejected candidate**, not a method. No
report-week number exists for any tuned switch point.

**Findings:**

1. **Lift and instability are the same phenomenon.** Mean recovery rises as
   the freeze moves earlier, and ordering instability rises with it and
   faster. No switch point on the grid delivers both a lift over the pure
   method (+3.06p) and pure-method-class stability (0.06p).
2. **Late freezes can be actively harmful.** Switch 62 and 75 each contain
   a single ordering that drives accuracy **below the frozen baseline**
   (−1.28p, −1.13p; verified in the raw JSON, not a display artifact). A
   method that can underperform doing nothing is not deployable.
3. **The only stable point is marginal.** Switch 50 is the sole
   configuration with near-pure-method stability (0.18p) and it yields
   +3.46p on W-46, barely above the pure method. Even it exceeds the
   0.12p ceiling.
4. **The archived +4.31p (§6) is explained.** It was a favorable draw from
   a high-variance procedure under a single ordering, not a stable
   capability. Multi-ordering evaluation dissolves it.
5. **The protocol earned its keep.** A single-ordering evaluation would
   have reported the schedule as a method beating the tuned baseline by
   >1 point. The pre-registered stability criterion instead rejected it,
   on the tuning week, before the report week was ever consulted.

**Interpretation for the manuscript:** this is reported as a negative
result. The value is that the evaluation discipline converted an appealing
number into a correct rejection. Whether some stabilized variant of a
two-phase schedule could pass the same criterion is left open; the
schedule as tested does not.

## 11. Raw artifacts

- Clean (v3): `PREREGISTRATION_switchpoint.md` (hash-locked, SHA-256
  `4ebd14fb…712e45`), `11_switchpoint_select.py`,
  `switchpoint_select.json` (25 units, K=5 x 5 switch points, W-46 only).
- Clean (v2.1): `oracle_ceiling.txt` + `oracle_matched_progress.json`,
  `filtered100_errorbars.txt` + `filtered100_progress.json`,
  `switchpoint_probe.txt` + `.json` (coarse probe; superseded by §10 but
  retained as the motivating observation), `w45_depth_probe.txt` + `.json`
  (now with the W-2022-47 key).
- Clean: `headline_K8_final_eval.txt` (K=5), `errorbars_progress.json`,
  `mechanism_steps50_run.txt` + JSON, `bnstats_steps50_run.txt` + JSON,
  `collapse_steps50_run.txt` + `collapse_check_q0.5_steps50.json`,
  `inperiod_reference.json`, `depth_probe45_run.txt`,
  `hybrid_steps50_run.txt` + `hybrid_progress.json`,
  `leakage_demo_run.txt` + `leakage_demo.json`.
- Superseded (retained as evidence, marked): STEP-1 and K=8
  consoles, old `mechanism_progress.json`, `ordering_effect_progress.json`,
  `bnstats_progress.json` (steps=100), `collapse_check_q0.5.json`
  (steps=100).
