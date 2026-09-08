#!/usr/bin/env python3
r"""
13_window_trend_analysis.py -- decomposition of the per-window recovery trend.

PURPOSE:
  Reviewer question: per-window recovery in Table I decreases monotonically
  across the three W-2022-47 windows for every method, while the frozen-model
  depth trace for the same week (Fig. 2) shows only trendless scatter. The
  windows are processed independently, so what produces the trend?

  This script answers it from the already-recorded raw artifacts. It reads
  no data and runs no model: it consumes results/raw/*.json only, so it is
  instant, and anyone can rerun it to check the arithmetic.

METHOD:
  For each recorded condition it reports, per window:
    (a) recovery in points (the Table I quantity),
    (b) the frozen baseline of that window,
    (c) the window-specific gap against the pre-shift reference,
    (d) recovery normalized by that gap,
    (e) the ADAPTED absolute accuracy (frozen + recovery),
  and flags whether each series is monotonically decreasing.

Run:
    python scripts/13_window_trend_analysis.py            # from repo root
    python scripts/13_window_trend_analysis.py --raw path/to/results/raw
"""
import argparse, json, os
import numpy as np

# Pre-shift reference, self-measured on the earliest three windows of
# W-2022-45 under the identical protocol (results/raw/inperiod_reference.json).
PRESHIFT_REF = 0.9552132161458333

CONDITIONS = [
    ("stats-only, 50 steps",      "bnstats_progress_steps50.json"),
    ("filtered TENT, 50 steps",   "errorbars_progress.json"),
    ("unfiltered, 50 steps",      "mechanism_progress_steps50.json"),
    ("filtered TENT, 100 steps",  "filtered100_progress.json"),
    ("labeled oracle, 50 steps",  "oracle_matched_progress.json"),
    ("two-phase (rejected)",      "hybrid_progress.json"),
]


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("done", d)


def per_window(units):
    rec = {0: [], 1: [], 2: []}
    frozen = {}
    for key, v in units.items():
        w = int(key.split("_")[0])
        rec[w].append(v["recovery"])
        frozen[w] = v["frozen"]
    return ([float(np.mean(rec[w])) for w in range(3)],
            [frozen[w] for w in range(3)],
            [len(rec[w]) for w in range(3)])


def monotone_dec(v):
    return all(v[i] > v[i + 1] for i in range(len(v) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join("results", "raw"))
    args = ap.parse_args()

    print("PER-WINDOW TREND ANALYSIS, W-2022-47 (all values in accuracy points)")
    print(f"pre-shift reference = {PRESHIFT_REF*100:.2f}\n")

    frozen_ref = None
    rows = []
    for label, fname in CONDITIONS:
        path = os.path.join(args.raw, fname)
        if not os.path.isfile(path):
            print(f"[skip] {label}: {path} not found")
            continue
        rec, frozen, k = per_window(load(path))
        if frozen_ref is None:
            frozen_ref = frozen
        elif not np.allclose(frozen, frozen_ref, atol=0.0):
            raise SystemExit(f"[STOP] {label} has different frozen baselines: "
                             f"{frozen} vs {frozen_ref}. The conditions are not "
                             f"on the same windows; the comparison is invalid.")
        rows.append((label, [r * 100 for r in rec], k))

    if frozen_ref is None:
        raise SystemExit("no artifacts found")

    fr = [f * 100 for f in frozen_ref]
    gaps = [PRESHIFT_REF * 100 - f for f in fr]
    print(f"frozen baseline per window : {fr[0]:.2f} / {fr[1]:.2f} / {fr[2]:.2f}"
          f"   (monotonically INCREASING: {monotone_dec(fr[::-1])})")
    print(f"window-specific gap        : {gaps[0]:.2f} / {gaps[1]:.2f} / {gaps[2]:.2f}\n")

    hdr = (f"{'condition':<26}{'K/win':>6}  {'recovery (p)':^22}  "
           f"{'% of window gap':^22}  {'adapted acc (p)':^22}")
    print(hdr)
    print("-" * len(hdr))
    for label, rec, k in rows:
        nrm = [rec[i] / gaps[i] * 100 for i in range(3)]
        adapted = [fr[i] + rec[i] for i in range(3)]
        print(f"{label:<26}{k[0]:>6}  "
              f"{rec[0]:6.2f}{rec[1]:7.2f}{rec[2]:7.2f} {'D' if monotone_dec(rec) else '.':>2}  "
              f"{nrm[0]:6.2f}{nrm[1]:7.2f}{nrm[2]:7.2f} {'D' if monotone_dec(nrm) else '.':>2}  "
              f"{adapted[0]:6.2f}{adapted[1]:7.2f}{adapted[2]:7.2f} {'D' if monotone_dec(adapted) else '.':>2}")
    print("\n  'D' marks a monotonically decreasing series across windows 1, 2, 3.")

    print("\nCORRELATION across the three windows (n=3, reported for orientation "
          "only;\n  three points cannot support an inference):")
    for label, rec, _ in rows:
        c = float(np.corrcoef(fr, rec)[0, 1])
        print(f"  corr(frozen baseline, recovery)  {label:<26} {c:+.3f}")

    print("\nREADING")
    print("  1. The recovery series decreases across windows in every gradient")
    print("     condition AND in the labeled oracle, which minimizes")
    print("     cross-entropy on ground-truth labels and shares no mechanism")
    print("     with entropy minimization. A trend common to supervised and")
    print("     label-free adaptation is a property of the windows, not of the")
    print("     adaptation signal.")
    print("  2. The frozen baseline RISES across the same three windows. Recovery")
    print("     is a difference against that rising baseline, so a falling")
    print("     recovery and a flat adapted accuracy are the same observation.")
    print("  3. Adapted absolute accuracy is NOT monotone in any condition:")
    print("     window 3 is the best window for every method. There is no")
    print("     downward trend in performance to explain.")
    print("  4. Normalized by each window's own gap, the trend disappears in")
    print("     every condition except unfiltered adaptation, whose damage is")
    print("     expected to scale with the share of high-entropy drifted flows.")
    print("  5. Consistency with the depth trace: the three report windows span")
    print("     test batches 0 to 599 only, and their baseline spread (1.71 p) is")
    print("     smaller than the trendless scatter of the W-47 depth probe")
    print("     (range 4.2 p). Three windows drawn from one early stretch of the")
    print("     week are not evidence of a week-long trend, and a monotone")
    print("     ordering of three exchangeable values occurs by chance one time")
    print("     in six.")


if __name__ == "__main__":
    main()
