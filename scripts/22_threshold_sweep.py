#!/usr/bin/env python3
r"""
22_threshold_sweep.py -- post-hoc robustness of the class partition.

DECLARED POST-HOC. This analysis is not in
PREREGISTRATION_streaming_delayed_label.md. It was prompted by a sensitivity
check on class_partition.json performed after v4 was complete, and its
decision rules are fixed in ADDENDUM_partition_threshold.md, hash-locked
before the C2 rerun. The pre-registered 10-point results remain primary and
are never replaced by anything here.

WHAT IT DOES:
  Reads the per-class counts that 18_nondrifted_control.py --c2 now records,
  together with class_partition.json, and answers two questions with no
  further model runs.

  1. THRESHOLD SWEEP. Recomputes the affected/unaffected decomposition and
     the break-even prevalence at every cut, not just the pre-registered one.

  2. THE CONTINUOUS VERSION. Drops the binary partition entirely and relates
     each class's accuracy change under adaptation to its measured drift
     magnitude. If the mechanism in the manuscript holds, that relationship
     is increasing. This is the analysis that makes the threshold stop being
     load-bearing.

  It then evaluates R1 to R4 of the addendum and prints a verdict for each.

ANCHORS: refuses to report anything unless recomputing the 10-point partition
from the per-class counts reproduces the recorded aggregates.

Run:
    python scripts/22_threshold_sweep.py
    python scripts/22_threshold_sweep.py --min-support 100
"""
import argparse, json, os, sys
import numpy as np

C2_CKPT   = "nondrifted_c2_progress.json"
PART_JSON = "class_partition.json"
N_WINDOWS = 3
CONDS     = ("stats", "filtered")
SWEEP     = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25)
PREREG    = 0.10

# Recorded under the pre-registered partition (RESULTS.md 14.2), the anchors
# this script refuses to proceed without.
ANCHOR = {"filtered": (6.87, -2.79), "stats": (7.18, -4.89)}

_SEARCH = [".", "results/raw", "results"]


def _resolve(name):
    if os.path.isabs(name) or os.path.isfile(name):
        return name
    for d in _SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return name


def load(name):
    p = _resolve(name)
    if not os.path.isfile(p):
        sys.exit(f"[STOP] {name} not found (searched {', '.join(_SEARCH)}).")
    with open(p) as f:
        return json.load(f)


def spearman(x, y):
    """Rank correlation without a scipy dependency. Ties get average ranks."""
    def rank(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        # average ranks within tied groups
        s = a[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(np.arange(i, j + 1))
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def decompose(units, drop, thr, min_support=0):
    """Return (d_affected, d_unaffected, f) per window at a given threshold,
    in accuracy points, K-averaged. All arithmetic, no model."""
    out = []
    for w in range(N_WINDOWS):
        fz = units[f"w{w}_frozen"]
        tot = np.asarray(fz["per_class_total"], dtype="int64")
        cf = np.asarray(fz["per_class_correct"], dtype="int64")
        n_classes = len(tot)
        aff = np.zeros(n_classes, dtype=bool)
        for c in range(n_classes):
            d = drop.get(c)
            if d is not None and tot[c] >= min_support:
                aff[c] = d > thr
        una = (tot > 0) & (~aff)
        una &= np.array([drop.get(c) is not None for c in range(n_classes)])
        na, nu = int(tot[aff].sum()), int(tot[una].sum())
        if na == 0 or nu == 0:
            out.append(None)
            continue
        base_a = cf[aff].sum() / na
        base_u = cf[una].sum() / nu
        row = {"f": na / (na + nu), "n_affected": na, "n_unaffected": nu}
        for cond in CONDS:
            ks = [k for k in range(10) if f"w{w}_{cond}_{k}" in units]
            da, du = [], []
            for k in ks:
                ca = np.asarray(units[f"w{w}_{cond}_{k}"]["per_class_correct"],
                                dtype="int64")
                da.append(ca[aff].sum() / na - base_a)
                du.append(ca[una].sum() / nu - base_u)
            row[cond] = (float(np.mean(da)) * 100, float(np.mean(du)) * 100,
                         len(ks))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, default=0,
                    help="exclude classes below this many flows in the window "
                         "from the AFFECTED side; declared, not pre-registered")
    args = ap.parse_args()

    part = load(PART_JSON)
    ck = load(C2_CKPT)
    units = ck.get("done", ck)
    if "per_class_total" not in units.get("w0_frozen", {}):
        sys.exit(f"[STOP] {C2_CKPT} has no per-class counts. Rerun\n"
                 f"  python scripts/18_nondrifted_control.py --c2 --size S --K 3\n"
                 f"with the patched script first. This analysis adds no model "
                 f"runs of its own but it cannot invent the counts.")

    C = part["classes"]
    drop = {}
    for k, v in C.items():
        if v["recall_W45"] is not None and v["recall_W46"] is not None:
            drop[int(k)] = v["recall_W45"] - v["recall_W46"]
    name = {int(k): v["name"] for k, v in C.items()}

    # ---------------------------------------------------------- anchor
    print("=" * 72)
    print("0. ANCHOR: the pre-registered partition must reproduce from counts")
    print("=" * 72)
    ref = decompose(units, drop, PREREG)
    ok = True
    for cond in CONDS:
        da = float(np.mean([r[cond][0] for r in ref if r]))
        du = float(np.mean([r[cond][1] for r in ref if r]))
        ea, eu = ANCHOR[cond]
        good = abs(da - ea) <= 0.02 and abs(du - eu) <= 0.02
        ok &= good
        print(f"  {cond:>9}: affected {da:+.2f}p (recorded {ea:+.2f}), "
              f"unaffected {du:+.2f}p (recorded {eu:+.2f})  "
              f"{'OK' if good else 'MISMATCH'}")
    if not ok:
        sys.exit("\n[STOP] the rerun does not reproduce the recorded "
                 "decomposition. Nothing below is comparable with the record. "
                 "Establish why before reporting anything.")
    print("  Reproduced. The rerun is comparable with the record.\n")

    # ---------------------------------------------------------- sweep
    print("=" * 72)
    print("1. THRESHOLD SWEEP (post-hoc)")
    print("=" * 72)
    hdr = f"  {'thr':>6} {'classes':>8} {'f':>7}"
    for cond in CONDS:
        hdr += f" {cond + ' aff':>11} {cond + ' unaff':>13} {'f*':>7}"
    print(hdr)
    sweep_rows = {}
    for thr in SWEEP:
        rows = decompose(units, drop, thr, args.min_support)
        rows = [r for r in rows if r]
        if not rows:
            continue
        n_aff = sum(1 for c, d in drop.items() if d > thr)
        f = float(np.mean([r["f"] for r in rows]))
        rec = {"f": f, "n_affected_classes": n_aff}
        cells = []
        for cond in CONDS:
            da = float(np.mean([r[cond][0] for r in rows]))
            du = float(np.mean([r[cond][1] for r in rows]))
            fstar = float(np.mean([-r[cond][1] / (r[cond][0] - r[cond][1])
                                   for r in rows]))
            rec[cond] = {"d_affected": da, "d_unaffected": du,
                         "f_star": fstar if du < 0 else None}
            cells += [f"{da:11.2f}", f"{du:13.2f}",
                      (f"{fstar*100:6.1f}%" if du < 0 else "   n/a")]
        sweep_rows[thr] = rec
        mark = " <- pre-registered" if abs(thr - PREREG) < 1e-9 else ""
        print(f"  {thr:6.3f} {n_aff:8d} {f*100:6.1f}% " + " ".join(cells) + mark)

    # ------------------------------------------------------- continuous
    print("\n" + "=" * 72)
    print("2. THE CONTINUOUS VERSION: per-class effect against drift magnitude")
    print("=" * 72)
    per_class = {}
    for w in range(N_WINDOWS):
        fz = units[f"w{w}_frozen"]
        tot = np.asarray(fz["per_class_total"], dtype="int64")
        cf = np.asarray(fz["per_class_correct"], dtype="int64")
        for cond in CONDS:
            ks = [k for k in range(10) if f"w{w}_{cond}_{k}" in units]
            ca = np.mean([np.asarray(units[f"w{w}_{cond}_{k}"]["per_class_correct"],
                                     dtype="int64") for k in ks], axis=0)
            for c in drop:
                if c < len(tot) and tot[c] > 0:
                    d = (ca[c] - cf[c]) / tot[c] * 100
                    per_class.setdefault((cond, c), []).append((d, int(tot[c])))
    for cond in CONDS:
        cs = sorted({c for (cc, c) in per_class if cc == cond})
        for floor in (0, 100, 1000):
            xs, ys = [], []
            for c in cs:
                vals = per_class[(cond, c)]
                sup = float(np.mean([v[1] for v in vals]))
                if sup < floor:
                    continue
                xs.append(drop[c]); ys.append(float(np.mean([v[0] for v in vals])))
            if len(xs) < 8:
                continue
            r = spearman(xs, ys)
            print(f"  {cond:>9}, classes with >= {floor:5d} flows "
                  f"(n={len(xs):3d}): Spearman rho = {r:+.3f}")
    print("\n  Ten most drifted classes and what adaptation does to each:")
    print(f"    {'class':<24}{'drift':>8}{'stats':>9}{'filtered':>10}{'flows':>9}")
    for c in sorted(drop, key=lambda c: -drop[c])[:10]:
        row = []
        for cond in CONDS:
            v = per_class.get((cond, c))
            row.append(f"{np.mean([x[0] for x in v]):+8.2f}" if v else "     n/a")
        sup = per_class.get(("filtered", c))
        s = int(np.mean([x[1] for x in sup])) if sup else 0
        print(f"    {name.get(c, c):<24}{drop[c]*100:7.1f}p{row[0]}{row[1]}{s:9d}")

    # ------------------------------------------------------------ rules
    print("\n" + "=" * 72)
    print("3. ADDENDUM DECISION RULES")
    print("=" * 72)
    r1 = [t for t, r in sweep_rows.items()
          if t in (0.05, 0.10, 0.15, 0.20) and r["filtered"]["d_unaffected"] >= 0]
    print("  R1 sign            : "
          + (f"FIRES at thresholds {r1}; the filtered change on unaffected "
             f"traffic is no longer negative there, so the sign of the "
             f"transfer is threshold-dependent and the abstract must say so"
             if r1 else
             "holds; the filtered loss on unaffected traffic stays negative "
             "at every swept cut"))
    if 0.15 in sweep_rows:
        du15 = sweep_rows[0.15]["filtered"]["d_unaffected"]
        print(f"  R2 magnitude       : filtered loss at 0.15 is {du15:+.2f}p"
              f"  -> {'FIRES, report a range' if abs(du15) < 1.0 else 'holds'}")
    r3 = [t for t, r in sweep_rows.items()
          if r["filtered"]["f_star"] is not None
          and r["f"] < r["filtered"]["f_star"]]
    print(f"  R3 operating point : "
          f"{'FIRES at ' + str(r3) if r3 else 'holds at every swept cut'}")
    print(f"  R4 mechanism       : read the Spearman values in section 2; the "
          f"rule asks for a positive rank correlation.")
    print(f"  R5                 : the pre-registered 10-point results stand as "
          f"primary; nothing above replaces Table II.")

    out = {"declared": "post-hoc, ADDENDUM_partition_threshold.md",
           "prereg_threshold": PREREG, "min_support": args.min_support,
           "sweep": {str(k): v for k, v in sweep_rows.items()}}
    with open("threshold_sweep.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\n  written to threshold_sweep.json")


if __name__ == "__main__":
    main()
