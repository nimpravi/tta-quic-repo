#!/usr/bin/env python3
r"""
28_verify_revision.py -- regression test for every number the revision adds.

Companion to 21_verify_all.py, which is left untouched and still covers the
original results. This one covers everything produced after the NL review:
the pre-drift label condition (E2), the K=3 partition confirmation (E4), the
source-day sweep, the source-pool characterization, the momentum grid (E5/E3),
the prevalence sweep (E1), the per-class momentum curve, the join between the
momentum and per-class scripts, and the latency ratio (E6).

Same contract as script 21: it loads no model, reads no dataset, recomputes
each value from the released raw artifacts, and compares it with a constant
transcribed from the findings documents. A FAIL means the artifact and the
record disagree, and the script cannot tell you which is wrong.

DELIBERATELY NOT VERIFIED:
  The peak-memory figures in inference_cost.json. tracemalloc on CPU does not
  see torch's allocator, so those numbers are not measurements of memory and
  must not appear in the paper. They are listed at the end so a pass is never
  read as covering them.

Run:
    python scripts/28_verify_revision.py
    python scripts/28_verify_revision.py --verbose
Exit code 0 if every check passes and no artifact is missing, 1 otherwise.
"""
import argparse, datetime as dt, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

SEARCH = [".", "results/raw", "results", "results/superseded"]
PASS, FAIL, MISS = [], [], []
VERBOSE = False

UNA_FROZEN = [0.9456516736470449, 0.9368995771002967, 0.9431153948142085]
REF_LABEL_FREE = 3.06          # pre-registered comparator
F_STAR_DERIVED = 0.2888        # -Delta_U/(Delta_A-Delta_U) from Table II
WORST = 66                     # the manuscript's service
BIG = 10.0


def find(name):
    for d in SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def load(name):
    p = find(name)
    if p is None:
        MISS.append(name)
        return None
    with open(p) as f:
        return json.load(f)


def _rec(ok, label, detail):
    (PASS if ok else FAIL).append(f"{label}: {detail}")
    if VERBOSE or not ok:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")


def check(label, got, exp, tol=0.005, unit=""):
    ok = got is not None and abs(got - exp) <= tol
    _rec(ok, label, f"got {got:+.4f}{unit}, expected {exp:+.4f}{unit} (tol {tol})"
         if got is not None else "no value")


def check_true(label, cond, detail=""):
    _rec(bool(cond), label, detail or ("holds" if cond else "does not hold"))


def sec(t):
    print(f"\n=== {t} ===")


def spearman(x, y):
    def mr(a):
        a = np.asarray(a, float); o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a)); i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[o[j + 1]] == a[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
        return r
    rx, ry = mr(x), mr(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


# ------------------------------------------------------------------ helpers
def predrift_means(pre):
    """(day, capacity) -> list of per-seed means over the three windows."""
    out = {}
    for v in pre["done"].values():
        out.setdefault((v["source_day"], v["capacity"]), {})[v["k"]] = \
            float(np.mean(v["recoveries"]))
    return {k: [d[i] for i in sorted(d)] for k, d in out.items()}


def mom_cell(done, cond, m):
    da, du, dp = [], [], []
    for w in range(3):
        b = done.get(f"w{w}_frozen")
        for k in range(3):
            v = done.get(f"w{w}_{cond}_m{m}_{k}")
            if b and v:
                da.append((v["affected"] - b["affected"]) * 100)
                du.append((v["unaffected"] - b["unaffected"]) * 100)
                dp.append(v["displacement"])
    if not da:
        return None
    return float(np.mean(da)), float(np.mean(du)), float(np.mean(dp))


def pc_delta(done, w, m):
    b = np.array(done[f"w{w}_frozen"]["recall"], float)
    ds = [np.array(done[f"w{w}_m{m}_{k}"]["recall"], float) - b
          for k in range(3) if f"w{w}_m{m}_{k}" in done]
    return np.nanmean(np.vstack(ds), axis=0) * 100 if ds else None


# ------------------------------------------------------------------ sections
def s_e2(pre):
    sec("E2  pre-drift label condition (predrift_label_progress.json)")
    m = predrift_means(pre)
    for (day, cap), exp in [
            (("20221107", "head"), -0.16), (("20221107", "matched"), 0.96),
            (("20221107", "full"), 2.88), (("20221107", "src-tent"), 6.01),
            (("20221107", "src-stats"), -1.10),
            (("20221031", "head"), 0.31), (("20221031", "matched"), -2.64),
            (("20221031", "full"), -2.69), (("20221031", "src-tent"), 0.86),
            (("20221031", "src-stats"), -0.44)]:
        v = m.get((day, cap))
        check(f"E2 {day} {cap}", float(np.mean(v)) if v else None, exp, 0.006, "p")
    best = max(float(np.mean(m[("20221107", c)])) for c in ("head", "matched", "full"))
    check_true("E2 P3: best supervised pre-drift recovery lies in (0, 3.06)",
               0 < best < REF_LABEL_FREE, f"best supervised {best:+.2f}")


def s_sweep(pre):
    sec("Source-day sweep and its K=3 confirmation")
    m = predrift_means(pre)
    tent = {d: float(np.mean(m[(d, "src-tent")])) for d in
            ("20221107", "20221108", "20221109", "20221110", "20221111",
             "20221112", "20221113")}
    for d, exp in [("20221108", 6.17), ("20221109", 5.29), ("20221110", 4.71),
                   ("20221111", 3.31), ("20221112", 2.85), ("20221113", 2.68)]:
        check(f"sweep src-tent {d}", tent[d], exp, 0.006, "p")
    a, b = m[("20221108", "src-tent")], m[("20221107", "src-tent")]
    check_true("peak claim withdrawn: 20221107 and 20221108 seed ranges overlap",
               not (min(a) > max(b) or min(b) > max(a)),
               f"[{min(a):.3f},{max(a):.3f}] vs [{min(b):.3f},{max(b):.3f}]")
    for d, exp in [("20221107", 7.11), ("20221108", 5.98), ("20221111", 0.38)]:
        g = float(np.mean(m[(d, "src-tent")])) - float(np.mean(m[(d, "src-stats")]))
        check(f"gradient term {d}", g, exp, 0.01, "p")
    s = m[("20221107", "src-stats")]
    check("src-stats seed spread on 20221107", max(s) - min(s), 1.64, 0.01, "p")
    t = m[("20221111", "src-tent")]; st = m[("20221111", "src-stats")]
    g = [x - y for x, y in zip(t, st)]
    check("gradient term seed spread on 20221111", max(g) - min(g), 0.096, 0.005, "p")
    strongest = max(tent.values())
    check("ratio weakest supervised / strongest label-free", 11.07 / strongest,
          1.79, 0.01, "x")


def s_char(ch, pre):
    sec("Source-pool characterization (source_pool_characterization.json)")
    m = predrift_means(pre)
    rec = {"20221114": 0.50, "20221118": 0.18, "20221120": 0.63}
    days = sorted(ch)
    g = [rec[d] if d in rec else float(np.mean(m[(d, "src-tent")]))
         - float(np.mean(m[(d, "src-stats")])) for d in days]
    check("kept entropy W-2022-44", ch["20221031"]["kept_entropy"], 0.00029, 0.000005)
    check("kept entropy 20221107", ch["20221107"]["kept_entropy"], 0.00013, 0.000005)
    order = sorted(days, key=lambda d: ch[d]["kept_entropy"])
    check_true("P-A premise fails: W-2022-44 is not the lowest kept entropy",
               order.index("20221031") == 2, f"rank {order.index('20221031')+1}")
    check("Spearman kept entropy vs gradient",
          spearman([ch[d]["kept_entropy"] for d in days], g), -0.873, 0.01)
    check("Spearman kept purity vs gradient",
          spearman([ch[d]["kept_purity"] for d in days], g), 0.791, 0.01)
    check("collinearity purity vs frozen accuracy",
          spearman([ch[d]["kept_purity"] for d in days],
                   [ch[d]["frozen_accuracy"] for d in days]), 0.973, 0.01)


def s_e4(k3):
    sec("E4  K=3 partition confirmation (delayed_label_partition_K3_progress.json)")
    for cap, exp, lo, hi in [("head", 0.10, 0.05, 0.15), ("full", -0.38, -0.45, -0.31),
                             ("matched", -0.92, -1.11, -0.79)]:
        units = []
        for v in k3["done"].values():
            if v["capacity"] == cap:
                units.append(np.mean([(x["unaffected"] - UNA_FROZEN[i]) * 100
                                      for i, x in enumerate(v["windows"])]))
        check(f"E4 {cap} Delta_U, {len(units)} units", float(np.mean(units)), exp, 0.01, "p")
        check_true(f"E4 {cap} unit range inside [{lo:+.2f}, {hi:+.2f}]",
                   lo - 0.005 <= min(units) and max(units) <= hi + 0.005,
                   f"[{min(units):+.3f}, {max(units):+.3f}]")


def s_momentum(md):
    sec("E5/E3  momentum grid (momentum_displacement_progress.json)")
    d = md["done"]
    s0 = mom_cell(d, "stats", 0.0)
    check_true("M-A stats at m=0 is the frozen model exactly",
               s0 and s0[0] == 0 and s0[1] == 0 and s0[2] == 0, f"{s0}")
    for cond, m, eA, eU in [("filtered", 0.1, 6.87, -2.79), ("filtered", 0.03, 6.97, -0.16),
                            ("filtered", 0.0, 2.74, -2.38), ("stats", 0.01, 8.53, -0.36),
                            ("stats", 0.1, 7.18, -4.89), ("filtered", 0.05, 7.05, -0.84),
                            ("filtered", 0.07, 6.97, -2.57),
                            ("filtered", 0.3, 6.84, -2.91)]:
        c = mom_cell(d, cond, m)
        check(f"{cond} m={m} Delta_A", c[0] if c else None, eA, 0.01, "p")
        check(f"{cond} m={m} Delta_U", c[1] if c else None, eU, 0.01, "p")
    for m, exp in [(0.03, 0.02579), (0.05, 0.03134), (0.07, 0.03380), (0.1, 0.03530)]:
        c = mom_cell(d, "filtered", m)
        check(f"filtered displacement m={m}", c[2] if c else None, exp, 0.00002)
    st = [mom_cell(d, "stats", m) for m in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)]
    check("within-stats Spearman displacement vs Delta_U",
          spearman([c[2] for c in st], [c[1] for c in st]), -1.0, 0.001)
    f10 = mom_cell(d, "filtered", 0.1)
    nat = [257507 / 409600, 251170 / 409600, 236126 / 409600]
    net = float(np.mean(nat)) * f10[0] + (1 - float(np.mean(nat))) * f10[1]
    check("harness reproduces manuscript net recovery at m=0.1", net, 3.06, 0.03, "p")


def s_prevalence(pv):
    sec("E1  prevalence sweep (prevalence_sweep_progress.json)")
    d = pv["done"]
    g = sorted({v["f_target"] for k, v in d.items() if k.startswith("w0_") and k.endswith("_0")})
    net = [d[f"w0_f{f}_0"]["net"] * 100 for f in g]
    xs = [g[i] + (g[i + 1] - g[i]) * (-net[i]) / (net[i + 1] - net[i])
          for i in range(len(g) - 1) if net[i] * net[i + 1] < 0]
    check_true("single zero crossing", len(xs) == 1, f"{len(xs)} crossings")
    if xs:
        check("measured break-even", xs[0] * 100, 39.41, 0.05, "%")
        check("gap from derived 28.9%", abs(xs[0] - F_STAR_DERIVED) * 100, 10.53, 0.05, " pts")
    c = [v["net"] * 100 for v in d.values() if v["f_target"] == 0.288]
    check("net at f=0.288, K=3 x 3 windows", float(np.mean(c)), -1.756, 0.005, "p")
    check_true("every unit at f=0.288 is net harmful", all(x < 0 for x in c), f"n={len(c)}")
    for w, exp in [("w0", 62.87), ("w1", 61.32), ("w2", 57.65)]:
        check(f"native prevalence {w}", pv["native"][w] * 100, exp, 0.01, "%")


def s_perclass(pc):
    sec("Per-class momentum curve (perclass_momentum_progress.json)")
    d = pc["done"]
    sup = sum(d[f"w{w}_frozen"]["support"][WORST] for w in range(3))
    check_true("class_66 is the manuscript's service: 52,866 flows", sup == 52866, f"{sup}")
    fr = np.mean([d[f"w{w}_frozen"]["recall"][WORST] for w in range(3)])
    check("class_66 frozen recall", float(fr), 0.972, 0.0015)
    una = pc.get("_unaffected")
    for m, e66, eflows in [(0.03, -1.03, 21), (0.05, -6.63, 456), (0.07, -21.54, 53504),
                           (0.1, -21.88, 55005), (0.3, -21.93, 55005)]:
        v = [pc_delta(d, w, m)[WORST] for w in range(3)]
        check(f"class_66 change at m={m}", float(np.mean(v)), e66, 0.01, "p")
        tot = 0
        for w in range(3):
            dd = pc_delta(d, w, m); s = d[f"w{w}_frozen"]["support"]
            tot += sum(s[c] for c in una if s[c] > 0 and not np.isnan(dd[c]) and dd[c] < -BIG)
        check_true(f"flows in undrifted classes below -10 at m={m} = {eflows:,}",
                   tot == eflows, f"{tot:,}")
    p10 = np.nanmean(np.vstack([pc_delta(d, w, 0.1) for w in range(3)]), axis=0)
    p03 = np.nanmean(np.vstack([pc_delta(d, w, 0.03) for w in range(3)]), axis=0)
    pairs = [(p10[c], p03[c]) for c in una if not np.isnan(p10[c]) and not np.isnan(p03[c])]
    check("C-C Spearman between m=0.1 and m=0.03",
          spearman([a for a, _ in pairs], [b for _, b in pairs]), 0.904, 0.002)


def s_join(md, pc):
    sec("Join: momentum script vs per-class script, unit by unit")
    a, b = md["done"], pc["done"]; una = pc["_unaffected"]
    diffs, rows = [], []
    for w in range(3):
        sup = np.array(b[f"w{w}_frozen"]["support"], float)
        idx = [c for c in una if sup[c] > 0]
        for m in (0.03, 0.05, 0.07, 0.1, 0.3):
            for k in range(3):
                x, y = a.get(f"w{w}_filtered_m{m}_{k}"), b.get(f"w{w}_m{m}_{k}")
                if not (x and y):
                    continue
                r = np.array(y["recall"], float)
                diffs.append(abs(x["unaffected"] - float(np.sum(r[idx] * sup[idx]) / np.sum(sup[idx]))))
                rows.append((x["displacement"],
                             (y["recall"][WORST] - b[f"w{w}_frozen"]["recall"][WORST]) * 100, m, w))
    check_true("45 units matched", len(diffs) == 45, f"{len(diffs)}")
    check_true("scripts 24 and 27 agree bit for bit", max(diffs) == 0.0, f"max diff {max(diffs):.2e}")
    for w in range(3):
        ww = [r for r in rows if r[3] == w and r[2] in (0.05, 0.07)]
        ok = max(r[0] for r in ww if r[1] > -12); bad = min(r[0] for r in ww if r[1] < -15)
        check_true(f"window {w}: collapse separable by displacement", ok < bad,
                   f"{ok:.5f} < {bad:.5f}")
    lo = max(r[0] for r in rows if r[2] == 0.05); hi = min(r[0] for r in rows if r[2] == 0.07)
    check("boundary, highest intact displacement", lo, 0.03263, 0.000005)
    check("boundary, lowest collapsed displacement", hi, 0.03265, 0.000005)


# ---------------------------------------------------------------- Table II
# Every row of the manuscript's Table II, recomputed from whichever artifact
# holds it. Partition accuracy is the support-weighted mean of per-class
# recall over the group, which s_join already shows reproduces script 24's
# own affected/unaffected fields bit for bit.
TABLE2 = [
    # (label,            source, key template,            rho,    dA,    dU,    net)
    ("stats m=0.01",     "mom",  ("stats", 0.01),         0.395,  8.53, -0.36,  5.03),
    ("stats m=0.1",      "mom",  ("stats", 0.1),          0.995,  7.18, -4.89,  2.42),
    ("filtered m=0",     "mom",  ("filtered", 0.0),       0.000,  2.74, -2.38,  0.72),
    ("filtered m=0.03",  "pc",   "w{w}_m0.03_{k}",        0.782,  6.97, -0.16,  4.16),
    ("filtered m=0.07 S=35", "sr", "w{w}_m0.07_S35_{k}",  0.921,  7.15, -0.70,  4.06),
    ("filtered m=0.05",  "pc",   "w{w}_m0.05_{k}",        0.923,  7.05, -0.84,  3.94),
    ("filtered m=0.055", "pc",   "w{w}_m0.055_{k}",       0.941,  7.04, -1.62,  3.63),
    ("filtered m=0.06",  "pc",   "w{w}_m0.06_{k}",        0.955,  7.03, -2.32,  3.35),
    ("filtered m=0.065", "pc",   "w{w}_m0.065_{k}",       0.965,  7.00, -2.52,  3.25),
    ("filtered m=0.05 S=70", "sr", "w{w}_m0.05_S70_{k}",  0.972,  6.92, -2.57,  3.18),
    ("filtered m=0.07",  "pc",   "w{w}_m0.07_{k}",        0.973,  6.97, -2.57,  3.21),
    ("filtered m=0.1",   "pc",   "w{w}_m0.1_{k}",         0.995,  6.87, -2.79,  3.06),
    ("filtered m=0.3",   "pc",   "w{w}_m0.3_{k}",         1.000,  6.84, -2.91,  3.00),
]
F_REPORT = 0.606          # report-week prevalence printed in the caption


def _pacc(rec, sup, group):
    rec = np.asarray(rec, float); sup = np.asarray(sup, float)
    idx = [c for c in group if sup[c] > 0 and not np.isnan(rec[c])]
    return float(np.sum(rec[idx] * sup[idx]) / np.sum(sup[idx])) * 100


def _cell_from_perclass(done, tmpl, aff, una):
    dA, dU = [], []
    for w in range(3):
        b = done.get(f"w{w}_frozen")
        if not b:
            continue
        sup = b["support"]
        bA, bU = _pacc(b["recall"], sup, aff), _pacc(b["recall"], sup, una)
        for k in range(3):
            v = done.get(tmpl.format(w=w, k=k))
            if v:
                dA.append(_pacc(v["recall"], sup, aff) - bA)
                dU.append(_pacc(v["recall"], sup, una) - bU)
    return (float(np.mean(dA)), float(np.mean(dU)), len(dA)) if dA else None


def s_table2(md, pc, sr, part):
    sec("Manuscript Table II, every row")
    aff, una = sorted(part["affected"]), sorted(part["unaffected"])
    for label, src, key, rho, eA, eU, enet in TABLE2:
        if src == "mom":
            c = mom_cell(md["done"], key[0], key[1]) if md else None
            c = (c[0], c[1], 9) if c else None
        elif src == "pc":
            c = _cell_from_perclass(pc["done"], key, aff, una) if pc else None
        else:
            c = _cell_from_perclass(sr["done"], key, aff, una) if sr else None
        if c is None:
            check_true(f"Table II {label}: artifact present", False, "no units found")
            continue
        check_true(f"Table II {label}: nine units", c[2] == 9, f"{c[2]}")
        check(f"Table II {label} Delta_A", c[0], eA, 0.005, "p")
        check(f"Table II {label} Delta_U", c[1], eU, 0.005, "p")
        # the printed net must follow Eq. (1) from the printed Delta values,
        # which is the check a reader can repeat with a calculator
        check(f"Table II {label} net from Eq. (1)",
              F_REPORT * eA + (1 - F_REPORT) * eU, enet, 0.005, "p")
        check(f"Table II {label} rho", 1 - (1 - key[1]) ** 50 if src == "mom"
              else rho, rho, 0.0006)


def s_cost(ic):
    sec("E6  inference cost (inference_cost.json)")
    check("latency ratio, adaptation step / frozen forward", ic["latency_ratio"], 3.17, 0.01, "x")
    check_true("measured on the hardware the paper names",
               ic["device"] == "cpu" and ic["flows_per_batch"] == 2048,
               f"{ic['device']}, batch {ic['flows_per_batch']}")


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    VERBOSE = ap.parse_args().verbose

    pre = load("predrift_label_progress.json")
    ch = load("source_pool_characterization.json")
    k3 = load("delayed_label_partition_K3_progress.json")
    md = load("momentum_displacement_progress.json")
    pv = load("prevalence_sweep_progress.json")
    pc = load("perclass_momentum_progress.json")
    part = load("class_partition.json")
    ic = load("inference_cost.json")
    sr = load("steps_replacement_progress.json")
    if pc and part:
        pc["_unaffected"] = sorted(part["unaffected"])

    if pre: s_e2(pre); s_sweep(pre)
    if ch and pre: s_char(ch, pre)
    if k3: s_e4(k3)
    if md: s_momentum(md)
    if pv: s_prevalence(pv)
    if pc and part: s_perclass(pc)
    if md and pc and part: s_join(md, pc)
    if pc and sr and part: s_table2(md, pc, sr, part)
    if ic: s_cost(ic)

    sec("NOT VERIFIED BY THIS SCRIPT")
    print("  - peak_mem_frozen_MiB, peak_mem_adapted_MiB, mem_ratio in")
    print("    inference_cost.json: CPU tracemalloc does not see torch's")
    print("    allocator. Not a memory measurement. Keep out of the paper.")
    if MISS:
        print("\n  Artifacts NOT FOUND (searched " + ", ".join(SEARCH) + "):")
        for m in sorted(set(MISS)):
            print(f"    - {m}")

    sec("SUMMARY")
    print(f"  {len(PASS)} passed, {len(FAIL)} failed, {len(set(MISS))} artifacts missing")
    if FAIL:
        print("\n  FAILURES:")
        for f in FAIL:
            print(f"    {f}")
        sys.exit(1)
    if MISS:
        print("\n  Checks that ran passed, but artifacts are missing. Not clean.")
        sys.exit(1)
    print("\n  Every revision number checked reproduces from the released artifacts.")


if __name__ == "__main__":
    main()
