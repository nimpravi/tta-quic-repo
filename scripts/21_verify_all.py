#!/usr/bin/env python3
r"""
21_verify_all.py -- regression test for every number in the paper.

WHAT IT IS:
  A single script that recomputes, from the released raw artifacts alone,
  every value destined for the manuscript, and compares each against the value
  recorded here. It loads no model, reads no dataset, and takes seconds. A
  reviewer can clone the repository and run it.

  Each expected constant below is transcribed from the manuscript or
  RESULTS.md. A FAIL therefore means one of two things, and the script cannot
  tell you which: either an artifact changed, or the manuscript is wrong. Both
  are worth knowing before submission.

WHAT IT DELIBERATELY DOES NOT DO:
  It does not re-run any experiment, so it cannot detect a wrong experiment,
  only an inconsistent record. It also lists, by name, every artifact it does
  NOT parse, so that "the verification passed" can never be mistaken for "all
  artifacts were checked".

Run:
    python scripts/21_verify_all.py
    python scripts/21_verify_all.py --verbose
Exit code 0 if every check passes, 1 otherwise.
"""
import argparse, json, os, sys
import numpy as np

SEARCH = [".", "results/raw", "results", "results/superseded"]
PASS, FAIL, MISS = [], [], []
VERBOSE = False


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


def check(label, got, exp, tol=0.005, unit=""):
    if got is None:
        FAIL.append(f"{label}: could not compute")
        return False
    ok = abs(got - exp) <= tol
    (PASS if ok else FAIL).append(
        f"{label}: got {got:.4f}{unit}, expected {exp:.4f}{unit}"
        + ("" if ok else f"  (tol {tol})"))
    if VERBOSE or not ok:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got:.4f} vs {exp:.4f}")
    return ok


def check_exact(label, got, exp):
    ok = (got == exp)
    (PASS if ok else FAIL).append(f"{label}: {got!r} vs {exp!r}")
    if VERBOSE or not ok:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got!r} vs {exp!r}")
    return ok


def units(d):
    return d.get("done", d)


def pooled(name):
    """(pooled mean, pooled std, per-window means, frozen per window), in points."""
    d = load(name)
    if d is None:
        return None
    u = units(d)
    pw = {0: [], 1: [], 2: []}
    fz = {}
    for k, v in u.items():
        w = int(k.split("_")[0])
        pw[w].append(v["recovery"])
        fz[w] = v["frozen"]
    allr = [v["recovery"] for v in u.values()]
    return (np.mean(allr) * 100, np.std(allr) * 100,
            [np.mean(pw[w]) * 100 for w in range(3)],
            [fz[w] for w in range(3)])


# ---- anchors that must hold everywhere -------------------------------------
W47 = [0.72239013671875, 0.72425537109375, 0.73946044921875]
W45 = [0.955947265625, 0.95069580078125, 0.95899658203125]
REF = 0.9552132161458333
W46_60 = 0.7534749348958333


def sec(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    VERBOSE = a.verbose

    # ---------------------------------------------------------------- drift
    sec("1. DRIFT MAGNITUDE AND THE GAP DENOMINATOR")
    ip = load("inperiod_reference.json")
    if ip:
        for i, v in enumerate(ip["W-2022-45"]["per_window"]):
            check_exact(f"W-45 window {i+1} frozen", v, W45[i])
        check("W-45 mean (pre-shift reference)", ip["W-2022-45"]["mean"], REF, 1e-12)
        for i, v in enumerate(ip["W-2022-46"]["per_window"]):
            check(f"W-46 window {i+1} frozen", v,
                  [0.72326171875, 0.72526123046875, 0.74511474609375][i], 1e-12)
    eb = pooled("errorbars_progress.json")
    if eb:
        for i in range(3):
            check_exact(f"W-47 window {i+1} frozen", eb[3][i], W47[i])
        gap = (REF - np.mean(W47)) * 100
        check("gap (points)", gap, 22.65, 0.005)

    # ------------------------------------------------------- Table I and II
    sec("2. EPISODIC DECOMPOSITION (Table I) AND STEP COUNT (Table II)")
    spec = [("stats-only, 50 steps", "bnstats_progress_steps50.json",
             2.43, 0.15, [2.64, 2.35, 2.31]),
            ("filtered, 50 steps", "errorbars_progress.json",
             3.06, 0.27, [3.42, 2.95, 2.81]),
            ("unfiltered, 50 steps", "mechanism_progress_steps50.json",
             1.32, 0.53, [1.99, 1.18, 0.78]),
            ("filtered, 100 steps", "filtered100_progress.json",
             2.96, 0.27, [3.34, 2.82, 2.73]),
            ("labeled oracle, 50 steps", "oracle_matched_progress.json",
             11.55, 0.38, [12.02, 11.52, 11.10]),
            ("two-phase (rejected)", "hybrid_progress.json",
             4.31, None, [4.90, 4.00, 4.02])]
    frozen_seen = {}
    for label, fn, m, s, pw in spec:
        r = pooled(fn)
        if r is None:
            continue
        check(f"{label} pooled mean", r[0], m, 0.006, "p")
        if s is not None:
            check(f"{label} pooled std", r[1], s, 0.006, "p")
        for i in range(3):
            check(f"{label} window {i+1}", r[2][i], pw[i], 0.006, "p")
        frozen_seen[label] = r[3]
    if frozen_seen:
        ok = all(f == W47 for f in frozen_seen.values())
        (PASS if ok else FAIL).append(
            "frozen W-47 accuracies bit-identical across all Table I/II artifacts")
        print(f"  {'PASS' if ok else 'FAIL'}  frozen W-47 accuracies "
              f"bit-identical across {len(frozen_seen)} artifacts")
    h = pooled("hybrid_progress.json")
    if h:
        n = 9
        ddof1 = h[1] * np.sqrt(n / (n - 1))
        print(f"  NOTE  two-phase std is {h[1]:.2f}p at ddof=0 and "
              f"{ddof1:.2f}p at ddof=1. RESULTS.md section 6 records 0.49, "
              f"which is the ddof=1 value; every other std in the record is "
              f"ddof=0. Make them consistent.")

    # ------------------------------------------------------------ Table IV
    sec("3. PRE-REGISTERED SWITCH-POINT SELECTION (Table IV)")
    sp = load("switchpoint_select.json")
    if sp:
        g = {}
        for v in sp.values():
            g.setdefault(v["switch"], []).append(v["recovery"] * 100)
        exp = {25: (5.11, 0.27), 37: (3.98, 0.30), 50: (3.46, 0.18),
               62: (2.45, 1.88), 75: (2.23, 1.72)}
        for s, (m, sd) in exp.items():
            arr = np.array(g[s])
            check(f"switch {s} mean", arr.mean(), m, 0.006, "p")
            check(f"switch {s} order-std", arr.std(), sd, 0.006, "p")
        below = sorted(round(min(g[s]), 2) for s in (62, 75))
        check("switch 62 worst ordering", min(g[62]), -1.28, 0.006, "p")
        check("switch 75 worst ordering", min(g[75]), -1.13, 0.006, "p")
        fr = {v["frozen"] for v in sp.values()}
        check_exact("W-46 frozen identical across all 25 selection units",
                    len(fr) == 1 and fr.pop() == W46_60, True)

    # ------------------------------------------------------- Experiment D
    sec("4. EXPERIMENT D: matched stability reference")
    d = load("w46_stability_reference.json")
    if d:
        r = np.array([d[str(k)]["recovery"] for k in range(5)])
        check("D mean", r.mean(), 2.42, 0.006, "p")
        check("D order-std", r.std(), 0.09, 0.006, "p")
        check_exact("D frozen matches the switch-point anchor",
                    d["0"]["frozen"], W46_60)
        print(f"  NOTE  the matched order-std is {r.std():.2f}p against the "
              f"0.06p the 0.12p ceiling was built from. A matched ceiling "
              f"would have been {2*r.std():.2f}p; the selected switch point's "
              f"0.27p still exceeds it, so the rejection survives.")

    # ------------------------------------------------------- Experiment E
    sec("5. EXPERIMENT E: across-day replication")
    e = load("acrossday_progress.json")
    if e:
        u = units(e)
        days = sorted({k.split("_")[0] for k in u})
        per = {}
        for cond in ("stats", "filtered"):
            means = []
            for dy in days:
                v = [u[k]["recovery"] for k in u
                     if k.startswith(dy + "_" + cond + "_")]
                if v:
                    means.append(np.mean(v) * 100)
            per[cond] = np.array(means)
        check("E filtered across-day mean", per["filtered"].mean(), 3.04, 0.02, "p")
        check("E filtered across-day std", per["filtered"].std(), 0.24, 0.02, "p")
        check("E stats across-day mean", per["stats"].mean(), 2.22, 0.02, "p")
        check("E no day below the frozen baseline",
              float(min(per["filtered"].min(), per["stats"].min())) > 0, True, 0.5)
        fz = u.get("20221121_frozen", {}).get("accuracy")
        if fz is not None:
            check_exact("E 20221121 frozen reproduces Table I window 1", fz, W47[0])
        # E's 20221121 units use the same seeds as Table I window 1
        eb_raw = units(load("errorbars_progress.json") or {})
        if eb_raw:
            a = [round(u[f"20221121_filtered_{k}"]["recovery"] * 100, 4)
                 for k in range(3) if f"20221121_filtered_{k}" in u]
            b = [round(eb_raw[f"0_{k}"]["recovery"] * 100, 4) for k in range(3)]
            ok = a == b
            (PASS if ok else FAIL).append(
                "E 20221121 filtered units bit-reproduce Table I window 1")
            print(f"  {'PASS' if ok else 'FAIL'}  E 20221121 filtered units "
                  f"reproduce Table I window 1 exactly: {a} vs {b}")

    # ------------------------------------------------------- Experiment A
    sec("6. EXPERIMENT A: streaming")
    s = load("streaming_results.json")
    cfgA = load("streaming_config.json")
    if cfgA:
        check("A selected lr", cfgA["lr"], 1e-4, 1e-12)
        check("A selected quantile", cfgA["quant"], 0.5, 1e-12)
    if s:
        fw = s["frozen"]["per_window"]
        for i in range(3):
            check_exact(f"A frozen window {i+1}", fw[f"window_{i+1}"], W47[i])
        cf = s["causal-filtered"]["per_window"]
        rec = [(cf[f"window_{i+1}"] - fw[f"window_{i+1}"]) * 100 for i in range(3)]
        check("A causal-filtered 3-window mean recovery",
              float(np.mean(rec)), 2.88, 0.006, "p")
        check("A causal-filtered full-week recovery",
              (s["causal-filtered"]["accuracy_overall"]
               - s["frozen"]["accuracy_overall"]) * 100, 2.63, 0.006, "p")
        check("A batch-transductive buffer value (vs causal)",
              float(np.mean([(s["batchtrans-filtered"]["per_window"][f"window_{i+1}"]
                              - cf[f"window_{i+1}"]) * 100 for i in range(3)])),
              -0.20, 0.02, "p")
        check("A full-week batch count", s["frozen"]["n_batches"], 3227, 0)

    # ------------------------------------------------------------ ordering
    sec("7. A0: STREAM ORDER AUDIT")
    au = load("streaming_order_audit.json")
    if au:
        for wk, nb, nf in (("W-2022-46", 2464, 5044543), ("W-2022-47", 3227, 6607244)):
            r = au[wk]
            check_exact(f"{wk} uncapped", r["capped"], False)
            check(f"{wk} batches", r["n_batches"], nb, 0)
            check(f"{wk} flows", r["n_flows"], nf, 0)
            check(f"{wk} batch-order criterion",
                  r["batch_median_in_order_fraction"], 1.0, 1e-9)
            check(f"{wk} displacement p99.9 below one batch",
                  float(r["time_last_displacement_positions"]["99.9"] < 2048), 1.0, 0)
        r = au["W-2022-47"]
        for i, hrs in enumerate([11.19, 4.46, 8.12]):
            check(f"W-47 report window {i+1} span (hours)",
                  r["report_windows"][f"window_{i+1}"]["span_hours"], hrs, 0.02)
        days = [d for d, v in r["day_map_from_indices"].items() if v.get("flows")]
        allin = all(r["report_window_days"][f"window_{i}"] == ["20221121"]
                    for i in (1, 2, 3))
        check_exact("all three W-47 report windows fall inside 20221121", allin, True)

    # ------------------------------------------------------- Experiment B
    sec("8. EXPERIMENT B: delayed-label baselines")
    b = load("delayed_label_progress.json")
    cfgB = load("delayed_label_config.json")
    if cfgB:
        for cap, lr, st in (("matched", 1e-3, 100), ("head", 1e-3, 100),
                            ("full", 1e-4, 100), ("src-tent", 1e-3, 50)):
            check(f"B {cap} selected steps", cfgB["per_capacity"][cap]["steps"], st, 0)
            if lr is not None:
                check(f"B {cap} selected lr", cfgB["per_capacity"][cap]["lr"], lr, 1e-12)
        check_exact("B src-stats has no learning rate",
                    cfgB["per_capacity"]["src-stats"]["lr"], None)
    if b:
        u = units(b)
        exp = {(1, "matched"): 13.55, (3, "matched"): 13.43, (7, "matched"): 12.04,
               (1, "head"): 11.07, (3, "head"): 11.28, (7, "head"): 11.21,
               (1, "full"): 14.20, (3, "full"): 14.72, (7, "full"): 13.51,
               (1, "src-stats"): 2.21, (3, "src-stats"): 2.74, (7, "src-stats"): 2.16,
               (1, "src-tent"): 2.84, (3, "src-tent"): 2.92, (7, "src-tent"): 2.66}
        for (dl, cap), m in exp.items():
            v = [np.mean(x["recoveries"]) for x in u.values()
                 if x["delta"] == dl and x["capacity"] == cap]
            check(f"B {cap} delta={dl}d", float(np.mean(v)), m, 0.02, "p")
        sup = max(np.mean([np.mean(x["recoveries"]) for x in u.values()
                           if x["delta"] == dl and x["capacity"] == c])
                  for dl in (1, 3, 7) for c in ("matched", "head", "full"))
        check("B-K1 margin over the label-free comparator", sup - 3.06, 11.66, 0.02, "p")
        print(f"  NOTE  B-K1 fired at every delta. The worst supervised "
              f"baseline at any delay still exceeds the best label-free "
              f"result anywhere in the study.")

    # ------------------------------------------------------- Experiment C
    sec("9. EXPERIMENT C: cost on non-drifted traffic")
    c1 = load("nondrifted_c1_progress.json")
    if c1:
        u = units(c1)
        for cond, exp in (("stats", [-0.09, -0.07, -0.04]),
                          ("filtered", [-0.32, -0.29, -0.30])):
            for w in range(3):
                v = [u[k]["recovery"] for k in u
                     if k.startswith(f"W-2022-45_w{w}_{cond}_")]
                if v:
                    check(f"C1 W-45 window {w+1} {cond}", float(np.mean(v)),
                          exp[w], 0.02, "p")
    c2 = load("nondrifted_c2_progress.json")
    part = load("class_partition.json")
    if part:
        check("partition affected classes", len(part["affected"]), 29, 0)
        check("partition unaffected classes", len(part["unaffected"]), 73, 0)
        goog = sum(1 for c in part["affected"]
                   if part["classes"][str(c)].get("provider") == "google")
        check("google classes in the affected set", goog, 21, 0)
    if c2:
        u = units(c2)
        rec = {"stats": [2.64, 2.35, 2.31], "filtered": [3.42, 2.95, 2.81]}
        print("  reconstruction of Table I from the class partition:")
        for w in range(3):
            f0 = u[f"w{w}_frozen"]
            frac = f0["n_affected"] / (f0["n_affected"] + f0["n_unaffected"])
            for cond in ("stats", "filtered"):
                v = [u[f"w{w}_{cond}_{k}"] for k in range(3)
                     if f"w{w}_{cond}_{k}" in u]
                da = (np.mean([x["affected"] for x in v]) - f0["affected"]) * 100
                du = (np.mean([x["unaffected"] for x in v]) - f0["unaffected"]) * 100
                net = frac * da + (1 - frac) * du
                tol = 0.05 if cond == "stats" else 0.06   # filtered Table I is K=5
                check(f"C2 window {w+1} {cond} net reconstructs Table I",
                      net, rec[cond][w], tol, "p")
        das = {c: [] for c in ("stats", "filtered")}
        dus = {c: [] for c in ("stats", "filtered")}
        for w in range(3):
            f0 = u[f"w{w}_frozen"]
            for cond in ("stats", "filtered"):
                v = [u[f"w{w}_{cond}_{k}"] for k in range(3)]
                das[cond].append((np.mean([x["affected"] for x in v]) - f0["affected"]) * 100)
                dus[cond].append((np.mean([x["unaffected"] for x in v]) - f0["unaffected"]) * 100)
        check("C2 filtered gain on affected", float(np.mean(das["filtered"])), 6.87, 0.02, "p")
        check("C2 filtered loss on unaffected", float(np.mean(dus["filtered"])), -2.79, 0.02, "p")
        check("C2 stats gain on affected", float(np.mean(das["stats"])), 7.18, 0.02, "p")
        check("C2 stats loss on unaffected", float(np.mean(dus["stats"])), -4.89, 0.02, "p")
        be = {c: float(np.mean([-dus[c][i] / (das[c][i] - dus[c][i]) for i in range(3)]))
              for c in ("stats", "filtered")}
        check("break-even drifted fraction, filtered", be["filtered"], 0.288, 0.005)
        check("break-even drifted fraction, stats-only", be["stats"], 0.404, 0.005)

    # --------------------------------------------------- post-hoc partition
    sec("10. POST-HOC PARTITION OF THE SUPERVISED CONDITIONS (not pre-registered)")
    ph = load("delayed_label_partition_progress.json")
    if ph and c2:
        u2, uc = units(ph), units(c2)
        agg = {}
        for k, v in u2.items():
            da, du = [], []
            for i in range(3):
                f0 = uc[f"w{i}_frozen"]
                da.append((v["windows"][i]["affected"] - f0["affected"]) * 100)
                du.append((v["windows"][i]["unaffected"] - f0["unaffected"]) * 100)
            agg.setdefault(v["capacity"], ([], []))
            agg[v["capacity"]][0].extend(da); agg[v["capacity"]][1].extend(du)
        exp = {"head": (18.30, 0.11), "matched": (22.13, -0.91),
               "full": (23.61, -0.40), "matched+tta": (20.25, -1.69),
               "src-stats": (7.11, -5.25), "src-tent": (6.79, -3.50)}
        for cap, (ea, eu) in exp.items():
            if cap not in agg:
                continue
            check(f"post-hoc {cap} on affected", float(np.mean(agg[cap][0])), ea, 0.02, "p")
            check(f"post-hoc {cap} on unaffected", float(np.mean(agg[cap][1])), eu, 0.02, "p")
        # the k=0 units must bit-reproduce Experiment B
        if b:
            ub = units(b)
            bad = []
            for k, v in u2.items():
                key = f"d{v['delta']}_{v['capacity']}_0"
                if key not in ub:
                    continue
                a = ub[key]["accuracies"]
                c = [w["overall"] for w in v["windows"]]
                if not all(abs(x - y) < 1e-12 for x, y in zip(a, c)):
                    bad.append(f"{v['capacity']} d={v['delta']}")
            ok = (bad == ["full d=1", "full d=3", "full d=7"] or bad == [])
            (PASS if ok else FAIL).append(
                f"post-hoc k=0 units reproduce Experiment B (except {bad})")
            print(f"  {'PASS' if ok else 'FAIL'}  post-hoc k=0 units reproduce "
                  f"Experiment B bit-for-bit except: {bad or 'nothing'}")
            if bad:
                print(f"  NOTE  'full' is the only condition that puts the whole "
                      f"model in training mode. If the architecture contains "
                      f"dropout, that condition is stochastic and unseeded, and "
                      f"the repository's determinism claim needs an explicit "
                      f"exception for it. Verify by listing the model's Dropout "
                      f"modules; do not assert it without checking.")

    # ------------------------------------------------------------ coverage
    sec("11. COVERAGE")
    not_parsed = ["collapse_check_q0.5_steps50.json (Table III class-level "
                  "behaviour)", "leakage_demo.json (section III-E audit)",
                  "w45_depth_probe.json (Figure 2)",
                  "switchpoint_probe.json (superseded coarse probe)"]
    print("  Artifacts this script does NOT parse, and which therefore remain")
    print("  unverified by it. Check these by hand before submission:")
    for n in not_parsed:
        print(f"    - {n}")
    if MISS:
        print("\n  Artifacts referenced but NOT FOUND (searched "
              f"{', '.join(SEARCH)}):")
        for m in sorted(set(MISS)):
            print(f"    - {m}")

    sec("SUMMARY")
    print(f"  {len(PASS)} passed, {len(FAIL)} failed, "
          f"{len(set(MISS))} artifacts missing")
    if FAIL:
        print("\n  FAILURES:")
        for f in FAIL:
            print(f"    {f}")
        print("\n  A failure means the artifact and the recorded value disagree.")
        print("  This script cannot tell you which one is wrong. Find out before")
        print("  the number reaches the manuscript.")
        sys.exit(1)
    if MISS:
        print("\n  All checks that could run passed, but artifacts are missing,")
        print("  so this is NOT a clean verification. Locate them and rerun.")
        sys.exit(1)
    print("\n  Every number checked reproduces from the released artifacts.")


if __name__ == "__main__":
    main()
