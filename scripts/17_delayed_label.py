#!/usr/bin/env python3
r"""
17_delayed_label.py -- Experiment B of the streaming/delayed-label
pre-registration: what supervised retraining delivers when labels arrive late.

THE QUESTION:
  Label-free adaptation is motivated by labels being unavailable at inference
  time. The operational alternative is not "no labels ever", it is "labels
  arrive N days late, then retrain". This script measures that alternative on
  the same flows, under the same evaluation code path, so the two can be put
  in one table.

DESIGN:
  Evaluation is on the three report windows of W-2022-47, batches 0 to 599,
  which the stream-order audit placed inside 20221121. For a label delay of
  delta days, the retraining data is the labeled traffic of
  (evaluation day - delta). With the default evaluation day of 20221121 and
  delta in {1, 3, 7}, every label source is 20221120, 20221118 or 20221114,
  all inside the tuning week. No W-2022-47 label ever enters training, and no
  flow being evaluated is ever seen with its label.

CAPACITIES (all supervised, cross-entropy on ground truth):
  matched   batch-normalization affine parameters and running statistics only,
            the same parameter set the label-free method adapts. The
            capacity-matched comparison.
  head      the final classification layer only, with the backbone in
            evaluation mode so batch-normalization statistics stay at their
            W-2022-44 values. This is the common operational shortcut, and it
            isolates labels without statistic recalibration, the mirror image
            of the stats-only condition.
  full      every parameter. This is what an operator would actually do, and
            the comparison is not honest without it.
  combined  a retrained model followed by label-free filtered adaptation on
            the current unlabeled window, at the frozen episodic
            configuration. The only condition here that could become a
            deployment recommendation.

KILL RULE B-K1, pre-registered and checked automatically below:
  If any delayed-label baseline at any delta exceeds the label-free filtered
  recovery by more than 2.0 points, the paper does not present label-free
  adaptation as an operational recommendation, and the framing changes to a
  stopgap confined to the label-latency window, with the crossover delta
  reported. The script prints this verdict rather than leaving it to
  judgement after the fact.

ORDER OF OPERATIONS ENFORCED BY THIS SCRIPT:
  --tune runs inside W-2022-46 only and writes delayed_label_config.json.
  --report refuses to run until that file exists.

Run:
    python scripts/17_delayed_label.py --tune   --size S
    python scripts/17_delayed_label.py --report --size S
    python scripts/17_delayed_label.py --tune   --size S --smoke 5   # plumbing
Resumable via delayed_label_progress.json.
"""
import argparse, copy, datetime as dt, json, os, sys, time
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
AUDIT_JSON = "streaming_order_audit.json"
CONFIG_JSON = "delayed_label_config.json"
CKPT       = "delayed_label_progress.json"

EVAL_DAY      = "20221121"          # the day holding the three report windows
DELTAS        = [1, 3, 7]           # pre-registered, all reported
POOL_BATCHES  = 200                 # labeled pool drawn from the source day
REPORT_WINDOWS = [(0, 200), (200, 400), (400, 600)]
BN_MOM        = 0.1

# Label-free references already in the record, for the comparison table.
REF_FROZEN_W   = [0.72239013671875, 0.72425537109375, 0.73946044921875]
REF_LABELFREE  = {"stats-only (episodic)": 2.43, "filtered (episodic)": 3.06,
                  "filtered (streaming, causal)": 2.88,
                  "labeled oracle (transductive, unattainable)": 11.55}
BK1_COMPARATOR = 3.06               # episodic filtered, the recorded headline
BK1_MARGIN     = 2.0                # points

# Frozen episodic configuration, used by the combined condition only.
TTA_LR, TTA_STEPS, TTA_QUANT = 1e-3, 50, 0.5

# Tuning grids, pre-registered. Baselines are tuned to their own advantage.
# Supervised capacities, and the two LABEL-FREE controls on the same source
# day. Without the controls the supervised numbers cannot be read: a model
# retrained on yesterday also recalibrates its batch-normalization statistics
# on yesterday, and that costs no labels. src-stats and src-tent isolate it.
GRID = {"matched":   {"lr": [1e-4, 1e-3], "steps": [50, 100]},
        "head":      {"lr": [1e-4, 1e-3], "steps": [50, 100]},
        "full":      {"lr": [1e-5, 1e-4], "steps": [50, 100]},
        "src-stats": {"lr": [None],       "steps": [50, 100]},
        "src-tent":  {"lr": [1e-4, 1e-3], "steps": [50, 100]}}
SUPERVISED = ("matched", "head", "full")
LABEL_FREE_DELAYED = ("src-stats", "src-tent")
TUNE_EVAL_DAY = "20221116"          # inside W-2022-46; source is delta=1


def parse_day(s):
    return dt.datetime.strptime(s, "%Y%m%d").date()


def shift_day(day, delta):
    return (parse_day(day) - dt.timedelta(days=delta)).strftime("%Y%m%d")


def build(size, period_name, dates=None):
    import torch
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import mm_cesnet_v2, MM_CESNET_V2_Weights
    weights = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
    model = mm_cesnet_v2(weights=weights, model_dir=MODEL_DIR)
    model.eval()
    tr = weights.transforms
    ds = CESNET_QUIC22(DATA_DIR, size=size)
    kw = dict(dataset=ds, apps_selection=AppSelection.ALL_KNOWN,
              train_period_name=TRAIN_WEEK, test_period_name=period_name,
              batch_size=BATCH, train_workers=0, test_workers=0,
              use_packet_histograms=True,
              ppi_transform=tr.get("ppi_transform"),
              flowstats_transform=tr.get("flowstats_transform"),
              flowstats_phist_transform=tr.get("flowstats_phist_transform"))
    if dates is not None:
        kw["test_dates"] = list(dates)
    kw = {k: v for k, v in kw.items() if v is not None}
    cfg = DatasetConfig(**kw)
    ds.set_dataset_config_and_initialize(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, ds.get_test_dataloader(), device


def fwd(model, batch, device):
    import torch
    parts = list(batch) if isinstance(batch, (tuple, list)) else [batch]
    ppi = fs = y = None
    for p in parts:
        arr = np.asarray(p)
        if arr.ndim == 3:                          ppi = arr
        elif arr.ndim == 2 and arr.shape[1] > 0:   fs = arr
        elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer): y = arr
    if ppi is None or fs is None or y is None:
        raise RuntimeError(f"batch parse failed: {[np.asarray(p).shape for p in parts]}")
    return model((torch.as_tensor(ppi).float().to(device),
                  torch.as_tensor(fs).float().to(device))), y


def collect(loader, skip, n, label=""):
    out = []
    for i, b in enumerate(loader):
        if i < skip: continue
        out.append(b)
        if len(out) >= n: break
    if len(out) < n:
        print(f"  [WARN]{' '+label if label else ''} wanted {n} from offset "
              f"{skip}, got {len(out)}")
    return out


def accuracy_on_batches(model, batches, device):
    import torch
    from sklearn.metrics import accuracy_score
    def _run():
        was = {n: mod.training for n, mod in model.named_modules()}
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for b in batches:
                lo, y = fwd(model, b, device)
                ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
        for n, mod in model.named_modules():
            mod.train(was[n])
        return accuracy_score(np.concatenate(ys), np.concatenate(ps))
    return guarded_eval(model, _run)


def find_head(m):
    """Last Linear module in forward order, i.e. the classifier."""
    import torch.nn as nn
    last = None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            last = (name, mod)
    if last is None:
        raise RuntimeError("no Linear layer found; cannot isolate a head")
    return last


def retrain(base_model, pool, device, capacity, lr, steps, order):
    """Adapt on the source day, then return the model. The model never sees a
    flow from the evaluation day. Supervised capacities use cross-entropy on
    ground truth; src-stats and src-tent use no labels at all and exist to
    show how much of the supervised gain is really recalibration on recent
    data."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []

    if capacity in ("src-stats", "src-tent"):
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mod.train(); mod.momentum = BN_MOM
                if capacity == "src-tent":
                    mod.requires_grad_(True)
                    if mod.weight is not None: params.append(mod.weight)
                    if mod.bias is not None:   params.append(mod.bias)
        if capacity == "src-stats":
            with torch.no_grad():
                for s_ in range(steps):
                    fwd(m, pool[order[s_ % len(order)]], device)
            return m
        opt = torch.optim.Adam(params, lr=lr)
        for s_ in range(steps):
            lo, _ = fwd(m, pool[order[s_ % len(order)]], device)
            ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
            sel = ent <= torch.quantile(ent.detach(), TTA_QUANT)
            loss = ent[sel].mean() if sel.any() else ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return m

    if capacity == "matched":
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mod.requires_grad_(True); mod.train(); mod.momentum = BN_MOM
                if mod.weight is not None: params.append(mod.weight)
                if mod.bias is not None:   params.append(mod.bias)
    elif capacity == "head":
        # Backbone stays in evaluation mode, so batch-normalization statistics
        # remain at their W-2022-44 values. Labels without recalibration.
        name, head = find_head(m)
        head.requires_grad_(True)
        params = [p for p in head.parameters()]
    elif capacity == "full":
        m.train(); m.requires_grad_(True)
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mod.momentum = BN_MOM
        params = [p for p in m.parameters() if p.requires_grad]
    else:
        raise ValueError(capacity)
    opt = torch.optim.Adam(params, lr=lr)
    for s in range(steps):
        b = pool[order[s % len(order)]]
        lo, y = fwd(m, b, device)
        loss = F.cross_entropy(lo, torch.as_tensor(y).long().to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def tta_on_window(model, window, device, order):
    """Frozen-configuration label-free filtered adaptation, for the combined
    condition. Identical to the recorded episodic method."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = BN_MOM
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias is not None:   params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=TTA_LR)
    for s in range(TTA_STEPS):
        lo, _ = fwd(m, window[order[s % len(order)]], device)
        ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), TTA_QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def day_start(day):
    if not os.path.isfile(AUDIT_JSON):
        return None
    with open(AUDIT_JSON) as f: audit = json.load(f)
    for per in audit.values():
        if per.get("capped"): continue
        dm = per.get("day_map_from_indices", {})
        if day in dm and dm[day].get("flows"):
            return int(dm[day]["first_batch_index"])
    return None


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f: return json.load(f)
    return {"done": {}}


def save_ckpt(c):
    with open(CKPT, "w") as f: json.dump(c, f, indent=1)


def eval_windows(m, windows, device):
    return [accuracy_on_batches(m, w, device) for w in windows]


def do_tune(args):
    nb = args.smoke
    smoke = nb is not None
    if smoke:
        print(f"*** SMOKE RUN: {nb} steps and a {nb}-batch pool. "
              f"{CONFIG_JSON} will NOT be written. ***\n")
    src = shift_day(TUNE_EVAL_DAY, 1)
    print(f"=== EXPERIMENT B TUNING, inside W-2022-46 only ===")
    print(f"    evaluate on {TUNE_EVAL_DAY}, first {POOL_BATCHES} batches of "
          f"that day; retrain on {src} (delta = 1)")
    print(f"    {TEST_WEEK} IS NOT TOUCHED BY THIS MODE\n")

    off = day_start(TUNE_EVAL_DAY)
    if off is None:
        sys.exit(f"[STOP] {TUNE_EVAL_DAY} not found in {AUDIT_JSON}. Run the "
                 f"stream-order audit uncapped first.")
    base, vloader, device = build(args.size, "W-2022-46")
    print(f"device={device}")
    pool_n = nb or POOL_BATCHES
    vwin = collect(vloader, off, pool_n, label=f"eval {TUNE_EVAL_DAY}")
    frozen = accuracy_on_batches(base, vwin, device)
    print(f"  {TUNE_EVAL_DAY} frozen accuracy = {frozen:.4f}\n")

    _, sloader, _ = build(args.size, f"DAY-{src}", dates=[src])
    pool = collect(sloader, 0, pool_n, label=f"source {src}")
    print(f"  labeled pool from {src}: {len(pool)} batches "
          f"({len(pool)*2048:,} flows)\n")

    order = list(np.random.default_rng(0).permutation(len(pool)))
    cfg = {}
    for cap, g in GRID.items():
        best = None
        for lr in g["lr"]:
            for st in ([nb] if smoke else g["steps"]):
                m = retrain(base, pool, device, cap, lr, st, order)
                a = accuracy_on_batches(m, vwin, device)
                lrs = "  n/a  " if lr is None else f"{lr:.0e}"
                print(f"  {cap:>9} lr={lrs} steps={st:>3}: {a:.4f} "
                      f"({(a-frozen)*100:+.2f}p)"
                      + ("   [label-free]" if cap in LABEL_FREE_DELAYED else ""))
                if best is None or a > best["accuracy"]:
                    best = {"lr": lr, "steps": st, "accuracy": a}
                del m
        cfg[cap] = best
        lrs = "n/a" if best["lr"] is None else f"{best['lr']:.0e}"
        print(f"  -> {cap}: lr={lrs}, steps={best['steps']} "
              f"({(best['accuracy']-frozen)*100:+.2f}p)\n")

    if smoke:
        print("  [SMOKE] plumbing exercised; nothing written.")
        return
    out = {"per_capacity": cfg, "tune_eval_day": TUNE_EVAL_DAY,
           "tune_source_day": src, "tune_frozen": frozen,
           "pool_batches": POOL_BATCHES,
           "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(CONFIG_JSON, "w") as f: json.dump(out, f, indent=1)
    print(f"==== SELECTED (frozen from here on) ====")
    for cap, b in cfg.items():
        lrs = "n/a" if b["lr"] is None else f"{b['lr']:.0e}"
        print(f"  {cap:>9}: lr={lrs}, steps={b['steps']}")
    print(f"  written to {CONFIG_JSON}. Record it in RESULTS.md, then --report.")


def do_report(args):
    if not os.path.isfile(CONFIG_JSON):
        sys.exit(f"[STOP] {CONFIG_JSON} not found. Run --tune first; the "
                 f"report week is not touched before the baselines are tuned "
                 f"inside W-2022-46 and recorded.")
    with open(CONFIG_JSON) as f: cfg = json.load(f)
    caps = cfg["per_capacity"]
    print(f"=== EXPERIMENT B REPORT on {TEST_WEEK} ===")
    print(f"    evaluation day {EVAL_DAY}, three report windows "
          f"(batches 0 to 599)")
    for cap, b in caps.items():
        lrs = "n/a" if b["lr"] is None else f"{b['lr']:.0e}"
        print(f"    {cap:>9}: lr={lrs}, steps={b['steps']} "
              f"(selected {cfg['selected_at']})")
    print()

    ckpt = load_ckpt()
    if ckpt["done"]:
        print(f"[RESUME] {len(ckpt['done'])} units done\n")

    base, tloader, device = build(args.size, TEST_WEEK)
    print(f"device={device}")
    windows = [collect(tloader, a, b - a, label=f"window {i+1}")
               for i, (a, b) in enumerate(REPORT_WINDOWS)]
    frozen = eval_windows(base, windows, device)
    assert_anchor(frozen[0], REF_FROZEN_W[0], tol=0.0,
                  name="W-47 window 1 frozen (Table I anchor)")
    print(f"  [ANCHOR OK] frozen windows "
          f"{' / '.join(f'{v:.4f}' for v in frozen)}\n")

    for delta in DELTAS:
        src = shift_day(EVAL_DAY, delta)
        if src >= "20221121":
            print(f"  [note] delta={delta}: source {src} is inside the report "
                  f"week; its labels are from the past relative to the "
                  f"evaluated traffic, which is the simulated situation, but "
                  f"state it in the paper.")
        print(f"--- delta = {delta} day(s): labels from {src} ---")
        _, sloader, _ = build(args.size, f"DAY-{src}", dates=[src])
        pool = collect(sloader, 0, POOL_BATCHES, label=f"source {src}")
        print(f"  labeled pool: {len(pool)} batches ({len(pool)*2048:,} flows)")

        for cap in ("src-stats", "src-tent", "matched", "head", "full"):
            K = 1 if cap == "full" else args.K
            for k in range(K):
                key = f"d{delta}_{cap}_{k}"
                order = list(np.random.default_rng(1000 * delta + k)
                             .permutation(len(pool)))
                if key not in ckpt["done"]:
                    t0 = time.time()
                    m = retrain(base, pool, device, cap,
                                caps[cap]["lr"], caps[cap]["steps"], order)
                    accs = eval_windows(m, windows, device)
                    ckpt["done"][key] = {
                        "delta": delta, "capacity": cap, "k": k,
                        "source_day": src, "K_declared": K,
                        "lr": caps[cap]["lr"], "steps": caps[cap]["steps"],
                        "accuracies": accs,
                        "recoveries": [(a - f) * 100 for a, f in zip(accs, frozen)]}
                    save_ckpt(ckpt)
                    print(f"    {cap:>9} k={k}: "
                          + " / ".join(f"{v:+.2f}" for v in
                                       ckpt['done'][key]['recoveries'])
                          + f"  ({(time.time()-t0)/60:.1f}m)"
                          + ("   [label-free]"
                             if cap in LABEL_FREE_DELAYED else ""))
                    if args.combined and k == 0 and cap in SUPERVISED:
                        ckey = f"d{delta}_{cap}+tta_0"
                        if ckey not in ckpt["done"]:
                            caccs = []
                            for wi, w in enumerate(windows):
                                worder = list(np.random.default_rng(1000 * wi)
                                              .permutation(len(w)))
                                mc = tta_on_window(m, w, device, worder)
                                caccs.append(accuracy_on_batches(mc, w, device))
                                del mc
                            ckpt["done"][ckey] = {
                                "delta": delta, "capacity": cap + "+tta", "k": 0,
                                "source_day": src, "K_declared": 1,
                                "accuracies": caccs,
                                "recoveries": [(a - f) * 100
                                               for a, f in zip(caccs, frozen)]}
                            save_ckpt(ckpt)
                            print(f"    {cap+'+tta':>8} k=0: "
                                  + " / ".join(f"{v:+.2f}" for v in
                                               ckpt['done'][ckey]['recoveries']))
                    del m
        del pool
        print()

    # ---------------- results table ----------------
    print("==== EXPERIMENT B RESULT: recovery over the frozen model, points ====")
    print("  evaluation: three report windows of 20221121, identical flows for "
          "every row\n")
    print(f"  {'condition':<34}{'w1':>7}{'w2':>7}{'w3':>7}{'mean':>8}{'K':>4}")
    print("  " + "-" * 67)
    for name, v in REF_LABELFREE.items():
        print(f"  {name:<34}{'':>7}{'':>7}{'':>7}{v:8.2f}{'':>4}   [recorded]")
    print()
    summary = {}
    for delta in DELTAS:
        for cap in ("src-stats", "src-tent", "matched", "head", "full",
                    "matched+tta", "head+tta", "full+tta"):
            keys = [k for k, v in ckpt["done"].items()
                    if v["delta"] == delta and v["capacity"] == cap]
            if not keys: continue
            recs = np.array([ckpt["done"][k]["recoveries"] for k in keys])
            pw = recs.mean(axis=0)
            fam = ("delayed-data, LABEL-FREE" if cap in LABEL_FREE_DELAYED
                   else "delayed-label")
            label = f"{fam} {cap}, delta={delta}d"
            summary[label] = float(pw.mean())
            print(f"  {label:<34}{pw[0]:7.2f}{pw[1]:7.2f}{pw[2]:7.2f}"
                  f"{pw.mean():8.2f}{len(keys):4d}")

    # ---------------- kill rule B-K1 ----------------
    print("\n==== KILL RULE B-K1 ====")
    pure = {k: v for k, v in summary.items()
            if "+tta" not in k and k.startswith("delayed-label")}
    ctrl = {k: v for k, v in summary.items() if "LABEL-FREE" in k}
    if not pure:
        print("  no delayed-label results yet")
        return
    best_name = max(pure, key=pure.get)
    best = pure[best_name]
    print(f"  best delayed-label baseline: {best_name} at {best:+.2f}p")
    print(f"  label-free filtered (episodic, recorded): {BK1_COMPARATOR:+.2f}p")
    print(f"  difference: {best - BK1_COMPARATOR:+.2f}p "
          f"(threshold {BK1_MARGIN:+.2f}p)")
    if best - BK1_COMPARATOR > BK1_MARGIN:
        print("\n  *** B-K1 FIRES ***")
        print("  Per the pre-registration, the paper does NOT present "
              "label-free adaptation as an operational recommendation. The "
              "framing changes, in the abstract and the conclusion, to: "
              "label-free test-time adaptation is a stopgap whose value is "
              "confined to the interval between the onset of drift and the "
              "arrival of labels, and this study measures how small that "
              "value is. Report the crossover delta explicitly.")
        beaten = sorted((d for d in DELTAS
                         if any(summary.get(f"delayed-label {c}, delta={d}d", -99)
                                - BK1_COMPARATOR > BK1_MARGIN
                                for c in ("matched", "head", "full"))))
        print(f"  deltas at which a baseline clears the margin: {beaten}")
    else:
        print("\n  B-K1 does not fire. Label-free adaptation may still be "
              "presented as an operational option, and the delayed-label "
              "comparison is reported as the context that makes that claim "
              "meaningful.")
    if ctrl:
        bc = max(ctrl, key=ctrl.get)
        print(f"\n  LABEL-FREE CONTROL ON THE SAME SOURCE DAY")
        print(f"    best: {bc} at {ctrl[bc]:+.2f}p")
        if pure:
            d = pure[best_name] - ctrl[bc]
            print(f"    what the LABELS buy, over label-free adaptation on the "
                  f"same delayed data: {d:+.2f}p")
            if d < 1.0:
                print(f"    Under one point. The supervised baseline's "
                      f"advantage is then mostly recalibration on recent "
                      f"traffic, which needs no labels, and that is the "
                      f"finding to report rather than the raw supervised "
                      f"number.")
    below = [k for k, v in pure.items() if v <= 0]
    if below:
        print(f"\n  [B-K2] these baselines fail to beat the frozen model and "
              f"are reported as such, not dropped: {below}")
    for delta in DELTAS:
        for cap in ("matched", "head", "full"):
            a = summary.get(f"delayed-label {cap}, delta={delta}d")
            b = summary.get(f"delayed-label {cap}+tta, delta={delta}d")
            if a is not None and b is not None and b <= a:
                print(f"  [B-K4] {cap}+tta at delta={delta}d ({b:+.2f}p) does "
                      f"not beat {cap} alone ({a:+.2f}p); no combined "
                      f"recommendation is made for it.")
    print(f"\n  All deltas reported (B-K3). Raw: {CKPT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--K", type=int, default=3,
                    help="orderings for matched and head; full is always K=1")
    ap.add_argument("--smoke", type=int, default=None,
                    help="tiny tuning run; writes no config")
    ap.add_argument("--no-combined", dest="combined", action="store_false")
    args = ap.parse_args()
    if args.tune == args.report:
        sys.exit("[STOP] pass exactly one of --tune or --report.")
    (do_tune if args.tune else do_report)(args)


if __name__ == "__main__":
    main()
