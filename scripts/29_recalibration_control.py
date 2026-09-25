#!/usr/bin/env python3
r"""
29_recalibration_control.py -- is the recovery drift repair or source repair?

THE QUESTION. Re-estimating BN statistics on a day of unlabeled pre-drift
traffic recovers +6.01 points on the report week, while retraining on pre-drift
LABELS at full capacity recovers only +2.88. Unlabeled recalibration beating
labeled retraining on the same period is strange. One explanation is that the
released W-2022-44 checkpoint carries running statistics that do not match the
distribution the network was trained on. If so, part of what this study calls
drift recovery is repair of the source model, and every recovery figure in the
paper, including the 22.65-point gap, needs re-baselining.

THE TEST. Recalibrate on one pre-drift day, then evaluate on data that did NOT
drift. If the recalibrated model also improves there, the gain is not about
drift.

  source day for recalibration : 20221107 (pre-shift W-2022-45)
  evaluated on                 : 20221108 (pre-shift, different day, no leakage)
                                 20221031 (training week W-2022-44)
                                 W-2022-47 window 1 (the paper's anchor)

PRE-COMMITTED PREDICTIONS, recorded before the run:
  R-A  On the two pre-drift evaluation sets, recalibration changes accuracy by
       less than 0.5 points in absolute value. The checkpoint is well
       calibrated, the +6.01 is drift specific, and every published recovery
       number stands.
  R-B  On either pre-drift set, recalibration gains more than 1.0 point. The
       checkpoint's running statistics are stale. The paper must re-baseline:
       the honest gap is measured against a recalibrated source model, not the
       released one, and the headline 22.65 and 3.06 both change.
  R-C  The W-2022-47 anchor reproduces the recorded +6.01 for src-tent within
       0.3 points, confirming this script adapts the same way script 17 does.

  Between 0.5 and 1.0 points is the awkward middle. Decided now: report it as
  a partial calibration effect, state the size, and re-baseline nothing, but
  the limitation section must say the gap is measured against the released
  checkpoint rather than against a best-calibrated source model.

  If R-C fails, stop. Nothing else in this file is interpretable.

Run:
    python scripts/29_recalibration_control.py --size S --K 3
Output: recalibration_control.json
"""
import argparse, copy, json, os, sys, time
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
BATCH      = 256
POOL_BATCHES = 200
EVAL_BATCHES = 200

LR, STEPS, QUANT, BN_MOM = 1e-3, 50, 0.5, 0.1     # frozen config, not retuned
SOURCE_DAY = "20221107"
EVAL_SETS  = [("20221108", "pre-drift, different day"),
              ("20221031", "training week"),
              ("20221121", "report week, anchor")]
CONDS = ["src-stats", "src-tent"]
RECORDED_W47_SRCTENT = 6.01
ANCHOR_W47_W1 = 0.72239013671875

OUT_JSON = "recalibration_control.json"
_SEARCH = [".", "results/raw", "results"]


def _resolve(name):
    if os.path.isabs(name) or os.path.isfile(name):
        return name
    for d in _SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return name


def build(size, period_name, dates=None):
    import torch
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import mm_cesnet_v2, MM_CESNET_V2_Weights
    w = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
    model = mm_cesnet_v2(weights=w, model_dir=MODEL_DIR)
    model.eval()
    tr = w.transforms
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
        a = np.asarray(p)
        if a.ndim == 3:                                        ppi = a
        elif a.ndim == 2 and a.shape[1] > 0:                   fs = a
        elif a.ndim == 1 and np.issubdtype(a.dtype, np.integer): y = a
    if ppi is None or fs is None or y is None:
        raise RuntimeError("batch parse failed")
    return model((torch.as_tensor(ppi).float().to(device),
                  torch.as_tensor(fs).float().to(device))), y


def collect(loader, n, label=""):
    out = []
    for i, b in enumerate(loader):
        if len(out) >= n:
            break
        out.append(b)
    if len(out) < n:
        print(f"  [note] {label}: {len(out)} batches, short of {n}")
    return out


def accuracy(model, batches, device):
    import torch
    def _run():
        was = {n: m.training for n, m in model.named_modules()}
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for b in batches:
                lo, y = fwd(model, b, device)
                ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
        for n, m in model.named_modules():
            m.train(was[n])
        y, p = np.concatenate(ys), np.concatenate(ps)
        return float((y == p).mean())
    return guarded_eval(model, _run)


def recalibrate(base, pool, device, cond, order):
    """Identical semantics to script 17's src-stats and src-tent."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(base); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = BN_MOM
            if cond == "src-tent":
                mod.requires_grad_(True)
                if mod.weight is not None: params.append(mod.weight)
                if mod.bias is not None:   params.append(mod.bias)
    if cond == "src-stats":
        with torch.no_grad():
            for s in range(STEPS):
                fwd(m, pool[order[s % len(order)]], device)
        return m
    opt = torch.optim.Adam(params, lr=LR)
    for s in range(STEPS):
        lo, _ = fwd(m, pool[order[s % len(order)]], device)
        ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--K", type=int, default=3)
    a = ap.parse_args()

    res = json.load(open(_resolve(OUT_JSON))) if os.path.isfile(_resolve(OUT_JSON)) else {}
    print("=== RECALIBRATION CONTROL (declared post-hoc) ===")
    print(f"    recalibrate on {SOURCE_DAY}, evaluate on data that did not drift")
    print(f"    config lr={LR:.0e}, steps={STEPS}, q={QUANT}, m={BN_MOM}. Not retuned.")
    print("    predictions R-A to R-C are in this file's docstring\n")

    # Build the recalibrated models once from the source pool.
    model, loader, device = build(a.size, f"DAY-{SOURCE_DAY}", dates=[SOURCE_DAY])
    pool = collect(loader, POOL_BATCHES, label=f"source {SOURCE_DAY}")
    models = {"frozen": model}
    for cond in CONDS:
        for k in range(a.K):
            rng = np.random.default_rng(1000 + k)
            order = list(rng.permutation(len(pool)))
            models[f"{cond}_{k}"] = recalibrate(model, pool, device, cond, order)
            print(f"  built {cond} k={k}")
    del pool, loader

    t0 = time.time()
    for day, note in EVAL_SETS:
        period = f"DAY-{day}"
        _, eload, _ = build(a.size, period, dates=[day])
        ev = collect(eload, EVAL_BATCHES, label=f"eval {day}")
        base = accuracy(model, ev, device)
        res.setdefault(day, {})["frozen"] = base
        res[day]["note"] = note
        if day == "20221121":
            assert_anchor(base, ANCHOR_W47_W1, tol=0.0,
                          name="W-47 window 1 frozen (Table I anchor)")
            print(f"  [ANCHOR OK] {day} frozen = {base!r}")
        print(f"  {day} ({note}): frozen {base:.4f}")
        for cond in CONDS:
            vals = []
            for k in range(a.K):
                acc = accuracy(models[f"{cond}_{k}"], ev, device)
                vals.append(acc)
                print(f"      {cond} k={k}: {acc:.4f}  "
                      f"({(acc-base)*100:+.2f}p)  ({(time.time()-t0)/60:.1f}m)")
            res[day][cond] = vals
            json.dump(res, open(OUT_JSON, "w"), indent=1)
        del ev, eload

    print("\n==== RESULT: change from recalibrating on a pre-drift day ====")
    print(f"  {'evaluated on':>28} {'src-stats':>11} {'src-tent':>11}")
    for day, note in EVAL_SETS:
        r = res.get(day, {})
        if "frozen" not in r:
            continue
        cells = []
        for cond in CONDS:
            v = r.get(cond)
            cells.append(f"{(np.mean(v)-r['frozen'])*100:+11.2f}" if v else f"{'n/a':>11}")
        print(f"  {day+' '+note:>28} " + " ".join(cells))

    print("\n==== PRE-COMMITTED PREDICTIONS ====")
    anchor = res.get("20221121", {})
    if anchor.get("src-tent"):
        got = (np.mean(anchor["src-tent"]) - anchor["frozen"]) * 100
        ok = abs(got - RECORDED_W47_SRCTENT) <= 0.3
        print(f"  R-C anchor: src-tent on W-2022-47 {got:+.2f}p against the "
              f"recorded {RECORDED_W47_SRCTENT:+.2f}p  "
              f"[{'HOLDS' if ok else 'FAILS, STOP'}]")
        if not ok:
            print("      This script does not reproduce script 17. Nothing below")
            print("      is interpretable until that is resolved.")
            sys.exit(1)
    worst = 0.0
    for day, note in EVAL_SETS:
        if day == "20221121":
            continue
        r = res.get(day, {})
        for cond in CONDS:
            if r.get(cond):
                worst = max(worst, abs(np.mean(r[cond]) - r["frozen"]) * 100)
    if worst:
        if worst < 0.5:
            print(f"  R-A holds: largest pre-drift change {worst:.2f}p, under 0.5p.")
            print("      The released checkpoint is well calibrated. The +6.01 is")
            print("      drift specific and no published number re-baselines.")
        elif worst > 1.0:
            print(f"  R-B holds: largest pre-drift change {worst:.2f}p, over 1.0p.")
            print("      The checkpoint's running statistics are stale. Re-baseline")
            print("      the gap and every recovery figure against a recalibrated")
            print("      source model before the manuscript quotes them again.")
        else:
            print(f"  Middle case: largest pre-drift change {worst:.2f}p.")
            print("      Report the size, re-baseline nothing, and state in the")
            print("      limitations that the gap is measured against the released")
            print("      checkpoint rather than a best-calibrated source model.")
    print(f"\n  raw: {OUT_JSON}")


if __name__ == "__main__":
    main()
