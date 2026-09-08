#!/usr/bin/env python3
r"""
15_acrossday_replication.py -- Experiment E of the streaming/delayed-label
pre-registration: across-day replication at the already frozen configuration.

WHY:
  The stream-order audit established that the three windows behind Table I
  fall entirely inside one day, 20221121, and span 11.19, 4.46 and 8.12 hours
  of it. Their spread therefore measures variation across the diurnal cycle of
  a single Monday, not across days or across shifts. This script repeats the
  frozen-configuration conditions on one window per day of W-2022-47 so that
  the reported uncertainty is an across-day quantity.

  NO TUNING HAPPENS HERE. The configuration is the one frozen on W-2022-46
  (lr 1e-3, 50 steps, quantile 0.5) and is not revisited.

WINDOWS:
  200 batches (409,600 flows) anchored at each day's first batch, taken from
  the day map in streaming_order_audit.json rather than hardcoded, and checked
  against the recorded values. 20221126 holds only 182 batches, so its window
  extends 18 batches into 20221127; this is pre-registered, flagged in the
  output, and must be reported.

ANCHOR:
  The 20221121 window is batches 0 to 199, which is exactly Table I window 1.
  Its frozen accuracy is asserted against the recorded 0.72239013671875 at
  zero tolerance. If that fails, the harness does not reproduce the record and
  nothing it produces should be believed.

Run:
    python scripts/15_acrossday_replication.py --size S --K 3
Resumable via acrossday_progress.json. Delete that file to start fresh.
"""
import argparse, copy, json, os, sys, time
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
CKPT       = "acrossday_progress.json"
AUDIT_JSON = "streaming_order_audit.json"

WINDOW_BATCHES = 200
LR         = 1e-3          # frozen on W-2022-46
STEPS      = 50            # frozen on W-2022-46
QUANT      = 0.5           # frozen on W-2022-46
BN_MOM     = 0.1
DEFAULT_K  = 3

# Day-start batches recorded by the audit. Verified against the JSON at run
# time; the literals exist so a silently different audit file is caught.
EXPECTED_DAY_STARTS = {"20221121": 0, "20221122": 603, "20221123": 1233,
                       "20221124": 1856, "20221125": 2418, "20221126": 2823,
                       "20221127": 3004}
EXPECTED_LAST_BATCH = {"20221121": 603, "20221122": 1233, "20221123": 1856,
                       "20221124": 2418, "20221125": 2823, "20221126": 3004,
                       "20221127": 3226}
ANCHOR_W47_W1 = 0.72239013671875     # Table I window 1 frozen accuracy


# --- input path resolution -------------------------------------------------
# Reads search the working directory first, then results/raw and results, so
# these scripts keep working after the raw artifacts are moved into
# results/raw/. Writes are unaffected and still land in the working
# directory, so a rerun never overwrites a committed artifact in place.
_SEARCH = [".", "results/raw", "results"]


def _resolve(name):
    """Return an existing path for `name`, or `name` itself if not found so
    that the caller's own missing-file handling still runs."""
    if os.path.isabs(name) or os.path.isfile(name):
        return name
    for d in _SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return name


def build(size, week):
    import torch
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import mm_cesnet_v2, MM_CESNET_V2_Weights
    weights = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
    model = mm_cesnet_v2(weights=weights, model_dir=MODEL_DIR)
    model.eval()
    transforms = weights.transforms
    ds = CESNET_QUIC22(DATA_DIR, size=size)
    kw = dict(dataset=ds, apps_selection=AppSelection.ALL_KNOWN,
              train_period_name=TRAIN_WEEK, test_period_name=week,
              batch_size=BATCH, train_workers=0, test_workers=0,
              use_packet_histograms=True,
              ppi_transform=transforms.get("ppi_transform"),
              flowstats_transform=transforms.get("flowstats_transform"),
              flowstats_phist_transform=transforms.get("flowstats_phist_transform"))
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


def collect_window(loader, skip, n, label=""):
    batches = []
    for i, b in enumerate(loader):
        if i < skip: continue
        batches.append(b)
        if len(batches) >= n: break
    if len(batches) < n:
        print(f"  [WARN]{' '+label if label else ''} wanted {n} batches from "
              f"offset {skip}, got {len(batches)}")
    return batches


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


def _bn_prepare(base_model, with_grad):
    import torch.nn as nn
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = BN_MOM
            if with_grad:
                mod.requires_grad_(True)
                if mod.weight is not None: params.append(mod.weight)
                if mod.bias is not None:   params.append(mod.bias)
    return m, params


def run_condition(base_model, window, device, cond, order):
    """Returns adapted accuracy. Identical exposure across conditions: the same
    50 batches in the same order, differing only in the gradient signal."""
    import torch, torch.nn.functional as F
    if cond == "frozen":
        return accuracy_on_batches(base_model, window, device)
    m, params = _bn_prepare(base_model, with_grad=(cond == "filtered"))
    if cond == "stats":
        with torch.no_grad():
            for s in range(STEPS):
                fwd(m, window[order[s % len(order)]], device)
    elif cond == "filtered":
        opt = torch.optim.Adam(params, lr=LR)
        for s in range(STEPS):
            lo, _ = fwd(m, window[order[s % len(order)]], device)
            ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
            sel = ent <= torch.quantile(ent.detach(), QUANT)
            loss = ent[sel].mean() if sel.any() else ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    else:
        raise ValueError(cond)
    return accuracy_on_batches(m, window, device)


def load_day_starts():
    if not os.path.isfile(_resolve(AUDIT_JSON)):
        sys.exit(f"[STOP] {AUDIT_JSON} not found. Run "
                 f"scripts/14_stream_order_audit.py first; the day boundaries "
                 f"come from it and are not hardcoded.")
    with open(_resolve(AUDIT_JSON)) as f:
        audit = json.load(f)
    if TEST_WEEK not in audit:
        sys.exit(f"[STOP] {AUDIT_JSON} has no {TEST_WEEK} entry.")
    a = audit[TEST_WEEK]
    if a.get("capped"):
        sys.exit(f"[STOP] {AUDIT_JSON} records a capped run. Rerun the audit "
                 f"uncapped before using its day map.")
    dm = a["day_map_from_indices"]
    starts, lasts = {}, {}
    for d, di in dm.items():
        if not di.get("flows"): continue
        starts[d] = int(di["first_batch_index"])
        lasts[d] = int(di["last_batch_index"])
    if starts != EXPECTED_DAY_STARTS or lasts != EXPECTED_LAST_BATCH:
        sys.exit("[STOP] the audit's day map does not match the values this "
                 "script was written against.\n"
                 f"  audit starts : {starts}\n"
                 f"  expected     : {EXPECTED_DAY_STARTS}\n"
                 "  Something changed. Do not record numbers until this is "
                 "understood.")
    return starts, lasts, int(a["n_batches"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--days", default="", help="comma separated subset, for "
                    "partial runs; all seven by default")
    args = ap.parse_args()

    starts, lasts, n_batches = load_day_starts()
    days = [d.strip() for d in args.days.split(",") if d.strip()] or list(starts)
    for d in days:
        if d not in starts:
            sys.exit(f"[STOP] unknown day {d}")

    ckpt = {"done": {}}
    if os.path.exists(CKPT):
        with open(CKPT) as f: ckpt = json.load(f)
        print(f"[RESUME] {len(ckpt['done'])} units already done in {CKPT}\n")

    print(f"=== EXPERIMENT E: across-day replication on {TEST_WEEK} ===")
    print(f"    frozen config: lr={LR:.0e}, steps={STEPS}, q={QUANT} "
          f"(selected on W-2022-46, not revisited here)")
    print(f"    window = {WINDOW_BATCHES} batches = "
          f"{WINDOW_BATCHES*2048:,} flows; K={args.K}; week = {n_batches:,} batches\n")
    for d in days:
        spill = max(0, starts[d] + WINDOW_BATCHES - (lasts[d] + 1))
        note = f"  [SPILLS {spill} batches into the next day]" if spill else ""
        print(f"  {d}: window = batches {starts[d]} to "
              f"{starts[d]+WINDOW_BATCHES-1}{note}")
    print()

    model, loader, device = build(args.size, TEST_WEEK)
    print(f"device={device}\n")

    total = len(days) * (1 + 2 * args.K)
    done = 0
    t0 = time.time()
    for d in days:
        skip = starts[d]
        window = collect_window(loader, skip, WINDOW_BATCHES, label=d)
        if len(window) < WINDOW_BATCHES:
            print(f"  [STOP] {d}: short window ({len(window)}); the day map "
                  f"and the loader disagree.")
            sys.exit(1)

        key_f = f"{d}_frozen"
        if key_f in ckpt["done"]:
            frozen = ckpt["done"][key_f]["accuracy"]; done += 1
        else:
            frozen = accuracy_on_batches(model, window, device)
            if d == "20221121":
                assert_anchor(frozen, ANCHOR_W47_W1, tol=0.0,
                              name="W-47 window 1 frozen (Table I anchor)")
                print(f"  [ANCHOR OK] 20221121 frozen = {frozen!r} "
                      f"reproduces the Table I record exactly")
            ckpt["done"][key_f] = {"accuracy": frozen}
            with open(CKPT, "w") as f: json.dump(ckpt, f)
            done += 1
        print(f"  {d}: frozen = {frozen:.4f}")

        for cond in ("stats", "filtered"):
            for k in range(args.K):
                key = f"{d}_{cond}_{k}"
                if key in ckpt["done"]:
                    done += 1; continue
                rng = np.random.default_rng(1000 * starts[d] + k)
                order = list(rng.permutation(len(window)))
                acc = run_condition(model, window, device, cond, order)
                ckpt["done"][key] = {"accuracy": acc, "frozen": frozen,
                                     "recovery": acc - frozen}
                with open(CKPT, "w") as f: json.dump(ckpt, f)
                done += 1
                print(f"    [{done}/{total}] {d} {cond} ord {k}: "
                      f"adapted={acc:.4f} rec={(acc-frozen)*100:+.2f}p "
                      f"({(time.time()-t0)/60:.1f}m)")
        del window

    print("\n==== EXPERIMENT E RESULT ====")
    print(f"{'day':>10} {'frozen':>8} {'stats rec':>12} {'filtered rec':>14}  note")
    per_day = {}
    for d in days:
        fr = ckpt["done"].get(f"{d}_frozen", {}).get("accuracy")
        row = {"frozen": fr}
        for cond in ("stats", "filtered"):
            vals = [ckpt["done"][f"{d}_{cond}_{k}"]["recovery"]
                    for k in range(args.K) if f"{d}_{cond}_{k}" in ckpt["done"]]
            row[cond] = (float(np.mean(vals)) * 100, float(np.std(vals)) * 100,
                         len(vals)) if vals else None
        per_day[d] = row
        spill = max(0, starts[d] + WINDOW_BATCHES - (lasts[d] + 1))
        note = f"window spills {spill} batches into the next day" if spill else ""
        s = row["stats"]; fl = row["filtered"]
        print(f"{d:>10} {fr:8.4f} "
              f"{(f'{s[0]:+.2f} ± {s[1]:.2f}' if s else 'n/a'):>12} "
              f"{(f'{fl[0]:+.2f} ± {fl[1]:.2f}' if fl else 'n/a'):>14}  {note}")

    for cond in ("stats", "filtered"):
        means = [per_day[d][cond][0] for d in days if per_day[d][cond]]
        if len(means) < 2: continue
        print(f"\n  {cond}: across-day mean {np.mean(means):+.2f}p, "
              f"across-day std {np.std(means):.2f}p over {len(means)} days")
        print(f"    (the recorded across-window std on 20221121 alone is 0.26p "
              f"for the filtered condition)")
        if min(means) < 0:
            print(f"    [E-K2] at least one day is BELOW the frozen baseline "
                  f"({min(means):+.2f}p). Per the pre-registration this must "
                  f"be stated in the abstract.")
    print(f"\n  All seven days are reported (E-K1). Raw: {CKPT}")


if __name__ == "__main__":
    main()
