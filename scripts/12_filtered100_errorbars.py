#!/usr/bin/env python3
r"""
12_filtered100_errorbars.py -- fill Table II: filtered TENT (q=0.5) at
steps=100 with proper error bars.

The manuscript currently reports the filtered 100-step entry at K=1
(+2.98, one fixed ordering per window) while every other Table II cell
has K>=3 spreads. This script measures the same condition at K=3 seeded
orderings x 3 windows so the table is symmetric.

Condition: standard filtered TENT (BN train mode, momentum 0.1, affine
params via Adam, lr=1e-3, quantile=0.5), steps=100, final-step
evaluation. Identical to the headline pipeline except step count.

Sanity anchor: the k=0 units should reproduce the earlier K=1 values
(+3.35 / +2.83 / +2.76) to printed precision.

Run:
    python 12_filtered100_errorbars.py --size S --K 3
"""
import argparse, copy, itertools, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
IN_PERIOD  = 0.867
BATCH      = 256
CKPT       = "filtered100_progress.json"

TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3
DEFAULT_K  = 3

LR          = 1e-3
QUANT       = 0.5
STEPS       = 100
BN_MOMENTUM = 0.1
EXPECT_K0   = [3.35, 2.83, 2.76]   # earlier K=1 measurements


def build_loader(ds, cfg_kwargs, DatasetConfig):
    cfg = DatasetConfig(**cfg_kwargs)
    ds.set_dataset_config_and_initialize(cfg)
    return ds.get_test_dataloader()


def build(size, test_week):
    import torch
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import mm_cesnet_v2, MM_CESNET_V2_Weights
    weights = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
    model = mm_cesnet_v2(weights=weights, model_dir=MODEL_DIR)
    model.eval()
    transforms = weights.transforms
    ds = CESNET_QUIC22(DATA_DIR, size=size)
    cfg_kwargs = dict(
        dataset=ds, apps_selection=AppSelection.ALL_KNOWN,
        train_period_name=TRAIN_WEEK, test_period_name=test_week,
        batch_size=BATCH, train_workers=0, test_workers=0,
        use_packet_histograms=True,
        ppi_transform=transforms.get("ppi_transform"),
        flowstats_transform=transforms.get("flowstats_transform"),
        flowstats_phist_transform=transforms.get("flowstats_phist_transform"),
    )
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if v is not None}
    loader = build_loader(ds, cfg_kwargs, DatasetConfig)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, loader, device


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


def count_available_batches(loader, cap):
    n = 0
    for _ in loader:
        n += 1
        if n >= cap: break
    return n


def collect_window(loader, skip, n, label=""):
    batches = []
    for i, b in enumerate(loader):
        if i < skip: continue
        batches.append(b)
        if len(batches) >= n: break
    if len(batches) < n:
        print(f"  [WARN]{(' '+label) if label else ''} wanted n={n}, got {len(batches)} (skip={skip}).")
    return batches


def accuracy_on_batches(model, batches, device):
    import torch
    from sklearn.metrics import accuracy_score
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in batches:
            lo, y = fwd(model, b, device)
            ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
    return accuracy_score(np.concatenate(ys), np.concatenate(ps))


def tent_final(base_model, window_batches, device, steps, order):
    """Standard filtered TENT, final-step evaluation (verbatim logic of
    scripts/02_errorbars.py)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = BN_MOMENTUM
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    for s in range(steps):
        b = window_batches[order[s % len(order)]]
        lo, _ = fwd(m, b, device)
        ent = -(F.softmax(lo,1) * F.log_softmax(lo,1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    final = accuracy_on_batches(m, window_batches, device)
    return frozen, final


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f: return json.load(f)
    return {"done": {}}


def save_ckpt(d):
    with open(CKPT, "w") as f: json.dump(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    ckpt = load_ckpt()
    if ckpt["done"]:
        print(f"[RESUME] found {len(ckpt['done'])} completed orderings in {CKPT}. "
              f"Delete it to start fresh.\n")

    print(f"=== FILTERED TENT steps=100, K-ordering error bars on {TEST_WEEK} ===")
    tmodel, tloader, device = build(args.size, TEST_WEEK)
    print(f"device={device}")
    probe_cap = TARGET_TEST_EVAL_BATCHES*REPEATS + 50
    n_avail = count_available_batches(tloader, cap=probe_cap)
    test_eval_batches = min(TARGET_TEST_EVAL_BATCHES, n_avail // REPEATS)
    if test_eval_batches < MIN_TEST_EVAL_BATCHES:
        print(f"  [STOP] only {test_eval_batches}/repeat."); sys.exit(1)
    print(f"  n_avail={n_avail}, using {test_eval_batches}/window\n")

    tmodel, tloader, device = build(args.size, TEST_WEEK)
    windows = [collect_window(tloader, skip=r*test_eval_batches, n=test_eval_batches,
                              label=f"window {r+1}") for r in range(REPEATS)]
    tbase = accuracy_on_batches(tmodel, windows[0], device)
    print(f"W-47 self-check (window 1) = {tbase:.4f}")
    if not (0.62 <= tbase <= 0.78):
        print("  [STOP] baseline out of range."); sys.exit(1)

    total = REPEATS*args.K
    print(f"\n=== CONTROL: {args.K} orderings x {REPEATS} windows = {total} passes "
          f"(steps={STEPS}, lr={LR:.0e}, q={QUANT}, full TENT) ===\n")
    t0 = time.time(); count_done = 0
    for r, w in enumerate(windows):
        for k in range(args.K):
            key = f"{r}_{k}"
            if key in ckpt["done"]:
                count_done += 1; continue
            rng = np.random.default_rng(1000*r + k)
            order = list(rng.permutation(len(w)))
            fr, ad = tent_final(tmodel, w, device, STEPS, order)
            ckpt["done"][key] = {"frozen": fr, "recovery": ad-fr}
            save_ckpt(ckpt)
            count_done += 1
            elapsed = time.time()-t0
            print(f"  [{count_done}/{total}] window {r+1} ord {k+1}: "
                  f"frozen={fr:.4f} adapted={ad:.4f} rec={(ad-fr)*100:+.2f}p "
                  f"(elapsed {elapsed/60:.1f}m)")

    # ---- AGGREGATE (same as 02) ----
    per_window = {r: [] for r in range(REPEATS)}
    bases = {}
    for key, v in ckpt["done"].items():
        r = int(key.split("_")[0]); per_window[r].append(v["recovery"]); bases[r]=v["frozen"]
    all_recs = [v["recovery"] for v in ckpt["done"].values()]
    pw_mean = [float(np.mean(per_window[r])) for r in range(REPEATS)]
    pw_std  = [float(np.std(per_window[r]))  for r in range(REPEATS)]
    base_m  = float(np.mean([bases[r] for r in range(REPEATS)]))
    gap     = IN_PERIOD - base_m
    pooled_m, pooled_s = float(np.mean(all_recs)), float(np.std(all_recs))

    print(f"\n==== FILTERED-100 RESULT (S, n={test_eval_batches}/window, K={args.K}) ====")
    print(f"  W-47 frozen acc = {base_m:.4f} (gap {gap:.4f})")
    print(f"  filtered-100 recovery: {pooled_m*100:+.2f}p ± {pooled_s*100:.2f} "
          f"({pooled_m/gap:.1%} of gap)")
    print(f"  per-window means: {', '.join(f'{m*100:+.2f}' for m in pw_mean)}")
    print(f"  mean within-window order-std = {float(np.mean(pw_std))*100:.2f}p")
    print(f"\n  Use: replaces the K=1 (+2.98) cell in Table II; k=0 units")
    print(f"  should match {EXPECT_K0} to printed precision.")
    print(f"  Done. Delete {CKPT} before a fresh run.")


if __name__ == "__main__":
    main()
