#!/usr/bin/env python3
r"""
leakage_clean_tent_errorbars.py — STEP 2: honest error bars.

Builds on leakage_clean_tent.py (widened eval + window-aligned adapt/eval).
The pipeline was proven fully deterministic (loader bit-identical across
builds; adaptation bit-identical run-to-run). So "seeds" cannot mean
re-running an identical process -- that would fabricate ±0.00 precision.

The ONE genuine source of stochasticity in this label-free, fixed-weights
setup is the ORDER in which adaptation consumes batches from the window.
It perturbs (a) which batches pass the entropy-quantile filter each step and
(b) Adam's update trajectory. Evaluation is order-independent (full window),
so only adaptation order is varied.

DESIGN (recommended):
  - HEADLINE: Option A. For each W-47 window, run K seeded random
    adaptation-orderings; report mean recovery ± std over orderings.
  - SECONDARY: Option B. Report how the per-window mean moves across the
    3 disjoint windows (this is the ±spread you already had).

Tuning on W-46 is unchanged and still picks the frozen config with NO
ordering shuffle (deterministic), so the frozen-config choice is stable and
leakage-clean exactly as before. Shuffling is applied ONLY at W-47 report
time, to generate the error bars.

Run:
    python leakage_clean_tent_errorbars.py --size S
    python leakage_clean_tent_errorbars.py --size S --K 5   # faster
"""
import argparse, copy, itertools, sys
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
VAL_WEEK   = "W-2022-46"
TEST_WEEK  = "W-2022-47"
IN_PERIOD  = 0.867
BATCH      = 256

TUNE_EVAL_BATCHES        = 60
TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3            # number of disjoint W-47 windows (Option B factor)
DEFAULT_K  = 8            # orderings per window (Option A factor)

GRID_LR       = [1e-3, 5e-3]
GRID_STEPS    = [50, 100]
GRID_QUANTILE = [0.5, 1.0]


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
        if n >= cap:
            break
    return n


def collect_window(loader, skip, n, label=""):
    batches = []
    for i, b in enumerate(loader):
        if i < skip:
            continue
        batches.append(b)
        if len(batches) >= n:
            break
    if len(batches) < n:
        print(f"  [WARN]{(' ' + label) if label else ''} requested n={n} "
              f"but only collected {len(batches)} (skip={skip}).")
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


def tent(base_model, window_batches, device, lr, steps, quantile, order=None):
    """Label-free BN-stats TENT on a fixed window. Evaluation is always on the
    full window (order-independent). Adaptation consumes batches in `order`
    (a list of indices into window_batches); if None, natural order (used for
    deterministic tuning). Returns (frozen_acc, adapted_best_acc)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = 0.1
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=lr)
    best = frozen
    n = len(window_batches)
    if order is None:
        order = list(range(n))
    for s in range(steps):
        b = window_batches[order[s % len(order)]]
        lo, _ = fwd(m, b, device)
        ent = -(F.softmax(lo,1) * F.log_softmax(lo,1)).sum(1)
        if quantile < 1.0:
            sel = ent <= torch.quantile(ent.detach(), quantile)
            loss = ent[sel].mean() if sel.any() else ent.mean()
        else:
            loss = ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s+1) % 50 == 0:
            best = max(best, accuracy_on_batches(m, window_batches, device))
    return frozen, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    ap.add_argument("--K", type=int, default=DEFAULT_K,
                    help="random adaptation-orderings per window")
    args = ap.parse_args()

    # ---------- 1. TUNE on W-46 (deterministic, natural order) ----------
    print(f"=== TUNING on {VAL_WEEK} (validation, deterministic) ===")
    vmodel, vloader, device = build(args.size, VAL_WEEK)
    print(f"device={device}")
    val_window = collect_window(vloader, skip=0, n=TUNE_EVAL_BATCHES)
    vbase = accuracy_on_batches(vmodel, val_window, device)
    print(f"W-46 frozen acc = {vbase:.4f}  (n={len(val_window)})\n")
    print(f"{'lr':>7} {'steps':>6} {'quant':>6} | {'frozen':>7} {'adapt':>7} {'rec':>7}")
    print("-" * 48)
    tuning = []
    for lr, steps, q in itertools.product(GRID_LR, GRID_STEPS, GRID_QUANTILE):
        fr, ad = tent(vmodel, val_window, device, lr, steps, q)   # order=None
        rec = ad - fr; tuning.append(((lr, steps, q), rec))
        print(f"{lr:>7.0e} {steps:>6} {q:>6.1f} | {fr:>7.4f} {ad:>7.4f} {rec*100:>+6.2f}p")
    best_cfg, best_rec = max(tuning, key=lambda t: t[1])
    print(f"\nBEST on W-46: lr={best_cfg[0]:.0e} steps={best_cfg[1]} "
          f"quant={best_cfg[2]} (recovered {best_rec*100:+.2f}p on W-46)")
    print("  ^ FROZEN. W-47 played no part. Chosen with NO shuffle (deterministic).\n")

    # ---------- 2. PROBE W-47 capacity ----------
    print(f"=== PROBING {TEST_WEEK} capacity ===")
    _, tloader, device = build(args.size, TEST_WEEK)
    probe_cap = TARGET_TEST_EVAL_BATCHES * REPEATS + 50
    n_avail = count_available_batches(tloader, cap=probe_cap)
    hit_cap = (n_avail >= probe_cap)
    print(f"  W-47 has {'>=' if hit_cap else ''}{n_avail} batches (cap={probe_cap})")
    test_eval_batches = min(TARGET_TEST_EVAL_BATCHES, n_avail // REPEATS)
    if test_eval_batches < MIN_TEST_EVAL_BATCHES:
        print(f"  [STOP] only {test_eval_batches}/repeat (< {MIN_TEST_EVAL_BATCHES}).")
        sys.exit(1)
    print(f"  Using EVAL_BATCHES={test_eval_batches}/repeat"
          f"{' (target met)' if test_eval_batches==TARGET_TEST_EVAL_BATCHES else ' (clipped)'}")

    # rebuild fresh (probe consumed the loader), collect the 3 windows
    tmodel, tloader, device = build(args.size, TEST_WEEK)
    windows = [collect_window(tloader, skip=r*test_eval_batches, n=test_eval_batches,
                              label=f"window {r+1}") for r in range(REPEATS)]

    tbase_check = accuracy_on_batches(tmodel, windows[0], device)
    print(f"W-47 baseline self-check (window 1) = {tbase_check:.4f}")
    if not (0.62 <= tbase_check <= 0.78):
        print("  [STOP] W-47 baseline out of range; wiring off.")
        sys.exit(1)

    # ---------- 3. REPORT: K orderings per window ----------
    lr, steps, q = best_cfg
    print(f"\n=== REPORTING on {TEST_WEEK}: K={args.K} adaptation-orderings x "
          f"{REPEATS} windows ===")
    print(f"  (frozen config lr={lr:.0e} steps={steps} quant={q}; "
          f"eval on full window, only adaptation order varies)\n")

    per_window_mean = []      # Option B: how the mean moves across windows
    per_window_std  = []
    all_recs        = []      # pooled, for an overall headline number
    window_bases    = []

    for r, w in enumerate(windows):
        recs_this_window = []
        frozen_this = None
        for k in range(args.K):
            rng = np.random.default_rng(1000*r + k)   # reproducible per (window,k)
            order = list(rng.permutation(len(w)))
            fr, ad = tent(tmodel, w, device, lr, steps, q, order=order)
            frozen_this = fr   # frozen acc is order-independent; same every k
            recs_this_window.append(ad - fr)
            all_recs.append(ad - fr)
        rm, rs = float(np.mean(recs_this_window)), float(np.std(recs_this_window))
        per_window_mean.append(rm); per_window_std.append(rs)
        window_bases.append(frozen_this)
        print(f"  window {r+1} (skip={r*test_eval_batches}, base={frozen_this:.4f}): "
              f"recovery {rm*100:+.2f}p ± {rs*100:.2f}  over K={args.K} orderings "
              f"[min {min(recs_this_window)*100:+.2f}, max {max(recs_this_window)*100:+.2f}]")

    base_m  = float(np.mean(window_bases))
    gap     = IN_PERIOD - base_m
    # HEADLINE (Option A): pool all K x REPEATS orderings
    pooled_m = float(np.mean(all_recs)); pooled_s = float(np.std(all_recs))
    # SECONDARY (Option B): spread of per-window means
    across_window_s = float(np.std(per_window_mean))
    order_s_mean    = float(np.mean(per_window_std))  # avg within-window order std

    print(f"\n==== ERROR-BAR RESULT (S split, n={test_eval_batches}/window) ====")
    print(f"  W-47 frozen acc = {base_m:.4f}  (gap {gap:.4f})")
    print(f"  --- HEADLINE (Option A: adaptation-order variation) ---")
    print(f"  recovery = {pooled_m*100:+.2f}p ± {pooled_s*100:.2f} "
          f"(pooled over {args.K}x{REPEATS}={args.K*REPEATS} orderings)")
    print(f"  = {pooled_m/gap:.1%} of gap; conservative (mean-1sd) "
          f"{(pooled_m-pooled_s)*100:+.2f}p ({(pooled_m-pooled_s)/gap:.1%})")
    print(f"  mean within-window order-std = {order_s_mean*100:.2f}p "
          f"(sensitivity to arrival order)")
    print(f"  --- SECONDARY (Option B: across-window heterogeneity) ---")
    print(f"  per-window means: "
          f"{', '.join(f'{m*100:+.2f}' for m in per_window_mean)}  "
          f"(std across windows = {across_window_s*100:.2f}p)")
    print(f"\n  Interpretation guide:")
    print(f"    - If order-std is small (<0.3p): recovery is robust to arrival")
    print(f"      order (a GOOD result; report it plainly, don't inflate).")
    print(f"    - If across-window > order-std: within-week heterogeneity")
    print(f"      dominates; the 3-window spread is the honest uncertainty.")


if __name__ == "__main__":
    main()
