#!/usr/bin/env python3
r"""
leakage_clean_tent.py — HONEST TENT evaluation, no selection leakage.

STEP 1 CHANGES (widen W-47 eval) — kept from prior revision:
  - TUNE_EVAL_BATCHES (60) vs TARGET_TEST_EVAL_BATCHES (200) are separate.
  - count_available_batches() probes W-47 capacity (data-only) before
    committing to a widened EVAL_BATCHES, and clips honestly if the split
    can't support 200/repeat x REPEATS non-overlapping slices.

WINDOW-ALIGNMENT FIX (this revision) — the important one:
  Previously, tent()'s adaptation loop always pulled batches from
  `iter(loader)` starting at position 0, regardless of the `skip` used for
  evaluation. So repeat 3 (skip=400) was EVALUATED on batches 400-599 but
  ADAPTED on batches 0-99 every time -- the three "repeats" never actually
  adapted on the window they were scored against.

  Fix: collect_window(loader, skip, n) materializes exactly the batches for
  one window ONCE. tent() now adapts AND evaluates on that same cached
  window, cycling through it in stream order (same cycling behavior as
  before, just now aligned to the correct slice). No shuffling/seeding
  introduced yet -- that's the separate, later step.

Protocol (unchanged):
  1. TUNE on W-2022-46 (validation week): small grid over (lr, steps, quantile).
     Adaptation is label-free; W-46 labels used ONLY to score each config.
  2. FREEZE the config that recovers most on W-46.
  3. REPORT on W-2022-47 (test week): apply the frozen config, adapt label-free
     on each window, evaluate on that SAME window. W-47 never influences any
     tuning choice.

Windows/CPU/16GB friendly. Run:

    python leakage_clean_tent.py            # XS
    python leakage_clean_tent.py --size S   # later, for hardening
"""
import argparse, copy, itertools, sys
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
VAL_WEEK   = "W-2022-46"     # tune here
TEST_WEEK  = "W-2022-47"     # report here (frozen config only)
IN_PERIOD  = 0.867
BATCH      = 256

TUNE_EVAL_BATCHES        = 60    # tuning stage stays cheap
TARGET_TEST_EVAL_BATCHES = 200   # goal for the widened W-47 report
MIN_TEST_EVAL_BATCHES    = 60    # floor -- don't quietly go below what already worked
REPEATS    = 3

# small honest grid (tuned on W-46 only)
GRID_LR       = [1e-3, 5e-3]
GRID_STEPS    = [50, 100]
GRID_QUANTILE = [0.5, 1.0]   # 1.0 = no filtering (vanilla-ish); 0.5 = stabilized


def build_loader(ds, cfg_kwargs, DatasetConfig):
    cfg = DatasetConfig(**cfg_kwargs)
    ds.set_dataset_config_and_initialize(cfg)
    return ds.get_test_dataloader()


def build(size, test_week):
    """Pretrained model + a test dataloader for `test_week`, with the weights'
    own transforms attached (so flowstats arrive as the 43-dim the model wants)."""
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
    """Walk the loader WITHOUT running the model, just to see how many
    batches actually exist (capped so this can't run away on a huge split)."""
    n = 0
    for _ in loader:
        n += 1
        if n >= cap:
            break
    return n


def collect_window(loader, skip, n, label=""):
    """Materialize exactly n batches starting at index `skip`, walking the
    loader once. This window is then reused for BOTH adaptation and
    evaluation, so the two are guaranteed to be on the same slice of data."""
    batches = []
    for i, b in enumerate(loader):
        if i < skip:
            continue
        batches.append(b)
        if len(batches) >= n:
            break
    if len(batches) < n:
        print(f"  [WARN]{(' ' + label) if label else ''} requested n={n} batches "
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


def tent(base_model, window_batches, device, lr, steps, quantile):
    """Label-free BN-stats TENT. Adapts AND evaluates on the SAME
    window_batches (fixes the earlier bug where adaptation always used
    batches from position 0 regardless of which window was being scored)."""
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
    for s in range(steps):
        b = window_batches[s % n]   # cycle through the SAME window used for eval
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
    args = ap.parse_args()

    # ---------- 1. TUNE on W-46 ----------
    print(f"=== TUNING on {VAL_WEEK} (validation) ===")
    vmodel, vloader, device = build(args.size, VAL_WEEK)
    print(f"device={device}")
    val_window = collect_window(vloader, skip=0, n=TUNE_EVAL_BATCHES, label="W-46 tuning window")
    vbase = accuracy_on_batches(vmodel, val_window, device)
    print(f"W-46 frozen acc = {vbase:.4f}  (n={len(val_window)} batches)\n")
    print(f"{'lr':>7} {'steps':>6} {'quant':>6} | {'frozen':>7} {'adapt':>7} {'rec':>7}")
    print("-" * 48)
    tuning = []
    for lr, steps, q in itertools.product(GRID_LR, GRID_STEPS, GRID_QUANTILE):
        fr, ad = tent(vmodel, val_window, device, lr, steps, q)
        rec = ad - fr; tuning.append(((lr, steps, q), rec))
        print(f"{lr:>7.0e} {steps:>6} {q:>6.1f} | {fr:>7.4f} {ad:>7.4f} {rec*100:>+6.2f}p")
    best_cfg, best_rec = max(tuning, key=lambda t: t[1])
    print(f"\nBEST on W-46: lr={best_cfg[0]:.0e} steps={best_cfg[1]} "
          f"quant={best_cfg[2]} (recovered {best_rec*100:+.2f}p on W-46)")
    print("  ^ this config is now FROZEN. W-47 played no part in choosing it.\n")

    # ---------- 2. PROBE W-47 capacity, then REPORT with frozen config ----------
    print(f"=== PROBING {TEST_WEEK} capacity (data-only, no model calls) ===")
    tmodel, tloader, device = build(args.size, TEST_WEEK)
    probe_cap = TARGET_TEST_EVAL_BATCHES * REPEATS + 50
    n_avail = count_available_batches(tloader, cap=probe_cap)
    hit_cap = (n_avail >= probe_cap)
    print(f"  W-47 loader has {'>=' if hit_cap else ''}{n_avail} batches available "
          f"(probed up to cap={probe_cap})")

    test_eval_batches = min(TARGET_TEST_EVAL_BATCHES, n_avail // REPEATS)
    if test_eval_batches < MIN_TEST_EVAL_BATCHES:
        print(f"  [STOP] Only {test_eval_batches} batches/repeat available "
              f"(< floor of {MIN_TEST_EVAL_BATCHES}). Try a larger --size split.")
        sys.exit(1)
    if test_eval_batches < TARGET_TEST_EVAL_BATCHES:
        print(f"  [NOTE] Target was {TARGET_TEST_EVAL_BATCHES}/repeat; W-47 only "
              f"supports {test_eval_batches}/repeat for {REPEATS} non-overlapping "
              f"slices. Using {test_eval_batches}.")
    else:
        print(f"  Using EVAL_BATCHES={test_eval_batches}/repeat (target met).")

    # Rebuild the W-47 loader fresh -- it was consumed by the capacity probe.
    tmodel, tloader, device = build(args.size, TEST_WEEK)

    print(f"\n=== REPORTING on {TEST_WEEK} (test, frozen config) ===")
    windows = []
    for r in range(REPEATS):
        w = collect_window(tloader, skip=r*test_eval_batches, n=test_eval_batches,
                            label=f"repeat {r+1} window")
        windows.append(w)

    tbase_check = accuracy_on_batches(tmodel, windows[0], device)
    print(f"W-47 baseline self-check: frozen acc = {tbase_check:.4f}  "
          f"(n={test_eval_batches}, repeat-1 window)")
    if not (0.62 <= tbase_check <= 0.78):
        print("  [STOP] W-47 baseline out of range; wiring off. Not trusting result.")
        sys.exit(1)

    lr, steps, q = best_cfg
    bases, recs = [], []
    for r in range(REPEATS):
        fr, ad = tent(tmodel, windows[r], device, lr, steps, q)
        bases.append(fr); recs.append(ad - fr)
        print(f"  repeat {r+1} (n={test_eval_batches}, skip={r*test_eval_batches}, "
              f"adapt+eval on SAME window): frozen={fr:.4f} adapted={ad:.4f} "
              f"recovery={(ad-fr)*100:+.2f}p")

    base_m = float(np.mean(bases)); base_s = float(np.std(bases))
    rec_m, rec_s = float(np.mean(recs)), float(np.std(recs))
    gap = IN_PERIOD - base_m; low = rec_m - rec_s
    print(f"\n==== LEAKAGE-CLEAN RESULT (widened + window-aligned, n={test_eval_batches}/repeat) ====")
    print(f"  config tuned on W-46, reported on W-47 (never peeked)")
    print(f"  W-47 frozen acc = {base_m:.4f} ± {base_s:.4f}  (gap {gap:.4f})")
    print(f"  recovery = {rec_m*100:+.2f}p ± {rec_s*100:.2f} ({rec_m/gap:.1%} of gap)")
    print(f"  conservative (mean-1sd) = {low*100:+.2f}p ({low/gap:.1%} of gap)")
    if low < 0.03:
        print("  -> FRAGILE once leakage removed. Reconsider before writing.")
    else:
        print("  -> HOLDS leakage-clean. This is the number for the paper.")
        print("     Next: seed/adaptation-ordering variation for final error bars.")


if __name__ == "__main__":
    main()
