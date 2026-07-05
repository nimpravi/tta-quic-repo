#!/usr/bin/env python3
r"""
04_bnstats_control.py -- BN-STATISTICS-ONLY control (no gradient updates).

WHY THIS EXISTS:
  The headline (+4.34p, quant=0.5) and the mechanism control (+0.69p,
  quant=1.0) BOTH update BatchNorm running statistics during adaptation
  (BN modules in train mode, momentum=0.1) IN ADDITION to entropy-minimizing
  the BN affine parameters. So neither condition isolates what statistic
  recalibration alone contributes. This is the standard TENT-paper ablation
  ("BN-stats" / prediction-time BN).

  This script runs the SAME adaptation loop as tent_final() -- same window,
  same 100-step exposure, same batch order seeding, same BN momentum --
  but with NO optimizer and NO loss. Only running stats move. The affine
  parameters (weight/bias) stay at their pretrained values.

INTERPRETATION GUIDE (decided before seeing the number, so we can't
rationalize afterwards):
  - If BN-stats-only recovers close to +3.06p (the clean headline):
    entropy minimization adds little; the mechanism story must change.
  - If it lands well below +3.06p: filtered entropy-gradient adaptation
    adds a real increment over recalibration alone; report the
    decomposition (stats-only vs stats+filtered-grads vs stats+unfiltered,
    the last from script 03).
  - The prior steps=100 run of this control gave +2.23p and is SUPERSEDED
    (exposure no longer matched to the frozen config).

PROTOCOL:
  No tuning stage -- this is a control MATCHED to the frozen config's
  exposure (steps=100, batch cycling, momentum=0.1), not a tuned method.
  K=3 seeded orderings per window (order affects the running-stat
  trajectory, so bars are still meaningful), same rng scheme as scripts
  02/03: numpy.random.default_rng(1000*window + k).

Run:
    python 04_bnstats_control.py --size S --K 3
    # resumes from bnstats_progress.json if present; delete it to restart
"""
import argparse, copy, itertools, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
IN_PERIOD  = 0.867
BATCH      = 256
CKPT       = "bnstats_progress_steps50.json"

TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3
DEFAULT_K  = 3

# Matched to the frozen config chosen on W-46 (lr/quantile are irrelevant
# here -- there is no loss and no optimizer -- but steps and momentum must
# match so the control sees identical exposure).
STEPS       = 50    # matched to re-tuned frozen config (final-eval, 2026-07-02)
BN_MOMENTUM = 0.1


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


def bnstats_only(base_model, window_batches, device, steps, order):
    """Identical exposure to tent_final(), minus the optimizer: BN modules
    in train mode update running statistics from forward passes; affine
    params never move. Eval ONCE at the end, in eval mode (running stats),
    exactly like tent_final()."""
    import torch, torch.nn as nn
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model); m.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = BN_MOMENTUM
    with torch.no_grad():
        for s in range(steps):
            b = window_batches[order[s % len(order)]]
            fwd(m, b, device)   # forward only; running stats update in train mode
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

    print(f"=== BN-STATS-ONLY CONTROL on {TEST_WEEK} (no tuning stage; matched control) ===")
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
          f"(steps={STEPS}, momentum={BN_MOMENTUM}, NO gradients) ===\n")
    t0 = time.time(); count_done = 0
    for r, w in enumerate(windows):
        for k in range(args.K):
            key = f"{r}_{k}"
            if key in ckpt["done"]:
                count_done += 1; continue
            rng = np.random.default_rng(1000*r + k)
            order = list(rng.permutation(len(w)))
            fr, ad = bnstats_only(tmodel, w, device, STEPS, order)
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

    print(f"\n==== BN-STATS-ONLY RESULT (S, n={test_eval_batches}/window, K={args.K}) ====")
    print(f"  W-47 frozen acc = {base_m:.4f} (gap {gap:.4f})")
    print(f"  BN-STATS-ONLY recovery: {pooled_m*100:+.2f}p ± {pooled_s*100:.2f} "
          f"({pooled_m/gap:.1%} of gap)")
    print(f"  per-window means: {', '.join(f'{m*100:+.2f}' for m in pw_mean)}")
    print(f"  mean within-window order-std = {float(np.mean(pw_std))*100:.2f}p")
    print(f"\n  Compare against: quant=0.5 clean headline (+3.06p ± 0.27, K=5 final-eval)")
    print(f"  and quant=1.0 from script 03's steps=50 rerun.")
    print(f"  Done. Delete {CKPT} before a fresh run.")


if __name__ == "__main__":
    main()
