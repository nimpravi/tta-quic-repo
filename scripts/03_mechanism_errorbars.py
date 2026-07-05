#!/usr/bin/env python3
r"""
mechanism_errorbars.py — error bars on the MECHANISM contrast.

The claim: entropy filtering is load-bearing. quant=0.5 (stabilized) recovers;
quant=1.0 (vanilla, no filtering) fails. Currently quant=1.0's failure rests
on single deterministic grid runs. This puts error bars on the quant=1.0 side,
using the SAME seeded adaptation-order variation as the headline run, on the
SAME 3 windows, so the two conditions are directly comparable.

quant=0.5 reference comes from the CLEAN final-eval K=5 rerun (2026-07-02,
frozen config re-tuned to lr=1e-3, steps=50, quant=0.5):
    +3.06p ± 0.27 pooled  (per-window +3.42/+2.95/+2.81)
The earlier +4.34p (K=8, steps=100) is SUPERSEDED: it was produced by a
best-over-checkpoints evaluation (test-label leakage) in a pre-repo script
version. Final-eval only here; recovery MAY be negative and that is
reportable.
This script measures quant=1.0 the same way. Everything else identical to the
headline protocol: frozen config's lr/steps, window-aligned adapt+eval,
final-eval only (FINAL==BEST proven), K seeded orderings per window,
checkpointed per ordering.

Run:
    python mechanism_errorbars.py --size S --K 8
    python mechanism_errorbars.py --size S --K 5   # faster
"""
import argparse, copy, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
IN_PERIOD  = 0.867
BATCH      = 256
CKPT       = "mechanism_progress_steps50.json"

TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3
DEFAULT_K  = 8

LR, STEPS = 1e-3, 50           # frozen config (re-tuned under final-eval, 2026-07-02)
QUANT     = 1.0                # THE condition under test (no filtering)

# quant=0.5 reference (from the headline K=8 run), points:
Q05_POOLED   = (3.06, 0.27)
Q05_PERWIN   = [3.42, 2.95, 2.81]


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


def collect_window(loader, skip, n):
    batches = []
    for i, b in enumerate(loader):
        if i < skip: continue
        batches.append(b)
        if len(batches) >= n: break
    return batches


def acc(model, batches, device):
    import torch
    from sklearn.metrics import accuracy_score
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in batches:
            lo, y = fwd(model, b, device)
            ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
    return accuracy_score(np.concatenate(ys), np.concatenate(ps))


def tent_q10_final(base, window, device, order):
    """quant=1.0 (no filtering): loss = mean entropy over the whole batch.
    Natural/shuffled order per `order`, final-eval only."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = acc(base, window, device)
    m = copy.deepcopy(base); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = 0.1
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    for s in range(STEPS):
        b = window[order[s % len(order)]]
        lo, _ = fwd(m, b, device)
        ent = -(F.softmax(lo,1) * F.log_softmax(lo,1)).sum(1)
        loss = ent.mean()                      # QUANT=1.0 -> no selection
        opt.zero_grad(); loss.backward(); opt.step()
    return frozen, acc(m, window, device)


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f: return json.load(f)
    return {"done": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    ckpt = load_ckpt()
    if ckpt["done"]:
        print(f"[RESUME] {len(ckpt['done'])} orderings done. Delete {CKPT} to restart.\n")

    print(f"=== MECHANISM error bars: quant={QUANT} (no filtering), "
          f"lr={LR:.0e} steps={STEPS} ===")
    _, tloader, device = build(args.size, TEST_WEEK)
    probe_cap = TARGET_TEST_EVAL_BATCHES*REPEATS + 50
    n_avail = count_available_batches(tloader, cap=probe_cap)
    neb = min(TARGET_TEST_EVAL_BATCHES, n_avail // REPEATS)
    if neb < MIN_TEST_EVAL_BATCHES:
        print(f"  [STOP] only {neb}/window."); sys.exit(1)
    print(f"  device={device}, n_avail={n_avail}, using {neb}/window\n")

    tmodel, tloader, device = build(args.size, TEST_WEEK)
    windows = [collect_window(tloader, skip=r*neb, n=neb) for r in range(REPEATS)]

    total = REPEATS*args.K; count = 0; t0 = time.time()
    for r, w in enumerate(windows):
        for k in range(args.K):
            key = f"{r}_{k}"
            if key in ckpt["done"]:
                count += 1; continue
            rng = np.random.default_rng(1000*r + k)     # SAME seeds as headline run
            order = list(rng.permutation(len(w)))
            fr, ad = tent_q10_final(tmodel, w, device, order)
            ckpt["done"][key] = {"frozen": fr, "recovery": ad-fr}
            with open(CKPT, "w") as f: json.dump(ckpt, f)
            count += 1
            print(f"  [{count}/{total}] window {r+1} ord {k+1}: "
                  f"rec={(ad-fr)*100:+.2f}p  (elapsed {(time.time()-t0)/60:.1f}m)")

    # aggregate
    per = {r: [] for r in range(REPEATS)}; bases = {}
    for key, v in ckpt["done"].items():
        r = int(key.split("_")[0]); per[r].append(v["recovery"]); bases[r]=v["frozen"]
    allr = [v["recovery"] for v in ckpt["done"].values()]
    pw = [float(np.mean(per[r])) for r in range(REPEATS)]
    pws = [float(np.std(per[r])) for r in range(REPEATS)]
    base_m = float(np.mean([bases[r] for r in range(REPEATS)])); gap = IN_PERIOD-base_m
    pm, ps = float(np.mean(allr)), float(np.std(allr))

    print(f"\n==== MECHANISM CONTRAST (S, n={neb}/window, K={args.K}) ====")
    print(f"  quant=1.0 (NO filtering):  {pm*100:+.2f}p ± {ps*100:.2f}  "
          f"({pm/gap:.1%} of gap)")
    print(f"      per-window: {', '.join(f'{m*100:+.2f}' for m in pw)}  "
          f"(order-std {np.mean(pws)*100:.2f}p)")
    print(f"  quant=0.5 (filtering)   :  {Q05_POOLED[0]:+.2f}p ± {Q05_POOLED[1]:.2f}  "
          f"[K=5 final-eval rerun]")
    print(f"      per-window: {', '.join(f'{m:+.2f}' for m in Q05_PERWIN)}")
    lift = Q05_POOLED[0] - pm*100
    print(f"\n  FILTERING LIFT = {lift:+.2f}p  (0.5 minus 1.0)")
    print("\n================ VERDICT ================")
    if pm*100 < 1.0 and (Q05_POOLED[0] - ps*100) > (pm*100 + ps*100):
        print("quant=1.0 recovers <1p and its band is clearly below quant=0.5's.")
        print("=> MECHANISM CLAIM HOLDS with error bars: filtering is load-bearing.")
        print("   Report both conditions with bars; the contrast is clean.")
    elif pm*100 < Q05_POOLED[0] - Q05_POOLED[1]:
        print("quant=1.0 mean is below quant=0.5's lower bound, but not near-zero.")
        print("=> Filtering HELPS materially; phrase as 'substantially larger")
        print("   recovery with filtering' rather than 'fails without'.")
    else:
        print("quant=1.0 band OVERLAPS quant=0.5. The contrast is NOT clean.")
        print("=> Do NOT claim filtering is essential. Re-examine before writing.")
    print("========================================")


if __name__ == "__main__":
    main()
