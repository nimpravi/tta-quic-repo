#!/usr/bin/env python3
r"""
confirm_ordering_effect.py — does natural (temporal) order beat random
shuffles on windows 2 & 3, the way it did on window 1?

MINIMAL BY DESIGN. We already know, from prior runs:
  - window 1: natural +5.62p vs shuffle mean +4.74p ± 0.20  (natural ABOVE band)
  - window 2 shuffle mean = +4.12p ± 0.17   (from the K=8 run)
  - window 3 shuffle mean = +4.15p ± 0.26   (from the K=8 run)
So the ONLY thing missing is the NATURAL-order recovery for windows 2 & 3.
This script runs exactly TWO adaptations (natural order, windows 2 and 3),
final-eval only (FINAL==BEST was proven for this config), checkpointed.

~25 min on CPU. Run:
    python confirm_ordering_effect.py --size S
"""
import copy, os, json
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
N          = 200
LR, STEPS, QUANT = 1e-3, 100, 0.5
CKPT = "ordering_effect_progress.json"

# known shuffle stats from the K=8 run (mean, std) in POINTS
SHUFFLE_STATS = {
    0: (4.74, 0.20),   # window 1 (already have natural = +5.62p; included for context)
    1: (4.12, 0.17),   # window 2
    2: (4.15, 0.26),   # window 3
}
NATURAL_W1 = 5.62      # already measured, for the summary table


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


def tent_natural_final(base, window, device):
    """Natural order (0..n-1), final eval only. Returns (frozen, final)."""
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
        b = window[s % len(window)]
        lo, _ = fwd(m, b, device)
        ent = -(F.softmax(lo,1) * F.log_softmax(lo,1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return frozen, acc(m, window, device)


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f: return json.load(f)
    return {}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    args = ap.parse_args()

    done = load_ckpt()
    if done:
        print(f"[RESUME] have {list(done.keys())} already; skipping those.\n")

    model, loader, device = build(args.size, TEST_WEEK)
    print(f"device={device}, config lr={LR:.0e} steps={STEPS} quant={QUANT}\n")

    # windows 2 and 3 = skip 200 and 400
    targets = {1: 200, 2: 400}   # window-index (0-based) -> skip
    for widx, skip in targets.items():
        key = f"w{widx}"
        if key in done:
            print(f"window {widx+1}: cached natural rec = {done[key]['nat_rec']:+.2f}p")
            continue
        w = collect_window(loader, skip, N)
        fr, fin = tent_natural_final(model, w, device)
        nat = (fin - fr) * 100
        done[key] = {"frozen": fr, "natural_final": fin, "nat_rec": nat}
        with open(CKPT, "w") as f: json.dump(done, f)
        print(f"window {widx+1} (skip={skip}): frozen={fr:.4f} natural_final={fin:.4f} "
              f"natural_rec={nat:+.2f}p  [saved]")

    # ---- verdict table ----
    print("\n==== ORDERING EFFECT: natural vs shuffle, per window ====")
    print(f"{'window':>7} | {'natural':>8} | {'shuffle mean':>13} | {'above band?':>12}")
    print("-"*52)
    rows = []
    # window 1 (already known)
    sm, ss = SHUFFLE_STATS[0]
    above1 = NATURAL_W1 > sm + ss
    rows.append((1, NATURAL_W1, sm, ss, above1))
    print(f"{1:>7} | {NATURAL_W1:>+7.2f}p | {sm:>+6.2f}±{ss:.2f}p | "
          f"{'YES' if above1 else 'no':>12}")
    for widx in (1, 2):
        key = f"w{widx}"
        nat = done[key]["nat_rec"]
        sm, ss = SHUFFLE_STATS[widx]
        above = nat > sm + ss
        rows.append((widx+1, nat, sm, ss, above))
        print(f"{widx+1:>7} | {nat:>+7.2f}p | {sm:>+6.2f}±{ss:.2f}p | "
              f"{'YES' if above else 'no':>12}")

    n_above = sum(1 for r in rows if r[4])
    print("\n================ VERDICT ================")
    if n_above == 3:
        print("Natural order beats the shuffle band in ALL 3 windows.")
        print("=> Ordering effect is GENERAL. Report it as a real secondary")
        print("   finding with 3-window support. Headline stays +4.34p±0.36")
        print("   (shuffle/robustness); ordering effect is the bonus.")
    elif n_above >= 1:
        print(f"Natural beats the band in {n_above}/3 windows.")
        print("=> Effect is REAL but window-dependent. Report honestly as")
        print("   'observed, with window-dependent magnitude' -- softer claim.")
    else:
        print("Natural is WITHIN the shuffle band on windows 2 & 3.")
        print("=> Window 1 was a favorable draw; effect is NOT general.")
        print("   DROP the ordering finding. Headline +4.34p±0.36 alone.")
    print("========================================")
    print(f"\n(delete {CKPT} to rerun from scratch)")


if __name__ == "__main__":
    main()
