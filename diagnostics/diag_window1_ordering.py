#!/usr/bin/env python3
r"""
diagnose_window1.py — why is the earlier single-run window-1 (+5.62p) ABOVE
the max of all 8 shuffled orderings (+5.10p) in the K-run?

Two hypotheses, tested head-to-head on WINDOW 1 ONLY (skip=0, n=200):

  H1 (artifact): the earlier +5.62p came from best-of-checkpoints. The old
     window-aligned script did `best = max(best, eval @ step50, eval @ step100)`.
     The K-run reports FINAL-only. If H1 is right, natural-order best-of-ckpt
     ~= +5.62p while natural-order final ~= +4.7p, and the gap dissolves --
     it was apples vs oranges, nothing to explain.

  H2 (real effect): natural (temporal) arrival order genuinely helps more than
     random shuffles. If H2 is right, natural-order FINAL is still ~+5.6p,
     i.e. above the shuffled FINAL distribution [+4.36,+5.10]. That would be a
     genuine 'temporal order matters' finding, not an artifact.

For window 1 we run, all final-eval AND best-of-checkpoint:
  - natural order (0,1,2,...,199)
  - the 8 shuffle seeds used in the K-run (rng=default_rng(1000*0+k), k=0..7)

Then we print, for each, both the FINAL adapted acc and the BEST-of-checkpoint
adapted acc, so we can see exactly which definition +5.62p came from.

~30 min on CPU (9 adaptations). Reuses the main frozen config.
"""
import copy
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
N          = 200          # window size (matches the K-run)
SKIP       = 0            # window 1
LR, STEPS, QUANT = 1e-3, 100, 0.5   # frozen config
K_SEEDS    = 8


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


def tent_both(base, window, device, order):
    """Return (frozen, final_adapted, best_of_checkpoints_adapted).
    Checkpoints at step 50 and 100 -- exactly what the OLD window-aligned
    script's `best = max(...)` every-50 logic did."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = acc(base, window, device)
    m = copy.deepcopy(base); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = 0.1
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    best = frozen
    for s in range(STEPS):
        b = window[order[s % len(order)]]
        lo, _ = fwd(m, b, device)
        ent = -(F.softmax(lo,1) * F.log_softmax(lo,1)).sum(1)
        if QUANT < 1.0:
            sel = ent <= torch.quantile(ent.detach(), QUANT)
            loss = ent[sel].mean() if sel.any() else ent.mean()
        else:
            loss = ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s+1) % 50 == 0:
            best = max(best, acc(m, window, device))
    final = acc(m, window, device)
    return frozen, final, best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    args = ap.parse_args()

    model, loader, device = build(args.size, TEST_WEEK)
    window = collect_window(loader, SKIP, N)
    print(f"window 1: skip={SKIP} n={len(window)} device={device}")
    print(f"config: lr={LR:.0e} steps={STEPS} quant={QUANT}\n")

    print(f"{'ordering':>16} | {'frozen':>7} {'FINAL':>7} {'rec_f':>7} | "
          f"{'BEST':>7} {'rec_b':>7}")
    print("-"*66)

    # natural order
    nat = list(range(len(window)))
    fr, fin, best = tent_both(model, window, device, nat)
    print(f"{'natural':>16} | {fr:>7.4f} {fin:>7.4f} {(fin-fr)*100:>+6.2f}p | "
          f"{best:>7.4f} {(best-fr)*100:>+6.2f}p")

    # the 8 K-run shuffle seeds (window index r=0)
    fin_recs, best_recs = [], []
    for k in range(K_SEEDS):
        rng = np.random.default_rng(1000*0 + k)
        order = list(rng.permutation(len(window)))
        fr, fin, best = tent_both(model, window, device, order)
        fin_recs.append(fin-fr); best_recs.append(best-fr)
        print(f"{'shuffle k='+str(k):>16} | {fr:>7.4f} {fin:>7.4f} {(fin-fr)*100:>+6.2f}p | "
              f"{best:>7.4f} {(best-fr)*100:>+6.2f}p")

    print("-"*66)
    fr_ = np.array(fin_recs); br_ = np.array(best_recs)
    print(f"\nSHUFFLE FINAL recovery: mean {fr_.mean()*100:+.2f}p ± {fr_.std()*100:.2f} "
          f"[min {fr_.min()*100:+.2f}, max {fr_.max()*100:+.2f}]")
    print(f"SHUFFLE BEST  recovery: mean {br_.mean()*100:+.2f}p ± {br_.std()*100:.2f} "
          f"[min {br_.min()*100:+.2f}, max {br_.max()*100:+.2f}]")

    print("\n================ VERDICT ================")
    print("Compare the earlier single-run +5.62p against the columns above:")
    print("  - If natural-order BEST ~= +5.6p AND natural-order FINAL ~= +4.7p")
    print("    -> H1 CONFIRMED: +5.62p was a best-of-checkpoint number, not")
    print("       comparable to the K-run's final-only +4.34p. Discrepancy is")
    print("       an artifact of eval definition. Write with +4.34p, clean.")
    print("  - If natural-order FINAL is still ~+5.6p (above shuffle max)")
    print("    -> H2: temporal order genuinely helps. Report natural-order as")
    print("       headline with shuffle error bars; note the ordering effect.")
    print("========================================")


if __name__ == "__main__":
    main()
