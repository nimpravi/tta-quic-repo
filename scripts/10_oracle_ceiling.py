#!/usr/bin/env python3
r"""
10_oracle_ceiling.py -- labeled-oracle ceiling.

PURPOSE:
  Contextualize the label-free headline (+3.06p, 13.5% of gap) against
  what SUPERVISED adaptation would achieve on the identical windows.
  Two ceilings:

  (a) MATCHED-CAPACITY oracle (default): fine-tune ONLY the BN affine
      parameters with labeled cross-entropy, same steps (50), same lr
      (1e-3), same batches, same seeded orderings as the headline.
      This answers: of the recovery available to THIS adaptation
      capacity, what fraction does the label-free method capture?
      K=3 x 3 windows, ~1.5 h CPU.

  (b) FULL fine-tune (--full): all parameters trainable, labeled CE,
      same steps/lr/orderings, K=1. Loose upper ceiling. ~2 h CPU
      (backward through the whole network is several times costlier).

  Both are TRANSDUCTIVE oracles (train and evaluate on the same
  window, with labels); they upper-bound what any test-time method of
  the corresponding capacity could achieve there.

OUTPUT INTERPRETATION (write into the manuscript):
  headline_fraction_of_matched_ceiling = 3.06 / (matched ceiling mean)
  Reported in the manuscript conclusion and RESULTS.md §9.

Run:
    python 10_oracle_ceiling.py --size S --K 3
    python 10_oracle_ceiling.py --size S --full          # optional, K=1
Checkpoints: oracle_matched_progress.json / oracle_full_progress.json
"""
import argparse, copy, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256

TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3

LR    = 1e-3
STEPS = 50

HEADLINE = (3.06, 0.27)   # label-free filtered TENT, for the printout


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


def oracle_finetune(base_model, window_batches, device, order, full):
    """Labeled CE fine-tune on the window. Matched capacity (BN affine
    only, BN stats also updating in train mode, momentum 0.1 -- exactly
    the TTA parameter set) unless full=True (all params). Eval once at
    the end (final rule)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model)
    if full:
        m.train()
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mod.momentum = 0.1
        m.requires_grad_(True)
        params = [p for p in m.parameters() if p.requires_grad]
    else:
        m.requires_grad_(False); params = []
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mod.requires_grad_(True); mod.train(); mod.momentum = 0.1
                if mod.weight is not None: params.append(mod.weight)
                if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    for s in range(STEPS):
        b = window_batches[order[s % len(order)]]
        lo, y = fwd(m, b, device)
        loss = F.cross_entropy(lo, torch.as_tensor(y).long().to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    final = accuracy_on_batches(m, window_batches, device)
    return frozen, final


def load_ckpt(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {"done": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--full", action="store_true",
                    help="all-parameter fine-tune (loose ceiling), K forced to 1")
    args = ap.parse_args()
    K = 1 if args.full else args.K
    tag = "full" if args.full else "matched"
    ckpt_path = f"oracle_{tag}_progress.json"
    ckpt = load_ckpt(ckpt_path)
    if ckpt["done"]:
        print(f"[RESUME] {len(ckpt['done'])} units in {ckpt_path}.\n")

    print(f"=== ORACLE CEILING ({tag}) on {TEST_WEEK}: labeled CE, "
          f"lr={LR:.0e}, steps={STEPS}, K={K} ===")
    tmodel, tloader, device = build(args.size, TEST_WEEK)
    print(f"device={device}")
    probe_cap = TARGET_TEST_EVAL_BATCHES*REPEATS + 50
    n_avail = count_available_batches(tloader, cap=probe_cap)
    neb = min(TARGET_TEST_EVAL_BATCHES, n_avail // REPEATS)
    if neb < MIN_TEST_EVAL_BATCHES:
        print(f"  [STOP] only {neb}/window."); sys.exit(1)
    print(f"  n_avail={n_avail}, using {neb}/window\n")

    tmodel, tloader, device = build(args.size, TEST_WEEK)
    windows = [collect_window(tloader, skip=r*neb, n=neb, label=f"window {r+1}")
               for r in range(REPEATS)]
    tbase = accuracy_on_batches(tmodel, windows[0], device)
    print(f"W-47 self-check (window 1) = {tbase:.4f}")
    if not (0.62 <= tbase <= 0.78):
        print("  [STOP] baseline out of range."); sys.exit(1)

    total = REPEATS*K; done = 0; t0 = time.time()
    print(f"\n=== {K} orderings x {REPEATS} windows = {total} units ===\n")
    for r, w in enumerate(windows):
        for k in range(K):
            key = f"{r}_{k}"
            if key in ckpt["done"]:
                done += 1; continue
            rng = np.random.default_rng(1000*r + k)
            order = list(rng.permutation(len(w)))
            fr, ad = oracle_finetune(tmodel, w, device, order, args.full)
            ckpt["done"][key] = {"frozen": fr, "recovery": ad-fr}
            with open(ckpt_path, "w") as f: json.dump(ckpt, f)
            done += 1
            print(f"  [{done}/{total}] window {r+1} ord {k+1}: frozen={fr:.4f} "
                  f"oracle={ad:.4f} rec={(ad-fr)*100:+.2f}p "
                  f"(elapsed {(time.time()-t0)/60:.1f}m)")

    per_window = {r: [] for r in range(REPEATS)}
    for key, v in ckpt["done"].items():
        per_window[int(key.split("_")[0])].append(v["recovery"])
    all_recs = [v["recovery"] for v in ckpt["done"].values()]
    pm, ps = float(np.mean(all_recs))*100, float(np.std(all_recs))*100
    pw = [float(np.mean(per_window[r]))*100 for r in range(REPEATS)]

    print(f"\n==== ORACLE ({tag}) RESULT (S, n={neb}/window, K={K}) ====")
    print(f"  oracle recovery: {pm:+.2f}p ± {ps:.2f}")
    print(f"  per-window means: {', '.join(f'{x:+.2f}' for x in pw)}")
    print(f"\n  Calibration of the label-free headline:")
    print(f"    headline {HEADLINE[0]:+.2f}p ± {HEADLINE[1]:.2f}"
          f"  =  {HEADLINE[0]/pm:.1%} of the {tag} labeled ceiling")
    print(f"    (gap fractions: headline 13.5% of 22.65p; "
          f"{tag} ceiling {pm/22.65:.1%} of 22.65p)")
    print(f"\n  Recorded in RESULTS.md §9 and the manuscript conclusion.")


if __name__ == "__main__":
    main()
