#!/usr/bin/env python3
r"""
11_switchpoint_probe.py -- coarse switch-point sensitivity, TUNING WEEK ONLY.

PURPOSE:
  Sensitivity of the two-phase schedule to the switch point, measured
  WITHOUT promoting the schedule beyond a post-hoc observation and
  WITHOUT touching the report week. The probe runs the two-phase schedule
  with the statistics freeze at step 25, 50, and 75 (100 total steps,
  lr=1e-3, q=0.5) on the TUNING week W-2022-46 only, with the same
  deterministic natural-order adaptation used for tuning. W-2022-47 is
  never touched, so the leakage-clean status of every reported number
  is preserved, and the result supports exactly one sentence in
  Section III-E about whether the accidental switch point (50) is
  special.

  Reference points on the same W-46 window (from the clean tuning
  grid, final-eval): pure-50 +2.68p, pure-100 (never measured clean on
  W-46 at final eval under natural order in the current record; the
  probe prints the pure-100 value too, for completeness).

RUNTIME: 4 configs (switch 25/50/75 + pure-100 reference) x ~8-9 min
  on the 60-batch W-46 window = ~35 min CPU.

Run:
    python 11_switchpoint_probe.py --size S
Output: switchpoint_probe.json + console.
"""
import argparse, copy, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
VAL_WEEK   = "W-2022-46"
BATCH      = 256
OUT_JSON   = "switchpoint_probe.json"

TUNE_EVAL_BATCHES = 60
LR    = 1e-3
QUANT = 0.5
TOTAL = 100
SWITCHES = [25, 50, 75, None]   # None = pure TENT for all 100 steps

PURE50_W46 = 2.68   # clean tuning-grid reference on this same window


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


def accuracy_on_batches(model, batches, device):
    import torch
    from sklearn.metrics import accuracy_score
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in batches:
            lo, y = fwd(model, b, device)
            ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
    return accuracy_score(np.concatenate(ys), np.concatenate(ps))


def two_phase(base_model, window_batches, device, switch):
    """Natural-order (deterministic, matching tuning convention),
    filtered TENT for `switch` steps, then explicit m.eval() freeze,
    then filtered affine-only to TOTAL. switch=None = pure to TOTAL.
    Eval once at the end."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = 0.1
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    n = len(window_batches)
    for s in range(TOTAL):
        if switch is not None and s == switch:
            m.eval()   # explicit statistics freeze
        b = window_batches[s % n]
        lo, _ = fwd(m, b, device)
        ent = -(F.softmax(lo,1) * F.log_softmax(lo,1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    final = accuracy_on_batches(m, window_batches, device)
    return frozen, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    args = ap.parse_args()

    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f: results = json.load(f)
        print(f"[RESUME] configs done: {list(results)}\n")

    print(f"=== SWITCH-POINT PROBE on {VAL_WEEK} (tuning week ONLY; "
          f"W-47 untouched) ===")
    vmodel, vloader, device = build(args.size, VAL_WEEK)
    print(f"device={device}")
    w = collect_window(vloader, skip=0, n=TUNE_EVAL_BATCHES)
    vbase = accuracy_on_batches(vmodel, w, device)
    print(f"W-46 frozen acc = {vbase:.4f} (n={len(w)})\n")

    t0 = time.time()
    for sw in SWITCHES:
        key = "pure100" if sw is None else f"switch{sw}"
        if key in results:
            print(f"  {key:>9}: rec={results[key]*100:+.2f}p  [cached]")
            continue
        fr, ad = two_phase(vmodel, w, device, sw)
        results[key] = ad - fr
        with open(OUT_JSON, "w") as f: json.dump(results, f, indent=1)
        print(f"  {key:>9}: frozen={fr:.4f} adapted={ad:.4f} "
              f"rec={(ad-fr)*100:+.2f}p (elapsed {(time.time()-t0)/60:.1f}m)")

    print(f"\n==== SWITCH-POINT SENSITIVITY (W-46, natural order, "
          f"final-eval) ====")
    print(f"  pure-50 reference (clean grid): +{PURE50_W46:.2f}p")
    for sw in [25, 50, 75]:
        k = f"switch{sw}"
        if k in results:
            print(f"  two-phase, freeze@{sw:>2}: {results[k]*100:+.2f}p")
    if "pure100" in results:
        print(f"  pure-100 (no freeze)  : {results['pure100']*100:+.2f}p")
    print(f"\n  One-sentence use in III-E: report whether the 25/50/75 values")
    print(f"  are close (switch point not special) or spread (sensitive),")
    print(f"  on the tuning week, without touching W-47.")


if __name__ == "__main__":
    main()
