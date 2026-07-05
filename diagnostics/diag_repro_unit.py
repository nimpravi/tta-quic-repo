#!/usr/bin/env python3
r"""
diag_repro_unit.py -- rerun ONE archived unit to localize the reproducibility break.

CONTEXT:
  The archived K=8 headline run (results/raw/headline_K8_run.txt) recorded
  window-1 recoveries in [+4.36, +5.10] across 8 seeded orderings. A fresh
  run of the identical (window 1, k=0) unit produced +3.35p, below the
  archived minimum, while every frozen accuracy remained bit-identical to
  the archive. Data, windows, weights, and eval path are therefore
  unchanged; the adaptation numerics moved.

WHAT THIS SCRIPT DOES:
  Reruns exactly one unit -- window 1 (skip=0, n=200), ordering k=0,
  rng = numpy.random.default_rng(1000*0 + 0) -- using the adaptation code
  path of scripts/02_errorbars.py VERBATIM, and prints the versions of
  every numerically relevant library.

READ THE RESULT LIKE THIS (rules fixed before running):
  - recovery in [+4.36, +5.10]  -> archive reproduces; the discrepancy is
    a bug in the new analysis scripts. Stop and inspect them; do not
    rerun anything else.
  - recovery near +3.35 (say within ±0.3p) -> environment drift confirmed.
    The archived headline (K=8) and mechanism (K=3) numbers belong to an
    environment that no longer exists. Remediation: EITHER restore the
    original tent-env (if it still exists, rerun the new scripts inside
    it), OR pin the current environment (pip freeze > requirements-lock.txt)
    and rerun scripts 02 (K=8, or K=5 fallback), 03, 04, 05 under it so
    every reported number shares one environment.
  - anything else -> report both numbers; do not draft.

RUNTIME: one adaptation + two window evals, ~12 min CPU.

Run (inside the SAME environment you used for scripts 04/05/06):
    python diag_repro_unit.py --size S
Then, if the original tent-env still exists, run it there too and compare.
"""
import argparse, copy, sys
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256

EVAL_BATCHES = 200
LR    = 1e-3
STEPS = 100
QUANT = 0.5

ARCHIVED_W1_BAND = (4.36, 5.10)   # from results/raw/headline_K8_run.txt
NEW_RUN_VALUE    = 3.35           # from 05_collapse_check.py, window 1, k=0


def print_versions():
    import platform
    print("=== ENVIRONMENT ===")
    print(f"  python  : {platform.python_version()} ({platform.platform()})")
    for pkg in ["torch", "numpy", "sklearn", "cesnet_datazoo", "cesnet_models"]:
        try:
            mod = __import__(pkg)
            print(f"  {pkg:15s}: {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  {pkg:15s}: NOT INSTALLED")
    try:
        import torch
        print(f"  torch threads  : {torch.get_num_threads()} "
              f"(interop {torch.get_num_interop_threads()})")
    except Exception:
        pass
    print()


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


def tent_final(base_model, window_batches, device, lr, steps, quantile, order):
    """VERBATIM copy of tent_final from scripts/02_errorbars.py."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = 0.1
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=lr)
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
    final = accuracy_on_batches(m, window_batches, device)
    return frozen, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    args = ap.parse_args()

    print_versions()

    print(f"=== REPRO UNIT: {TEST_WEEK}, window 1 (skip=0, n={EVAL_BATCHES}), "
          f"ordering k=0, lr={LR:.0e} steps={STEPS} quant={QUANT} ===")
    tmodel, tloader, device = build(args.size, TEST_WEEK)
    print(f"device={device}")
    w = collect_window(tloader, skip=0, n=EVAL_BATCHES, label="window 1")

    rng = np.random.default_rng(1000*0 + 0)
    order = list(rng.permutation(len(w)))

    fr, ad = tent_final(tmodel, w, device, LR, STEPS, QUANT, order)
    rec = (ad - fr) * 100
    print(f"\n  frozen  = {fr:.10f}  (archive: 0.7223901367)")
    print(f"  adapted = {ad:.10f}")
    print(f"  recovery = {rec:+.2f}p")

    lo, hi = ARCHIVED_W1_BAND
    print("\n=== VERDICT ===")
    if lo <= rec <= hi:
        print(f"  Recovery is inside the archived window-1 band [{lo:+.2f}, {hi:+.2f}].")
        print("  -> Archive REPRODUCES in this environment. The discrepancy is a bug")
        print("     in the new analysis scripts. Stop and inspect them.")
    elif abs(rec - NEW_RUN_VALUE) <= 0.3:
        print(f"  Recovery matches the new-run value ({NEW_RUN_VALUE:+.2f}p), below the")
        print(f"  archived band [{lo:+.2f}, {hi:+.2f}].")
        print("  -> ENVIRONMENT DRIFT confirmed. Archived adaptation numbers are not")
        print("     reproducible in this environment. Either restore the original")
        print("     tent-env and rerun scripts 04/05/06 there, or pin THIS environment")
        print("     (pip freeze > requirements-lock.txt) and rerun 02/03/04/05 under it.")
    else:
        print(f"  Recovery matches NEITHER the archived band [{lo:+.2f}, {hi:+.2f}]")
        print(f"  nor the new-run value ({NEW_RUN_VALUE:+.2f}p).")
        print("  -> Keep this log and investigate; do not rerun anything else yet.")
    if abs(fr - 0.72239013671875) > 1e-12:
        print("\n  [WARN] frozen accuracy does NOT match the archive bit-for-bit;")
        print("         the data/eval path also changed, not just adaptation numerics.")


if __name__ == "__main__":
    main()
