#!/usr/bin/env python3
r"""
06_inperiod_reference.py -- self-measured in-period reference (eval only).

WHY THIS EXISTS:
  The gap denominator currently mixes a CITED in-period number (~0.867 from
  the model's paper) with OUR measured W-47 accuracy under OUR protocol
  (200-batch windows, batch 256, weights' own transforms). This script measures the frozen W-44 model on:

    - W-2022-45  (adjacent post-training week, before the drift event):
      the self-consistent in-period reference, and
    - W-2022-46  (tuning week, for the record; not used in any claim)

  under the IDENTICAL windowing protocol used for W-47. No adaptation, no
  gradients -- pure evaluation, so this is fast (roughly the cost of three
  window evals per week).

  AFTER THIS RUN, the paper's gap and %-of-gap numbers should be recomputed
  as: gap = acc(W-45, self-measured) - acc(W-47, self-measured). If the
  self-measured W-45 lands near 0.867 the headline percentages barely move
  and the cited number can be mentioned as corroboration. Either way the
  denominator becomes internally consistent.

Run:
    python 06_inperiod_reference.py --size S
    # results appended to inperiod_reference.json
"""
import argparse, sys, os, json
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
BATCH      = 256
OUT_JSON   = "inperiod_reference.json"

REF_WEEKS  = ["W-2022-45", "W-2022-46"]   # W-45 is the reference; W-46 for the record

TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    args = ap.parse_args()

    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f: results = json.load(f)
        print(f"[RESUME] weeks already measured: {list(results)}\n")

    for week in REF_WEEKS:
        if week in results:
            continue
        print(f"=== FROZEN EVAL on {week} (W-44 model, no adaptation) ===")
        model, loader, device = build(args.size, week)
        print(f"device={device}")
        probe_cap = TARGET_TEST_EVAL_BATCHES*REPEATS + 50
        n_avail = count_available_batches(loader, cap=probe_cap)
        n_eval = min(TARGET_TEST_EVAL_BATCHES, n_avail // REPEATS)
        if n_eval < MIN_TEST_EVAL_BATCHES:
            print(f"  [SKIP] {week}: only {n_eval}/window available."); continue
        print(f"  n_avail={n_avail}, using {n_eval}/window")

        model, loader, device = build(args.size, week)
        accs = []
        for r in range(REPEATS):
            w = collect_window(loader, skip=r*n_eval, n=n_eval, label=f"window {r+1}")
            a = accuracy_on_batches(model, w, device)
            accs.append(a)
            print(f"  window {r+1}: acc = {a:.4f}")
        results[week] = {"per_window": accs,
                         "mean": float(np.mean(accs)),
                         "std":  float(np.std(accs)),
                         "n_per_window": n_eval}
        with open(OUT_JSON, "w") as f: json.dump(results, f, indent=1)
        print(f"  {week}: mean = {np.mean(accs):.4f} (std {np.std(accs):.4f})\n")

    if "W-2022-45" in results:
        ref = results["W-2022-45"]["mean"]
        print(f"==== SELF-MEASURED IN-PERIOD REFERENCE ====")
        print(f"  W-45 frozen acc = {ref:.4f} (cited paper value ~0.867)")
        print(f"  Recompute gap as: {ref:.4f} - 0.7287 = {ref-0.7287:.4f}")
        print(f"  Then: headline %%-of-gap = 4.34 / {(ref-0.7287)*100:.2f} = "
              f"{4.34/((ref-0.7287)*100):.1%}")
        print(f"  (Update RESULTS.md and all %-of-gap figures with the "
              f"self-measured denominator.)")


if __name__ == "__main__":
    main()
