#!/usr/bin/env python3
r"""
07_w45_depth_probe.py -- locate the drift event WITHIN W-2022-45 (eval only).

CONTEXT:
  Script 06 measured the frozen W-44 model at 0.9552 on the first 600
  batches of W-45, far above both the cited ~0.867 and the earlier "~0.87
  on W-45" working note. The literature dates the Google certificate
  change to DURING week 45. If the datazoo test loader is time-ordered,
  the first 600 batches are early-week (pre-event) traffic, and accuracy
  should DROP partway through the week, toward the ~0.73 seen on W-46/47.

WHAT THIS SCRIPT DOES:
  Walks deeper into the W-45 test loader and evaluates thin windows
  (60 batches) at increasing offsets, tracing accuracy vs position.

READ THE RESULT LIKE THIS (rules fixed before running):
  - Accuracy starts ~0.95 and drops to ~0.73-0.80 at some offset ->
    hypothesis confirmed: loader is time-ordered, event is mid-week,
    0.9552 (early-W-45) is the honest PRE-SHIFT reference, and the paper's
    gap denominator becomes acc(early W-45) - acc(W-47). Bonus: this trace
    is a clean one-panel drift figure and corroborates the literature's
    event timing.
  - Accuracy stays ~0.95 at every offset -> hypothesis WRONG (loader may
    be shuffled, or the event does not manifest in this split). Do NOT
    adopt 0.9552 as the denominator yet; the cited-0.867 discrepancy is
    unexplained and needs investigation (check cesnet-datazoo docs for
    test-loader ordering semantics).
  - Anything intermediate/noisy -> record the trace and re-examine
    before choosing the denominator.

RUNTIME: data-only capacity walk (up to --cap batches) plus
  len(offsets) x 60 eval batches. Roughly 20-30 min CPU at defaults.

Run:
    python 07_w45_depth_probe.py --size S
    # optionally: --cap 3000 to walk deeper, --week W-2022-46 for other weeks
    # results appended to w45_depth_probe.json
"""
import argparse, sys, os, json
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
BATCH      = 256
OUT_JSON   = "w45_depth_probe.json"

PROBE_N    = 60          # thin windows: enough signal, cheap


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
    ap.add_argument("--week", default="W-2022-45")
    ap.add_argument("--cap",  type=int, default=2000,
                    help="max batches to walk when probing capacity")
    args = ap.parse_args()

    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f: results = json.load(f)

    print(f"=== DEPTH PROBE on {args.week} (frozen W-44 model, eval only) ===")
    model, loader, device = build(args.size, args.week)
    print(f"device={device}")

    # Single pass: walk the loader once, evaluating PROBE_N-batch windows at
    # geometric-ish offsets as we encounter them (avoids re-walking per offset).
    offsets = [0, 200, 400, 700, 1000, 1400, 1800]
    offsets = [o for o in offsets if o + PROBE_N <= args.cap]
    targets = {o: [] for o in offsets}
    week_res = results.get(args.week, {})

    n_seen = 0
    for b in loader:
        for o in offsets:
            if str(o) not in week_res and o <= n_seen < o + PROBE_N:
                targets[o].append(b)
        n_seen += 1
        if n_seen >= args.cap:
            break
    print(f"  walked {n_seen} batches (cap {args.cap})")

    for o in offsets:
        key = str(o)
        if key in week_res:
            print(f"  offset {o:>5}: acc = {week_res[key]:.4f}  [cached]")
            continue
        if len(targets[o]) < PROBE_N:
            print(f"  offset {o:>5}: only {len(targets[o])} batches available, skipping")
            continue
        a = accuracy_on_batches(model, targets[o], device)
        week_res[key] = float(a)
        results[args.week] = week_res
        with open(OUT_JSON, "w") as f: json.dump(results, f, indent=1)
        print(f"  offset {o:>5}: acc = {a:.4f}   "
              f"(flows ~{o*BATCH:,} to ~{(o+PROBE_N)*BATCH:,})")

    accs = [week_res[str(o)] for o in offsets if str(o) in week_res]
    if len(accs) >= 3:
        print("\n=== READING ===")
        drop = max(accs) - min(accs)
        print(f"  trace: {', '.join(f'{a:.3f}' for a in accs)}  (max-min = {drop:.3f})")
        if drop >= 0.05 and accs.index(min(accs)) > accs.index(max(accs)):
            print("  Accuracy DECLINES with depth -> time-ordered loader + intra-week")
            print("  event supported. Early-week accuracy is the pre-shift reference.")
        elif drop < 0.02:
            print("  Trace is FLAT -> the time-ordering/mid-week-event hypothesis is")
            print("  NOT supported at this depth. Do not adopt 0.9552 as denominator")
            print("  yet; check datazoo test-loader ordering semantics, or increase")
            print("  --cap if the split is much larger than the walked depth.")
        else:
            print("  Trace is intermediate/noisy; reported for completeness.")


if __name__ == "__main__":
    main()
