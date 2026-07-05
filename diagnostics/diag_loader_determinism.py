#!/usr/bin/env python3
r"""
diagnose_nondeterminism.py — find the source of run-to-run variation.

Does NOT run TENT or touch the main script. Answers, in order:

  Q1. Within a single loader, does iterating twice give the same batch order?
      (tests: is get_test_dataloader() itself reproducible when re-iterated?)

  Q2. Across two fresh loaders from the same build(), same batches in same order?
      (tests: is loader construction deterministic?)

  Q3. Does torch/numpy seeding before build() make Q2 pass if it was failing?
      (tests: is the nondeterminism seed-controllable — the thing we need for step 2?)

It fingerprints each batch by a cheap hash of the label vector (and ppi sum),
so we compare ORDER and CONTENT without running the model. Uses W-46 at the
tuning eval size, matching where the shift was observed.
"""
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
VAL_WEEK   = "W-2022-46"
BATCH      = 256
N_PROBE    = 12    # first 12 batches is plenty to detect order/content drift


def _extract(batch):
    """Return (label_vector, ppi_sum) for fingerprinting, reusing the same
    tuple-parsing logic as the main script's fwd()."""
    parts = list(batch) if isinstance(batch, (tuple, list)) else [batch]
    ppi = y = None
    for p in parts:
        arr = np.asarray(p)
        if arr.ndim == 3:                                    ppi = arr
        elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer): y = arr
    return y, (None if ppi is None else float(np.asarray(ppi, dtype=np.float64).sum()))


def fingerprint(batch):
    y, ppi_sum = _extract(batch)
    if y is None:
        return ("NO_LABELS", ppi_sum)
    y = np.asarray(y)
    # hash of label sequence + count + ppi magnitude = identity of this batch
    return (int(y.shape[0]), int(y.sum()), hash(y.tobytes()), None if ppi_sum is None else round(ppi_sum, 3))


def build_loader(size, seed=None):
    import torch
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import MM_CESNET_V2_Weights

    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
        try:
            import random; random.seed(seed)
        except Exception:
            pass

    weights = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
    transforms = weights.transforms
    ds = CESNET_QUIC22(DATA_DIR, size=size)
    cfg_kwargs = dict(
        dataset=ds, apps_selection=AppSelection.ALL_KNOWN,
        train_period_name=TRAIN_WEEK, test_period_name=VAL_WEEK,
        batch_size=BATCH, train_workers=0, test_workers=0,
        use_packet_histograms=True,
        ppi_transform=transforms.get("ppi_transform"),
        flowstats_transform=transforms.get("flowstats_transform"),
        flowstats_phist_transform=transforms.get("flowstats_phist_transform"),
    )
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if v is not None}
    cfg = DatasetConfig(**cfg_kwargs)
    ds.set_dataset_config_and_initialize(cfg)
    return ds.get_test_dataloader()


def first_n_prints(loader, n=N_PROBE):
    fps = []
    for i, b in enumerate(loader):
        fps.append(fingerprint(b))
        if len(fps) >= n:
            break
    return fps


def compare(a, b, label):
    same_order = (a == b)
    same_set   = (sorted(map(str, a)) == sorted(map(str, b)))
    print(f"\n[{label}]")
    print(f"  identical order:   {same_order}")
    print(f"  identical content: {same_set}  (same batches, possibly reordered)")
    if not same_order and same_set:
        print("  -> loader RE-ORDERS between iterations (shuffling); content stable.")
    elif not same_set:
        print("  -> loader draws DIFFERENT batches (sampling nondeterminism).")
    else:
        print("  -> fully reproducible.")
    return same_order, same_set


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    args = ap.parse_args()
    print(f"size={args.size}, week={VAL_WEEK}, probing first {N_PROBE} batches\n")

    # Q1: iterate the SAME loader twice
    print("=== Q1: same loader, iterated twice ===")
    L = build_loader(args.size)
    a1 = first_n_prints(L)
    a2 = first_n_prints(L)
    q1_order, q1_set = compare(a1, a2, "Q1 same-loader re-iteration")

    # Q2: two FRESH loaders, no seeding
    print("\n=== Q2: two fresh build_loader() calls, no seed ===")
    b1 = first_n_prints(build_loader(args.size))
    b2 = first_n_prints(build_loader(args.size))
    q2_order, q2_set = compare(b1, b2, "Q2 fresh-build no-seed")

    # Q3: two fresh loaders WITH the same seed
    print("\n=== Q3: two fresh build_loader() calls, both seed=0 ===")
    c1 = first_n_prints(build_loader(args.size, seed=0))
    c2 = first_n_prints(build_loader(args.size, seed=0))
    q3_order, q3_set = compare(c1, c2, "Q3 fresh-build seed=0")

    print("\n================ VERDICT ================")
    if q2_order:
        print("Loader is deterministic across fresh builds even WITHOUT seeding.")
        print("=> The grid shift was NOT loader ordering. Suspect deepcopy/BN or")
        print("   float-reduction threading. Report this; I'll dig there next.")
    elif q3_order:
        print("Loader varies across fresh builds, but seeding FIXES it.")
        print("=> Nondeterminism is loader sampling/shuffling, and it IS")
        print("   seed-controllable. Step 2 seeding should seed before build().")
    else:
        print("Loader varies across fresh builds and seeding did NOT fix order.")
        print("=> Seed is applied in the wrong place or loader ignores global seed.")
        print("   Need to find the loader's own shuffle/generator knob. Report and")
        print("   I'll locate the specific parameter.")
    print("========================================")


if __name__ == "__main__":
    main()
