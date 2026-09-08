#!/usr/bin/env python3
r"""
19_w46_stability_reference.py -- Experiment D of the pre-registration.

WHAT IT REPAIRS:
  The pre-registered switch-point selection rejected the two-phase schedule
  using a stability ceiling of 0.12 points, described as roughly twice the
  pure method's 0.06. That 0.06 was measured on W-2022-47 windows of 200
  batches; the switch-point ordering standard deviations were measured on a
  W-2022-46 window of 60 batches. Those are not comparable. With 50 steps on
  a 60-batch window an ordering consumes 50 of 60 batches, so different
  orderings select almost the same subset and differ mainly in sequence; on a
  200-batch window an ordering consumes 50 of 200 and the subsets differ
  substantially. The ceiling was therefore calibrated against the wrong
  reference.

  This script supplies the missing matched reference: pure filtered adaptation
  at the frozen configuration, on the identical W-2022-46 60-batch window,
  with the identical K=5 seeded orderings the switch-point selection used.

  It touches no report-week data and changes no recorded number.

PRE-COMMITTED READING, from the locked pre-registration Section 6:
  D-R1  order-std at or below 0.06p -> the 0.12 ceiling was calibrated
        correctly and the switch-point rejection stands as recorded.
  D-R2  order-std above 0.06p -> the ceiling was stricter than intended. The
        rejection is then re-justified in the manuscript on the evidence that
        does not depend on the ceiling, namely the single orderings at switch
        points 62 and 75 that drove accuracy below the frozen baseline
        (-1.28p and -1.13p), and the mis-calibration is stated in the text.
        The numerical outcome of the original selection is NOT revised,
        because it was executed as pre-registered; only its interpretation is
        corrected.

Run:
    python scripts/19_w46_stability_reference.py --size S
Output: w46_stability_reference.json. Roughly half an hour on CPU.
"""
import argparse, copy, json, os, time
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
VAL_WEEK   = "W-2022-46"
BATCH      = 256
OUT_JSON   = "w46_stability_reference.json"

TUNE_EVAL_BATCHES = 60          # identical to 11_switchpoint_select.py
LR, STEPS, QUANT, BN_MOM = 1e-3, 50, 0.5, 0.1
K = 5                           # identical seeded orderings

ANCHOR_W46_FROZEN = 0.7534749348958333
PURE_W47_ORDER_STD = 0.06       # the value the 0.12 ceiling was set from
CEILING = 0.12
PURE50_W46_NATURAL = 2.68       # single natural-order reference on this window


def build(size, week):
    import torch
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import mm_cesnet_v2, MM_CESNET_V2_Weights
    weights = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
    model = mm_cesnet_v2(weights=weights, model_dir=MODEL_DIR)
    model.eval()
    tr = weights.transforms
    ds = CESNET_QUIC22(DATA_DIR, size=size)
    kw = dict(dataset=ds, apps_selection=AppSelection.ALL_KNOWN,
              train_period_name=TRAIN_WEEK, test_period_name=week,
              batch_size=BATCH, train_workers=0, test_workers=0,
              use_packet_histograms=True,
              ppi_transform=tr.get("ppi_transform"),
              flowstats_transform=tr.get("flowstats_transform"),
              flowstats_phist_transform=tr.get("flowstats_phist_transform"))
    kw = {k: v for k, v in kw.items() if v is not None}
    cfg = DatasetConfig(**kw)
    ds.set_dataset_config_and_initialize(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, ds.get_test_dataloader(), device


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


def collect(loader, skip, n):
    out = []
    for i, b in enumerate(loader):
        if i < skip: continue
        out.append(b)
        if len(out) >= n: break
    return out


def accuracy_on_batches(model, batches, device):
    import torch
    from sklearn.metrics import accuracy_score
    def _run():
        was = {n: mod.training for n, mod in model.named_modules()}
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for b in batches:
                lo, y = fwd(model, b, device)
                ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
        for n, mod in model.named_modules():
            mod.train(was[n])
        return accuracy_score(np.concatenate(ys), np.concatenate(ps))
    return guarded_eval(model, _run)


def pure_filtered(base_model, window, device, order):
    """Exactly the frozen episodic method: filtered TENT, 50 steps, statistics
    updating throughout, evaluated once at the final step."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = BN_MOM
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias is not None:   params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    for s in range(STEPS):
        lo, _ = fwd(m, window[order[s % len(order)]], device)
        ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return accuracy_on_batches(m, window, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    args = ap.parse_args()

    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    if res:
        print(f"[RESUME] {len(res)} orderings already done\n")

    print("=== EXPERIMENT D: matched stability reference on W-2022-46 ===")
    print(f"    pure filtered adaptation, lr={LR:.0e}, steps={STEPS}, "
          f"q={QUANT}, K={K}")
    print(f"    identical {TUNE_EVAL_BATCHES}-batch window and identical "
          f"seeds to 11_switchpoint_select.py")
    print(f"    W-2022-47 is not touched\n")

    model, loader, device = build(args.size, VAL_WEEK)
    print(f"device={device}")
    w = collect(loader, 0, TUNE_EVAL_BATCHES)
    frozen = accuracy_on_batches(model, w, device)
    assert_anchor(frozen, ANCHOR_W46_FROZEN, tol=0.0,
                  name="W-46 60-batch frozen (switch-point anchor)")
    print(f"  [ANCHOR OK] W-46 frozen = {frozen!r}  (n={len(w)} batches)\n")

    t0 = time.time()
    for k in range(K):
        if str(k) in res: continue
        rng = np.random.default_rng(1000 * 0 + k)     # identical to script 11
        order = list(rng.permutation(len(w)))
        acc = pure_filtered(model, w, device, order)
        res[str(k)] = {"k": k, "frozen": frozen, "adapted": acc,
                       "recovery": (acc - frozen) * 100}
        json.dump(res, open(OUT_JSON, "w"), indent=1)
        print(f"  ordering {k}: adapted={acc:.4f}  "
              f"rec={(acc-frozen)*100:+.2f}p  ({(time.time()-t0)/60:.1f}m)")

    r = np.array([res[str(k)]["recovery"] for k in range(K)])
    std = float(r.std())
    print(f"\n==== EXPERIMENT D RESULT ====")
    print(f"  pure filtered on the W-46 60-batch window, K={K}")
    print(f"  per-ordering: " + " / ".join(f"{v:+.2f}" for v in r))
    print(f"  mean {r.mean():+.2f}p, order-std {std:.2f}p")
    print(f"  (single natural-order reference on this window: "
          f"{PURE50_W46_NATURAL:+.2f}p)")
    print(f"\n  the value the 0.12p ceiling was set from, measured on W-47 "
          f"200-batch windows: {PURE_W47_ORDER_STD:.2f}p")
    print(f"  the matched value, measured here on the SAME window the "
          f"switch-point stds came from: {std:.2f}p")

    if std <= PURE_W47_ORDER_STD:
        print(f"\n  D-R1: order-std {std:.2f}p is at or below "
              f"{PURE_W47_ORDER_STD:.2f}p. The {CEILING:.2f}p ceiling was "
              f"calibrated correctly and the switch-point rejection stands as "
              f"recorded, with no change to its interpretation.")
    else:
        implied = 2 * std
        print(f"\n  D-R2: order-std {std:.2f}p EXCEEDS "
              f"{PURE_W47_ORDER_STD:.2f}p. A ceiling built the same way from "
              f"this matched reference would have been {implied:.2f}p, not "
              f"{CEILING:.2f}p.")
        sel = 0.27   # recorded order-std of the selected switch point s*=25
        still = "still" if sel > implied else "NOT"
        print(f"  The selected switch point s*=25 had order-std {sel:.2f}p, "
              f"which {still} exceeds the matched ceiling {implied:.2f}p.")
        print(f"  Per the pre-registration the recorded outcome is NOT "
              f"revised, because the selection was executed as written. The "
              f"manuscript states the mis-calibration and re-justifies the "
              f"rejection on the evidence that does not depend on the "
              f"ceiling: single orderings at switch points 62 and 75 drove "
              f"accuracy below the frozen baseline by 1.28 and 1.13 points, "
              f"and a method that can underperform doing nothing is not "
              f"deployable.")
    print(f"\n  raw: {OUT_JSON}")


if __name__ == "__main__":
    main()
