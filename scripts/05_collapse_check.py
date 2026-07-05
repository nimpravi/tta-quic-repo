#!/usr/bin/env python3
r"""
05_collapse_check.py -- does entropy minimization buy accuracy by collapsing
predictions onto head classes?

WHY THIS EXISTS:
  Entropy minimization is known to be able to raise overall accuracy while
  concentrating predictions on confident (usually majority) classes and
  abandoning tail classes. This script compares frozen
  vs adapted predictions per window on:

    - accuracy (sanity: should reproduce the headline within order noise)
    - macro-F1 (unweighted over classes -- the collapse-sensitive metric)
    - number of distinct predicted classes
    - normalized entropy of the PREDICTED-class distribution
      (1.0 = uniform; a large drop signals concentration)
    - count of classes whose recall improved / worsened / unchanged
    - top-5 classes by gained and by lost predicted mass

  DECISION RULE (fixed in advance): the headline claim survives if adapted
  macro-F1 is >= frozen macro-F1 (i.e., the recovery is not purchased by
  killing tail classes). If macro-F1 drops while accuracy rises, the paper
  must report both numbers and qualify the claim.

PROTOCOL:
  Frozen config (lr=1e-3, steps=50, quant as given), one seeded ordering
  per window (k=0, same rng scheme as 02/03: default_rng(1000*window + 0)).
  SANITY: at quant=0.5, per-window accuracy deltas should reproduce the
  clean headline's k=0 (ord 1) values: +3.41 / +2.94 / +2.97. A mismatch
  beyond ~0.05p means something is off; stop and investigate.
  Run once with --quant 0.5 (headline condition; required) and, if time
  permits, once with --quant 1.0 (writes to a separate JSON).

Run:
    python 05_collapse_check.py --size S --quant 0.5
    # per-window results checkpointed to collapse_check_q{quant}.json
"""
import argparse, copy, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
IN_PERIOD  = 0.867
BATCH      = 256

TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3

# Frozen config from W-46 tuning (see scripts 01/02). Quantile is a CLI arg
# so the same script covers the mechanism condition.
LR    = 1e-3
STEPS = 50   # re-tuned frozen config (final-eval, 2026-07-02)


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


def predict_on_batches(model, batches, device):
    """Like accuracy_on_batches but returns (y_true, y_pred) arrays."""
    import torch
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in batches:
            lo, y = fwd(model, b, device)
            ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
    return np.concatenate(ys), np.concatenate(ps)


def tent_adapt(base_model, window_batches, device, lr, steps, quantile, order):
    """Same adaptation as tent_final() in 02, but returns the adapted MODEL
    so we can extract full predictions (02 only returned accuracies)."""
    import torch, torch.nn as nn, torch.nn.functional as F
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
    return m


def pred_dist_entropy(preds, n_classes):
    """Normalized entropy of the predicted-class distribution. 1.0 = uniform."""
    counts = np.bincount(preds, minlength=n_classes).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    h = -(nz * np.log(nz)).sum()
    return float(h / np.log(n_classes))


def window_report(y, p_frozen, p_adapted):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    classes = np.unique(np.concatenate([y, p_frozen, p_adapted]))
    n_classes = int(classes.max()) + 1
    labels_present = np.unique(y)

    rec_f = recall_score(y, p_frozen,  labels=labels_present, average=None, zero_division=0)
    rec_a = recall_score(y, p_adapted, labels=labels_present, average=None, zero_division=0)
    d = rec_a - rec_f

    # predicted-mass shifts (fraction of all predictions moving to/from a class)
    n = len(y)
    mass_f = np.bincount(p_frozen,  minlength=n_classes) / n
    mass_a = np.bincount(p_adapted, minlength=n_classes) / n
    dmass = mass_a - mass_f
    top_gain = np.argsort(dmass)[::-1][:5]
    top_loss = np.argsort(dmass)[:5]

    return {
        "n_samples": int(n),
        "n_true_classes": int(len(labels_present)),
        "acc_frozen":  float(accuracy_score(y, p_frozen)),
        "acc_adapted": float(accuracy_score(y, p_adapted)),
        "macroF1_frozen":  float(f1_score(y, p_frozen,  labels=labels_present,
                                          average="macro", zero_division=0)),
        "macroF1_adapted": float(f1_score(y, p_adapted, labels=labels_present,
                                          average="macro", zero_division=0)),
        "distinct_pred_frozen":  int(len(np.unique(p_frozen))),
        "distinct_pred_adapted": int(len(np.unique(p_adapted))),
        "pred_entropy_frozen":  pred_dist_entropy(p_frozen,  n_classes),
        "pred_entropy_adapted": pred_dist_entropy(p_adapted, n_classes),
        "classes_recall_up":   int((d >  1e-12).sum()),
        "classes_recall_down": int((d < -1e-12).sum()),
        "classes_recall_same": int((np.abs(d) <= 1e-12).sum()),
        "top5_gain_class_ids": [int(c) for c in top_gain],
        "top5_gain_dmass":     [float(dmass[c]) for c in top_gain],
        "top5_loss_class_ids": [int(c) for c in top_loss],
        "top5_loss_dmass":     [float(dmass[c]) for c in top_loss],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    ap.add_argument("--quant", type=float, default=0.5)
    args = ap.parse_args()
    out_json = f"collapse_check_q{args.quant}_steps50.json"

    results = {}
    if os.path.exists(out_json):
        with open(out_json) as f: results = json.load(f)
        print(f"[RESUME] {len(results)} window(s) already in {out_json}.\n")

    print(f"=== COLLAPSE CHECK on {TEST_WEEK} (quant={args.quant}, frozen config) ===")
    tmodel, tloader, device = build(args.size, TEST_WEEK)
    print(f"device={device}")
    probe_cap = TARGET_TEST_EVAL_BATCHES*REPEATS + 50
    n_avail = count_available_batches(tloader, cap=probe_cap)
    test_eval_batches = min(TARGET_TEST_EVAL_BATCHES, n_avail // REPEATS)
    if test_eval_batches < MIN_TEST_EVAL_BATCHES:
        print(f"  [STOP] only {test_eval_batches}/repeat."); sys.exit(1)
    print(f"  n_avail={n_avail}, using {test_eval_batches}/window\n")

    tmodel, tloader, device = build(args.size, TEST_WEEK)
    windows = [collect_window(tloader, skip=r*test_eval_batches, n=test_eval_batches,
                              label=f"window {r+1}") for r in range(REPEATS)]

    t0 = time.time()
    for r, w in enumerate(windows):
        key = str(r)
        if key in results:
            continue
        rng = np.random.default_rng(1000*r + 0)          # k=0 ordering, matches 02
        order = list(rng.permutation(len(w)))
        y, p_frozen = predict_on_batches(tmodel, w, device)
        adapted = tent_adapt(tmodel, w, device, LR, STEPS, args.quant, order)
        _, p_adapted = predict_on_batches(adapted, w, device)
        rep = window_report(y, p_frozen, p_adapted)
        results[key] = rep
        with open(out_json, "w") as f: json.dump(results, f, indent=1)
        print(f"  window {r+1}: "
              f"acc {rep['acc_frozen']:.4f} -> {rep['acc_adapted']:.4f} "
              f"({(rep['acc_adapted']-rep['acc_frozen'])*100:+.2f}p) | "
              f"macroF1 {rep['macroF1_frozen']:.4f} -> {rep['macroF1_adapted']:.4f} "
              f"({(rep['macroF1_adapted']-rep['macroF1_frozen'])*100:+.2f}p) | "
              f"distinct preds {rep['distinct_pred_frozen']} -> {rep['distinct_pred_adapted']} | "
              f"pred-entropy {rep['pred_entropy_frozen']:.3f} -> {rep['pred_entropy_adapted']:.3f} | "
              f"recall up/down/same {rep['classes_recall_up']}/{rep['classes_recall_down']}"
              f"/{rep['classes_recall_same']} "
              f"(elapsed {(time.time()-t0)/60:.1f}m)")

    # ---- SUMMARY ----
    accs_d  = [results[str(r)]["acc_adapted"] - results[str(r)]["acc_frozen"] for r in range(REPEATS)]
    f1s_d   = [results[str(r)]["macroF1_adapted"] - results[str(r)]["macroF1_frozen"] for r in range(REPEATS)]
    ent_d   = [results[str(r)]["pred_entropy_adapted"] - results[str(r)]["pred_entropy_frozen"] for r in range(REPEATS)]
    print(f"\n==== COLLAPSE CHECK SUMMARY (quant={args.quant}, k=0 orderings) ====")
    print(f"  mean accuracy delta : {np.mean(accs_d)*100:+.2f}p  (sanity: k=0 headline vals +3.41/+2.94/+2.97)")
    print(f"  mean macro-F1 delta : {np.mean(f1s_d)*100:+.2f}p")
    print(f"  mean pred-entropy delta: {np.mean(ent_d):+.4f} (negative = concentration)")
    if np.mean(f1s_d) >= 0:
        print("  VERDICT: macro-F1 does not degrade -> recovery is not explained "
              "by head-class collapse. Report macro-F1 alongside accuracy.")
    else:
        print("  VERDICT: macro-F1 DEGRADES while accuracy improves -> partial "
              "collapse. The manuscript must report both and qualify the claim.")
    print(f"  Full per-class details in {out_json}.")


if __name__ == "__main__":
    main()
