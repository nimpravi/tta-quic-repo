#!/usr/bin/env python3
r"""
11_switchpoint_select.py -- PRE-REGISTERED switch-point selection, W-46 ONLY.

Governed by PREREGISTRATION_switchpoint.md
(SHA-256: 4ebd14fbe8b721e9bb86683febd724b4ba7b08fb8a2d7c83a7444c37f9712e45).

Supersedes the coarse 11_switchpoint_probe.py for the purpose of PROMOTION.
Differences, all mandated by the pre-registration:
  - Grid switch in {25, 37, 50, 62, 75} (finer; not forced onto 50).
  - K=5 seeded orderings per switch point (probe was natural-order only),
    so within-window order-std -- kill rule A -- is measurable HERE, on the
    tuning week, before W-47 is ever touched.
  - Deterministic selection: highest W-46 pooled mean; ties (<=0.05p)
    to the earlier switch point.

W-2022-47 IS NOT TOUCHED BY THIS SCRIPT. It runs on W-2022-46 only.

Selection output (s*, its pooled mean +- std, its order-std) is recorded
in PREREGISTRATION_switchpoint.md before the confirmatory W-47 run
(script 09 --switch s*). Kill rule A is checked here.

Run:
    python 11_switchpoint_select.py --size S
Output: switchpoint_select.json + console. Resumable.
"""
import argparse, copy, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
VAL_WEEK   = "W-2022-46"
BATCH      = 256
OUT_JSON   = "switchpoint_select.json"

TUNE_EVAL_BATCHES = 60          # matches the tuning-week window convention
LR       = 1e-3
QUANT    = 0.5
TOTAL    = 100
BN_MOM   = 0.1
SWITCHES = [25, 37, 50, 62, 75] # pre-registered grid
K        = 5                    # pre-registered orderings

# Pre-registered kill rule A (stability, evaluated on W-46 here):
ORDER_STD_CEILING = 0.12        # points
PURE50_W46 = 2.68               # clean tuning-grid reference, same window


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


def two_phase(base_model, window_batches, device, switch, order):
    """Filtered TENT for `switch` steps (stats updating), explicit m.eval()
    freeze, then filtered affine-only to TOTAL. `order` is the adaptation
    ordering (permutation of window indices). Eval once at the end."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = BN_MOM
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    n = len(window_batches)
    for s in range(TOTAL):
        if s == switch:
            m.eval()   # explicit statistics freeze
        b = window_batches[order[s % len(order)]]
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
        print(f"[RESUME] units done: {len(results)}\n")

    print(f"=== PRE-REGISTERED SWITCH-POINT SELECTION on {VAL_WEEK} ===")
    print(f"    grid={SWITCHES}  K={K}  (W-47 NOT touched)\n")
    vmodel, vloader, device = build(args.size, VAL_WEEK)
    print(f"device={device}")
    w = collect_window(vloader, skip=0, n=TUNE_EVAL_BATCHES)
    vbase = accuracy_on_batches(vmodel, w, device)
    print(f"W-46 frozen acc = {vbase:.4f} (n={len(w)})\n")

    total = len(SWITCHES) * K
    t0 = time.time(); done = 0
    for sw in SWITCHES:
        for k in range(K):
            key = f"s{sw}_k{k}"
            if key in results:
                done += 1; continue
            rng = np.random.default_rng(1000*0 + k)   # window index 0
            order = list(rng.permutation(len(w)))
            fr, ad = two_phase(vmodel, w, device, sw, order)
            results[key] = {"switch": sw, "k": k, "frozen": fr,
                            "recovery": ad - fr}
            with open(OUT_JSON, "w") as f: json.dump(results, f, indent=1)
            done += 1
            print(f"  [{done}/{total}] switch={sw:>2} ord {k+1}: "
                  f"frozen={fr:.4f} adapted={ad:.4f} rec={(ad-fr)*100:+.2f}p "
                  f"(elapsed {(time.time()-t0)/60:.1f}m)")

    # ---- AGGREGATE PER SWITCH ----
    print(f"\n==== W-46 SELECTION TABLE (pooled over K={K}) ====")
    print(f"  pure-50 reference (clean grid): +{PURE50_W46:.2f}p\n")
    print(f"  {'switch':>6} {'mean':>8} {'std':>7} {'per-ordering (p)':>28}")
    agg = {}
    for sw in SWITCHES:
        recs = [results[f"s{sw}_k{k}"]["recovery"]*100 for k in range(K)]
        m, sd = float(np.mean(recs)), float(np.std(recs))
        agg[sw] = (m, sd)
        cells = " ".join(f"{r:+.2f}" for r in recs)
        print(f"  {sw:>6} {m:>+7.2f} {sd:>7.2f}   {cells}")

    # ---- DETERMINISTIC SELECTION (pre-registered rule) ----
    # highest pooled mean; ties within 0.05p -> earlier switch point.
    best_sw = None; best_m = -1e9
    for sw in SWITCHES:  # SWITCHES is ascending, so earlier wins ties naturally
        m, sd = agg[sw]
        if m > best_m + 0.05:
            best_sw, best_m = sw, m
    s_star = best_sw
    m_star, sd_star = agg[s_star]

    print(f"\n==== SELECTED s* = {s_star}  "
          f"(W-46 pooled {m_star:+.2f}p +- {sd_star:.2f}) ====")

    # ---- KILL RULE A (stability, on W-46) ----
    print(f"\n---- KILL RULE A (order-std ceiling {ORDER_STD_CEILING:.2f}p) ----")
    if sd_star > ORDER_STD_CEILING:
        print(f"  FAIL: s*={s_star} order-std {sd_star:.2f}p > "
              f"{ORDER_STD_CEILING:.2f}p.")
        print(f"  --> STOP. Do NOT run the W-47 confirmatory. Submit Option A.")
        print(f"  --> Record this failure in PREREGISTRATION_switchpoint.md.")
    else:
        print(f"  PASS: s*={s_star} order-std {sd_star:.2f}p <= "
              f"{ORDER_STD_CEILING:.2f}p.")
        print(f"  --> Proceed to W-47 confirmatory:")
        print(f"      (set PHASE1={s_star}, PHASE2={TOTAL-s_star} in script 09,")
        print(f"       or run 09 with a --switch {s_star} argument), K=5, 3 windows.")
        if s_star != 50:
            print(f"  NOTE: s* != 50 -> disable the k=0 anchor assertion in")
            print(f"        script 09 (no archived trajectory to match; this is")
            print(f"        expected, not a broken anchor).")

    print(f"\n  Record s*, {m_star:+.2f}p +- {sd_star:.2f}, and the A verdict")
    print(f"  in PREREGISTRATION_switchpoint.md BEFORE running W-47.")


if __name__ == "__main__":
    main()
