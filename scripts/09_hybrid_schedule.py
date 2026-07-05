#!/usr/bin/env python3
r"""
09_hybrid_schedule.py -- DELIBERATE two-phase TENT (the reconstructed hybrid).

WHAT THIS IS:
  The forensic investigation established that the archived headline
  was produced by an ACCIDENTAL two-phase algorithm: a state-mutating
  accuracy probe at step 50 called model.eval() and never restored train
  mode, so steps 51-100 ran affine-only entropy minimization on FROZEN
  BatchNorm statistics. This script reimplements that schedule EXPLICITLY
  and measures it under the standard protocol, so the paper can report it
  as a clearly-labeled post-hoc observation with clean provenance.

ALGORITHM (explicit, documented):
  Phase 1 (steps 1-50):  standard filtered TENT. BN modules in train mode
                         (momentum 0.1, running stats updating), entropy
                         minimization on BN affine params, quantile 0.5.
  Switch (after step 50): m.eval() on the whole model -- BN statistics
                         frozen from here on. (This exactly reproduces the
                         accidental code path, minus the probe itself,
                         which performed no parameter/stat updates.)
  Phase 2 (steps 51-100): continue filtered entropy minimization on the
                         same affine params, frozen statistics.

PROVENANCE ANCHORS (fixed before running):
  The k=0 (ord 1) units must reproduce the leakage-demo trajectories to
  printed precision, since the parameter/statistic trajectory is identical:
    window 1: adapted 0.7697 (rec +4.73p)
    window 2: adapted 0.7644 (rec +4.01p)
    window 3: adapted 0.7765 (rec +3.71p)
  A mismatch beyond ±0.01p at k=0 means the reimplementation does NOT
  match the accidental schedule; STOP and report.

FRAMING RULES (for the manuscript, decided in advance):
  - Post-hoc observation only. The switch point (50) was never tuned; it
    is an artifact of the probe interval. No claim that it is optimal.
  - Seeded-ordering error bars only; the natural-order outlier (+5.62,
    window 1) is out of scope and stays out of the paper.
  - Flag switch-point tuning and the ordering interaction as future work.

Run (from repo root, inside tent-env):
    python -u 09_hybrid_schedule.py --size S --K 3 > hybrid_steps50_run.txt 2>&1
Resumable via hybrid_progress.json.  ~11 min/unit, 9 units, ~1.7 h.
"""
import argparse, copy, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
CKPT       = "hybrid_progress.json"

TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3
DEFAULT_K  = 3

LR          = 1e-3
PHASE1      = 50        # standard TENT steps (stats updating)
PHASE2      = 50        # affine-only steps (stats frozen)
QUANT       = 0.5
BN_MOMENTUM = 0.1

# k=0 provenance anchors from leakage_demo.json (same trajectory)
EXPECT_K0 = [4.73, 4.01, 3.71]

# Clean references for the comparison printout
PURE50_POOLED  = (3.06, 0.27)   # K=5 final-eval headline (pure TENT, steps=50)
BNSTATS_POOLED = (2.43, 0.15)   # stats-only matched control


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


def hybrid_tent(base_model, window_batches, device, order):
    """Two-phase schedule, explicit. Eval once at the end (final rule)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = BN_MOMENTUM
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)

    def step(s):
        b = window_batches[order[s % len(order)]]
        lo, _ = fwd(m, b, device)
        ent = -(F.softmax(lo,1) * F.log_softmax(lo,1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()

    for s in range(PHASE1):
        step(s)
    m.eval()   # EXPLICIT switch: BN statistics frozen; affine grads continue
    for s in range(PHASE1, PHASE1 + PHASE2):
        step(s)

    final = accuracy_on_batches(m, window_batches, device)
    return frozen, final


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f: return json.load(f)
    return {"done": {}}


def save_ckpt(d):
    with open(CKPT, "w") as f: json.dump(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    ckpt = load_ckpt()
    if ckpt["done"]:
        print(f"[RESUME] {len(ckpt['done'])} completed orderings in {CKPT}.\n")

    print(f"=== HYBRID SCHEDULE on {TEST_WEEK}: {PHASE1} TENT steps + "
          f"{PHASE2} affine-only steps (stats frozen), lr={LR:.0e} quant={QUANT} ===")
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

    total = REPEATS*args.K
    print(f"\n=== {args.K} orderings x {REPEATS} windows = {total} units ===\n")
    t0 = time.time(); done = 0
    anchor_fail = False
    for r, w in enumerate(windows):
        for k in range(args.K):
            key = f"{r}_{k}"
            if key in ckpt["done"]:
                done += 1; continue
            rng = np.random.default_rng(1000*r + k)
            order = list(rng.permutation(len(w)))
            fr, ad = hybrid_tent(tmodel, w, device, order)
            rec = (ad-fr)*100
            ckpt["done"][key] = {"frozen": fr, "recovery": ad-fr}
            save_ckpt(ckpt); done += 1
            note = ""
            if k == 0:
                if abs(rec - EXPECT_K0[r]) <= 0.01:
                    note = "  [k=0 ANCHOR: MATCH]"
                else:
                    note = f"  [k=0 ANCHOR: MISMATCH, expected {EXPECT_K0[r]:+.2f}p -- STOP]"
                    anchor_fail = True
            print(f"  [{done}/{total}] window {r+1} ord {k+1}: frozen={fr:.4f} "
                  f"adapted={ad:.4f} rec={rec:+.2f}p{note} "
                  f"(elapsed {(time.time()-t0)/60:.1f}m)")
            if anchor_fail:
                print("\n  k=0 anchor failed: the explicit schedule does not reproduce")
                print("  the accidental trajectory. Do not continue; investigate with this log.")
                sys.exit(1)

    # ---- AGGREGATE ----
    per_window = {r: [] for r in range(REPEATS)}
    bases = {}
    for key, v in ckpt["done"].items():
        r = int(key.split("_")[0]); per_window[r].append(v["recovery"]); bases[r]=v["frozen"]
    all_recs = [v["recovery"] for v in ckpt["done"].values()]
    pw_mean = [float(np.mean(per_window[r])) for r in range(REPEATS)]
    pw_std  = [float(np.std(per_window[r]))  for r in range(REPEATS)]
    pm, ps  = float(np.mean(all_recs)), float(np.std(all_recs))

    print(f"\n==== HYBRID RESULT (S, n={neb}/window, K={args.K}) ====")
    print(f"  hybrid recovery: {pm*100:+.2f}p ± {ps*100:.2f}")
    print(f"  per-window means: {', '.join(f'{m*100:+.2f}' for m in pw_mean)}")
    print(f"  mean within-window order-std = {float(np.mean(pw_std))*100:.2f}p")
    print(f"\n  References (same protocol, same environment):")
    print(f"    pure TENT steps=50 : {PURE50_POOLED[0]:+.2f}p ± {PURE50_POOLED[1]:.2f}")
    print(f"    BN-stats only      : {BNSTATS_POOLED[0]:+.2f}p ± {BNSTATS_POOLED[1]:.2f}")
    print(f"    hybrid (this run)  : {pm*100:+.2f}p ± {ps*100:.2f}")
    print(f"    hybrid lift over pure-50 = {pm*100 - PURE50_POOLED[0]:+.2f}p")
    print(f"\n  Manuscript framing: post-hoc observation, switch point untuned,")
    print(f"  seeded orderings only. Done. New run: delete {CKPT} first.")


if __name__ == "__main__":
    main()
