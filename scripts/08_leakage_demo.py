#!/usr/bin/env python3
r"""
08_leakage_demo.py -- falsification test for the checkpoint-leakage explanation.

CLAIM UNDER TEST:
  The archived headline (+4.34p, K=8, steps=100) differs from the clean
  final-eval numbers because of the evaluation rule alone: the pre-repo
  script selected best-over-checkpoints using test labels
  (best = max(frozen, acc@50, acc@100)), while the repo script evaluates
  once at step 100.

WHAT THIS SCRIPT DOES:
  Runs the OLD adaptation code path (lr=1e-3, steps=100, quant=0.5 -- the
  old frozen config) once per window at the k=0 seeded ordering, and
  reports BOTH evaluation rules computed from the SAME trajectory:
    best  = max(frozen, acc@step50, acc@step100)   [old rule, leaky]
    final = acc@step100                            [clean rule]
  No tuning stage; the old config is hardcoded. K=1 (k=0 only).

PREDICTIONS (fixed before running):
  final : should reproduce the steps=100 k=0 collapse-check deltas
          within a couple hundredths: +3.35 / +2.83 / +2.76.
  best  : should land in the archived K=8 per-window families:
          window 1 in [+4.36, +5.10] (archived mean +4.74),
          windows 2,3 near +4.12 / +4.15 (allow ±0.5, single ordering
          vs K=8 mean).
  If BOTH hold: the discrepancy is fully explained by the evaluation
  rule, demonstrated in the current environment on current data. Archive
  this output next to the correction note.
  If final matches but best does NOT reach the archived family: the
  evaluation rule explains only part of the gap; stop and investigate
  before drafting the self-audit paragraph.

RUNTIME: ~13 min/window (100 steps + 3 full-window evals), ~40 min total.

Run (from repo root, inside tent-env):
    python -u 08_leakage_demo.py --size S > leakage_demo_run.txt 2>&1
"""
import argparse, copy, sys, os, json, time
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
OUT_JSON   = "leakage_demo.json"

TARGET_TEST_EVAL_BATCHES = 200
MIN_TEST_EVAL_BATCHES    = 60
REPEATS    = 3

# OLD frozen config (as selected by the leaky tuning)
LR, STEPS, QUANT = 1e-3, 100, 0.5

EXPECT_FINAL = [3.35, 2.83, 2.76]          # steps=100 k=0, first collapse check
EXPECT_BEST  = [(4.36, 5.10), (3.6, 4.6), (3.6, 4.7)]   # archived families


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


def tent_both_rules(base_model, window_batches, device, lr, steps, quantile, order):
    """OLD adaptation code path (verbatim loop), instrumented to report BOTH
    evaluation rules from the same trajectory:
      best  = max(frozen, acc@50, acc@100)  [old, leaky]
      final = acc@100                       [clean]
    """
    import torch, torch.nn as nn, torch.nn.functional as F
    frozen = accuracy_on_batches(base_model, window_batches, device)
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum = 0.1
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias   is not None: params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=lr)
    best = frozen
    checkpoint_accs = {}
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
        if (s+1) % 50 == 0:
            a = accuracy_on_batches(m, window_batches, device)
            checkpoint_accs[s+1] = a
            best = max(best, a)
    final = checkpoint_accs[steps]
    return frozen, best, final, checkpoint_accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    args = ap.parse_args()

    print(f"=== LEAKAGE DEMO on {TEST_WEEK}: old rule (best-over-ckpts) vs clean "
          f"rule (final), SAME trajectory ===")
    print(f"    old config: lr={LR:.0e} steps={STEPS} quant={QUANT}, k=0 orderings\n")
    tmodel, tloader, device = build(args.size, TEST_WEEK)
    print(f"device={device}")
    probe_cap = TARGET_TEST_EVAL_BATCHES*REPEATS + 50
    n_avail = count_available_batches(tloader, cap=probe_cap)
    neb = min(TARGET_TEST_EVAL_BATCHES, n_avail // REPEATS)
    if neb < MIN_TEST_EVAL_BATCHES:
        print(f"  [STOP] only {neb}/window."); sys.exit(1)
    print(f"n_avail={n_avail}, using {neb}/window\n")

    tmodel, tloader, device = build(args.size, TEST_WEEK)
    windows = [collect_window(tloader, skip=r*neb, n=neb) for r in range(REPEATS)]

    results = {}
    t0 = time.time()
    print(f"{'win':>4} {'frozen':>8} {'acc@50':>8} {'acc@100':>8} | "
          f"{'BEST rec':>9} {'FINAL rec':>9} {'inflation':>9}")
    print("-" * 66)
    for r, w in enumerate(windows):
        rng = np.random.default_rng(1000*r + 0)
        order = list(rng.permutation(len(w)))
        fr, best, final, ck = tent_both_rules(tmodel, w, device, LR, STEPS, QUANT, order)
        brec, frec = (best-fr)*100, (final-fr)*100
        results[str(r)] = {"frozen": fr, "best": best, "final": final,
                           "ckpt_accs": ck, "best_rec_p": brec, "final_rec_p": frec}
        with open(OUT_JSON, "w") as f: json.dump(results, f, indent=1)
        print(f"{r+1:>4} {fr:>8.4f} {ck[50]:>8.4f} {ck[100]:>8.4f} | "
              f"{brec:>+8.2f}p {frec:>+8.2f}p {brec-frec:>+8.2f}p"
              f"   (elapsed {(time.time()-t0)/60:.1f}m)")

    print("\n==== VERDICT (rules fixed in docstring before running) ====")
    finals = [results[str(r)]["final_rec_p"] for r in range(REPEATS)]
    bests  = [results[str(r)]["best_rec_p"]  for r in range(REPEATS)]
    final_ok = all(abs(f-e) <= 0.05 for f, e in zip(finals, EXPECT_FINAL))
    best_ok  = all(lo <= b <= hi for b, (lo, hi) in zip(bests, EXPECT_BEST))
    print(f"  FINAL rule: {', '.join(f'{f:+.2f}' for f in finals)}  "
          f"(expected {', '.join(f'{e:+.2f}' for e in EXPECT_FINAL)})"
          f"  -> {'MATCH' if final_ok else 'MISMATCH'}")
    print(f"  BEST  rule: {', '.join(f'{b:+.2f}' for b in bests)}  "
          f"(expected in archived families)"
          f"  -> {'MATCH' if best_ok else 'MISMATCH'}")
    if final_ok and best_ok:
        print("\n  Both rules reproduce their respective histories from ONE trajectory.")
        print("  The archive/clean discrepancy is FULLY explained by the evaluation")
        print("  rule (test-label checkpoint selection). Archive this log next to")
        print("  the correction note; cite it in the self-audit paragraph.")
    elif final_ok and not best_ok:
        print("\n  Clean rule reproduces, but best-rule does not reach the archived")
        print("  family. The evaluation rule explains only part of the gap; STOP")
        print("  and investigate before drafting the self-audit paragraph.")
    else:
        print("\n  FINAL rule does not reproduce the collapse-check values; something")
        print("  is inconsistent in the pipeline. STOP and investigate with this log.")


if __name__ == "__main__":
    main()
