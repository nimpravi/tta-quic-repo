#!/usr/bin/env python3
r"""
25_prevalence_sweep.py -- measuring the break-even instead of deriving it.

The manuscript derives f* = 28.8% from two points and shows the value is
invariant under two candidate forms for how the effect scales with prevalence.
Invariance under respecification is not the same as being measured. A reviewer
was right that two points do not determine a functional form.

This resamples the report windows to a controlled affected-class prevalence and
finds where net recovery actually crosses zero.

HOW:
  A window's flows are pooled, split by the hash-locked partition, and resampled
  to a target prevalence f. Adaptation runs on the resampled pool and is
  evaluated on the same pool, because "net recovery" is the accuracy change on a
  population with that much drifted traffic in it. Both the adaptation batches
  and the evaluation population therefore carry prevalence f.

PRE-COMMITTED PREDICTIONS, recorded before the run:
  F-A  net recovery is negative at low prevalence and positive at high, and
       crosses zero exactly once on the grid.
  F-B  the crossing lies within 10 percentage points of the derived 28.8%.
  F-C  the sweep separates the two specifications two points could not.
       Constant Delta_A predicts net(f) linear in f with a non-zero intercept
       at f=0; proportional Delta_A predicts a curve through the origin. The
       script fits both and reports which residual is smaller.

  Kill rule. No crossing on the grid, or a crossing more than 15 points from
  28.8%, and equation (2) is an illustration rather than a prediction. The
  break-even paragraph and Fig. 1 would then be demoted, not defended.

TWO LIMITATIONS THAT MUST REACH THE MANUSCRIPT:
  1. Resampling to a target prevalence USES LABELS. This is a controlled test
     of the equation, not a procedure an operator could run. The text has to say
     so in the same breath as the result.
  2. Away from the window's native prevalence the smaller group is drawn with
     replacement. Which group gets duplicated depends on where the grid point
     sits relative to the window's native prevalence, so the duplication is
     worst at BOTH ends of the grid, not one. The factor for each group is
     recorded and printed for every point; heavy duplication makes the batch
     statistics depend on repeated flows, so report the factor alongside any
     quoted crossing and treat points above roughly 3x with suspicion.

Run:
    python scripts/25_prevalence_sweep.py --size S                 # the sweep
    python scripts/25_prevalence_sweep.py --size S --K 3 --windows 3 \
        --grid 0.25,0.288,0.35                                     # confirmation
Output: prevalence_sweep_progress.json (resumable per unit).
"""
import argparse, copy, hashlib, json, os, sys, time
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
WINDOW     = 200
FLOWS_PER_BATCH = 2048

# frozen episodic configuration, identical to script 18. Not retuned.
LR, STEPS, QUANT, BN_MOM = 1e-3, 50, 0.5, 0.1
COND = "filtered"                     # the paper's method
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.288, 0.35, 0.45, 0.60, 0.80]
F_STAR_DERIVED = 0.288                # what the manuscript predicts

PART_JSON = "class_partition.json"
PART_SHA  = "class_partition.sha256"
CKPT      = "prevalence_sweep_progress.json"
OUT_JSON  = "prevalence_sweep.json"
ANCHOR_W47_W1 = 0.72239013671875

_SEARCH = [".", "results/raw", "results"]


def _resolve(name):
    if os.path.isabs(name) or os.path.isfile(name):
        return name
    for d in _SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return name


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build(size, week):
    import torch
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import mm_cesnet_v2, MM_CESNET_V2_Weights
    w = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
    model = mm_cesnet_v2(weights=w, model_dir=MODEL_DIR)
    model.eval()
    tr = w.transforms
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


def split_batch(batch):
    """Return (ppi, flowstats, labels) from whatever shape the loader yields."""
    parts = list(batch) if isinstance(batch, (tuple, list)) else [batch]
    ppi = fs = y = None
    for p in parts:
        a = np.asarray(p)
        if a.ndim == 3:                                        ppi = a
        elif a.ndim == 2 and a.shape[1] > 0:                   fs = a
        elif a.ndim == 1 and np.issubdtype(a.dtype, np.integer): y = a
    if ppi is None or fs is None or y is None:
        raise RuntimeError("batch parse failed")
    return ppi, fs, y


def fwd(model, batch, device):
    import torch
    ppi, fs, y = split_batch(batch)
    return model((torch.as_tensor(ppi).float().to(device),
                  torch.as_tensor(fs).float().to(device))), y


def collect(loader, skip, n, label=""):
    out = []
    for i, b in enumerate(loader):
        if i < skip:
            continue
        if len(out) >= n:
            break
        out.append(b)
    if len(out) < n:
        print(f"  [note] {label}: {len(out)} batches, short of {n}")
    return out


def flatten(window):
    """Pool a window's batches into flat arrays so they can be resampled."""
    P, F, Y = [], [], []
    for b in window:
        ppi, fs, y = split_batch(b)
        P.append(ppi); F.append(fs); Y.append(y)
    return np.concatenate(P), np.concatenate(F), np.concatenate(Y)


def resample(P, F, Y, aff_idx, una_idx, f, n_batches, rng):
    """Build a pool of n_batches x FLOWS_PER_BATCH flows whose affected-class
    share is f. Returns the batches and the duplication factor of each group."""
    n_total = n_batches * FLOWS_PER_BATCH
    n_aff = int(round(f * n_total))
    n_una = n_total - n_aff
    dup_a = n_aff / max(len(aff_idx), 1)
    dup_u = n_una / max(len(una_idx), 1)
    pick_a = rng.choice(aff_idx, size=n_aff, replace=(n_aff > len(aff_idx)))
    pick_u = rng.choice(una_idx, size=n_una, replace=(n_una > len(una_idx)))
    idx = np.concatenate([pick_a, pick_u])
    rng.shuffle(idx)
    batches = [(P[idx[i:i + FLOWS_PER_BATCH]],
                F[idx[i:i + FLOWS_PER_BATCH]],
                Y[idx[i:i + FLOWS_PER_BATCH]])
               for i in range(0, n_total, FLOWS_PER_BATCH)]
    return batches, float(dup_a), float(dup_u)


def predict_on_batches(model, batches, device):
    import torch
    def _run():
        was = {n: mod.training for n, mod in model.named_modules()}
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for b in batches:
                lo, y = fwd(model, b, device)
                ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
        for n, mod in model.named_modules():
            mod.train(was[n])
        return np.concatenate(ys), np.concatenate(ps)
    return guarded_eval(model, _run)


def adapt(base_model, window, device, order):
    """Identical to script 18's adapt() for the filtered condition."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = BN_MOM
            mod.requires_grad_(True)
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias is not None:   params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    for s in range(STEPS):
        lo, _ = fwd(m, window[order[s % len(order)]], device)
        ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def load_ck(p):
    return json.load(open(p)) if os.path.exists(p) else {"done": {}, "native": {}}


def save_ck(p, c):
    json.dump(c, open(p, "w"), indent=1)


def crossing(fs, nets):
    """First zero crossing by linear interpolation. None if there is none."""
    out = []
    for i in range(len(fs) - 1):
        a, b = nets[i], nets[i + 1]
        if a == 0.0:
            out.append(fs[i])
        elif a * b < 0:
            out.append(fs[i] + (fs[i + 1] - fs[i]) * (-a) / (b - a))
    return out


def report(ck, grid, K, n_windows):
    done = ck["done"]
    print("\n==== E1: NET RECOVERY VERSUS DRIFT PREVALENCE ====")
    print(f"  {'f':>7} {'net':>9} {'dA':>9} {'dU':>9} {'dup_aff':>8} "
          f"{'dup_una':>8} {'n':>3}")
    fs, nets, das, dus = [], [], [], []
    for f in grid:
        nt, da, du, qa, qu = [], [], [], [], []
        for w in range(n_windows):
            for k in range(K):
                v = done.get(f"w{w}_f{f}_{k}")
                if not v:
                    continue
                nt.append(v["net"]); da.append(v["dA"]); du.append(v["dU"])
                qa.append(v["dup_aff"]); qu.append(v["dup_una"])
        if not nt:
            continue
        fs.append(f); nets.append(np.mean(nt) * 100)
        das.append(np.mean(da) * 100); dus.append(np.mean(du) * 100)
        print(f"  {f:7.3f} {np.mean(nt)*100:+9.2f} {np.mean(da)*100:+9.2f} "
              f"{np.mean(du)*100:+9.2f} {np.mean(qa):8.2f} {np.mean(qu):8.2f} "
              f"{len(nt):>3}")

    if len(fs) < 3:
        print("\n  not enough grid points yet for the predictions")
        return

    print("\n==== PRE-COMMITTED PREDICTIONS ====")
    xs = crossing(fs, nets)
    if len(xs) == 1:
        x = xs[0]
        print(f"  F-A single zero crossing at f = {x*100:.1f}%  [HOLDS]")
        d = abs(x - F_STAR_DERIVED) * 100
        v = ("HOLDS" if d <= 10 else
             "FAILS, demote the break-even to an illustration" if d > 15
             else "MARGINAL, between 10 and 15 points")
        print(f"  F-B distance from the derived {F_STAR_DERIVED*100:.1f}%: "
              f"{d:.1f} points  [{v}]")
    elif not xs:
        print(f"  F-A no zero crossing on [{min(fs)}, {max(fs)}]  "
              f"[FAILS, kill rule fires]")
        print(f"      net ranges {min(nets):+.2f} to {max(nets):+.2f}; "
              f"equation (2) is not predictive on this grid.")
    else:
        print(f"  F-A {len(xs)} crossings at "
              f"{', '.join(f'{x*100:.1f}%' for x in xs)}  [FAILS, not monotone]")

    # F-C: constant Delta_A vs proportional Delta_A
    A = np.array(das); U = np.array(dus); X = np.array(fs)
    c1 = np.polyfit(X, A, 0)            # constant
    r1 = float(np.sum((A - np.polyval(c1, X)) ** 2))
    g = float(np.sum(A * X) / np.sum(X * X)) if np.sum(X * X) > 0 else 0.0
    r2 = float(np.sum((A - g * X) ** 2))
    print(f"  F-C Delta_A across the grid: constant fit {c1[0]:+.2f} "
          f"(SSE {r1:.2f}) vs proportional fit {g:+.2f}f (SSE {r2:.2f})")
    print(f"      the data prefer the "
          f"{'CONSTANT' if r1 < r2 else 'PROPORTIONAL'} specification")
    print("      Fig. 1 plots both; this is the first evidence that separates")
    print("      them, and the caption should name the winner rather than")
    print("      presenting the pair as equally supported.")

    cells = [d for d in done.values()
             if isinstance(d, dict) and "dup_aff" in d]
    ha = max((d["dup_aff"] for d in cells), default=0.0)
    hu = max((d["dup_una"] for d in cells), default=0.0)
    nat = ck.get("native", {})
    print(f"\n  [CAVEAT] worst duplication on the grid: affected {ha:.2f}x, "
          f"unaffected {hu:.2f}x")
    if nat:
        print(f"  native prevalence per window: "
              + ", ".join(f"{k} {v*100:.2f}%" for k, v in sorted(nat.items())))
        print("  Duplication is worst at whichever end of the grid is furthest")
        print("  from these. Points above roughly 3x should be read as")
        print("  indicative rather than measured.")
    if len(fs) >= 2 and xs:
        near = min(cells, key=lambda d: abs(d["f_target"] - xs[0]))
        print(f"  at the crossing ({xs[0]*100:.1f}%) the duplication was "
              f"{near['dup_aff']:.2f}x / {near['dup_una']:.2f}x, which is the "
              f"number that matters for F-B")
    print("  Resampling uses labels. This measures the equation, not an")
    print("  operator procedure, and the manuscript must say so where it")
    print("  quotes the crossing.")
    json.dump({"grid": fs, "net": nets, "dA": das, "dU": dus,
               "crossings": xs, "f_star_derived": F_STAR_DERIVED},
              open(OUT_JSON, "w"), indent=1)
    print(f"\n  raw: {CKPT}   summary: {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--grid", default="", help="comma separated override")
    ap.add_argument("--batches", type=int, default=WINDOW,
                    help="batches per resampled pool")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    grid = [float(x) for x in a.grid.split(",") if x.strip()] or GRID

    for f in (PART_JSON, PART_SHA):
        if not os.path.isfile(_resolve(f)):
            sys.exit(f"[STOP] {f} not found (searched {', '.join(_SEARCH)}).")
    recorded = open(_resolve(PART_SHA)).read().split()[0].strip().lower()
    actual = sha256_file(_resolve(PART_JSON))
    if recorded != actual:
        sys.exit(f"[STOP] {PART_JSON} does not match {PART_SHA}.\n"
                 f"  recorded {recorded}\n  actual   {actual}")
    part = json.load(open(_resolve(PART_JSON)))
    aff, una = set(part["affected"]), set(part["unaffected"])

    ck = load_ck(_resolve(CKPT))
    if a.report:
        return report(ck, grid, a.K, a.windows)

    print("=== PREVALENCE SWEEP (declared post-hoc) ===")
    print(f"    partition verified against {PART_SHA} ({actual[:16]}...)")
    print(f"    condition {COND}, lr={LR:.0e}, steps={STEPS}, q={QUANT}, "
          f"m={BN_MOM}. Not retuned.")
    print(f"    grid {grid}")
    print(f"    K={a.K}, windows={a.windows}, {a.batches} batches per pool")
    print("    predictions F-A to F-C are in this file's docstring")
    print("    RESAMPLING USES LABELS: this measures the equation, not an")
    print("    operator procedure\n")
    if ck["done"]:
        print(f"[RESUME] {len(ck['done'])} units done\n")

    model, loader, device = build(a.size, TEST_WEEK)
    t0 = time.time()
    for w in range(a.windows):
        win = collect(loader, w * WINDOW, WINDOW, label=f"W-47 w{w+1}")
        y, p = predict_on_batches(model, win, device)
        if w == 0:
            assert_anchor(float((y == p).mean()), ANCHOR_W47_W1, tol=0.0,
                          name="W-47 window 1 frozen (Table I anchor)")
            print(f"  [ANCHOR OK] window 1 frozen = {float((y==p).mean())!r}")
        P, F, Y = flatten(win)
        del win
        aff_idx = np.flatnonzero(np.isin(Y, list(aff)))
        una_idx = np.flatnonzero(np.isin(Y, list(una)))
        native = len(aff_idx) / max(len(aff_idx) + len(una_idx), 1)
        ck["native"][f"w{w}"] = float(native); save_ck(_resolve(CKPT), ck)
        print(f"  window {w+1}: {len(aff_idx):,} affected, "
              f"{len(una_idx):,} unaffected, native prevalence "
              f"{native*100:.2f}%")
        for f in grid:
            for k in range(a.K):
                key = f"w{w}_f{f}_{k}"
                if key in ck["done"]:
                    continue
                rng = np.random.default_rng(10000 * w + 100 * k
                                            + int(round(f * 1000)))
                pool, dup_a, dup_u = resample(P, F, Y, aff_idx, una_idx, f,
                                              a.batches, rng)
                yb, pb = predict_on_batches(model, pool, device)
                mab = np.isin(yb, list(aff)); mub = np.isin(yb, list(una))
                b_net = float((yb == pb).mean())
                b_a = float((yb[mab] == pb[mab]).mean()) if mab.any() else float("nan")
                b_u = float((yb[mub] == pb[mub]).mean()) if mub.any() else float("nan")
                order = list(rng.permutation(len(pool)))
                m = adapt(model, pool, device, order)
                y2, p2 = predict_on_batches(m, pool, device)
                a_net = float((y2 == p2).mean())
                a_a = float((y2[mab] == p2[mab]).mean()) if mab.any() else float("nan")
                a_u = float((y2[mub] == p2[mub]).mean()) if mub.any() else float("nan")
                ck["done"][key] = {
                    "f_target": f,
                    "f_actual": float(mab.mean()),
                    "net": a_net - b_net, "dA": a_a - b_a, "dU": a_u - b_u,
                    "frozen_net": b_net, "dup_aff": dup_a, "dup_una": dup_u}
                save_ck(_resolve(CKPT), ck)
                del m, pool
                v = ck["done"][key]
                print(f"    f={f:<6} k={k}: net {v['net']*100:+6.2f}p  "
                      f"dA {v['dA']*100:+6.2f}p  dU {v['dU']*100:+6.2f}p  "
                      f"dup {dup_a:.2f}x/{dup_u:.2f}x  "
                      f"({(time.time()-t0)/60:.1f}m)")
        del P, F, Y
    report(ck, grid, a.K, a.windows)


if __name__ == "__main__":
    main()
