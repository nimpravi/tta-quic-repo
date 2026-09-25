#!/usr/bin/env python3
r"""
24_momentum_displacement.py -- what moving the statistics actually does.

Combines the two experiments the review asked for separately. Sweeping BN
momentum produces a family of adapted models whose running statistics have
moved by varying amounts, so measuring displacement alongside the partition
effects at each momentum answers both in one run:

  E5  a controlled demonstration of Section IV-A instead of an interpretation
  E3  whether BN displacement, which needs no labels, predicts the harm

WHY THE DESIGN SEPARATES THE TWO MECHANISMS:
  Prediction runs in eval mode, so it normalizes with running statistics.
  Momentum therefore controls exactly how far those statistics move, and
  nothing else. Two corners of the grid are diagnostic on their own:

    stats-only at m=0    running statistics never update and no gradients are
                         taken, so the returned model IS the frozen model.
                         Must give 0.00 / 0.00 and displacement 0.
    filtered at m=0      affine gradients are still taken, computed through
                         batch-statistic normalization, while the running
                         statistics stay put. The gradient term with the
                         displacement term switched off. Never measured before.

PRE-COMMITTED PREDICTIONS, recorded before the run:
  M-A  stats-only at m=0 gives exactly 0.00 on both partitions and zero
       displacement. If not, the harness is not measuring what it claims and
       nothing downstream is trustworthy. This is a stop condition.
  M-B  |Delta_U| grows monotonically with m for stats-only.
  M-C  Delta_A also grows with m, but proportionally less than |Delta_U| does
       between the smallest and the largest non-zero m.
  M-D  displacement predicts Delta_U across the whole grid and both
       conditions, Spearman below -0.8. This is the E3 claim.
  M-E  filtered at m=0 gives Delta_U near zero with Delta_A well above zero.

  Kill rules. M-A failing stops the run. M-B failing withdraws the displacement
  account in Section IV-A rather than softening it. M-D below |0.5| drops the
  early-warning framing; displacement is then reported as an unreliable
  monitor, which still supports the caution claim.

DISPLACEMENT METRIC (label-free, computable at inference):
  For each BN layer, the absolute shift of each channel's running mean from the
  source model's, standardized by that channel's source running standard
  deviation, averaged over channels and then over layers. Dimensionless, zero
  when the statistics have not moved, and it is the observable counterpart of
  the |mu_B - mu_s| = f*d term in (5).

Run:
    python scripts/24_momentum_displacement.py --size S --selftest   # 2 min
    python scripts/24_momentum_displacement.py --size S --K 3
    python scripts/24_momentum_displacement.py --size S --moms 0,0.1
Output: momentum_displacement_progress.json (resumable per unit).
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
N_WINDOWS  = 3

# frozen episodic configuration, identical to script 18. Not retuned.
LR, STEPS, QUANT = 1e-3, 50, 0.5
MOMS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]     # 0.1 is the paper's value
CONDS = ["stats", "filtered"]

PART_JSON = "class_partition.json"
PART_SHA  = "class_partition.sha256"
CKPT      = "momentum_displacement_progress.json"
OUT_JSON  = "momentum_displacement.json"

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


def fwd(model, batch, device):
    import torch
    parts = list(batch) if isinstance(batch, (tuple, list)) else [batch]
    ppi = fs = y = None
    for p in parts:
        a = np.asarray(p)
        if a.ndim == 3:                                        ppi = a
        elif a.ndim == 2 and a.shape[1] > 0:                   fs = a
        elif a.ndim == 1 and np.issubdtype(a.dtype, np.integer): y = a
    if ppi is None or fs is None or y is None:
        raise RuntimeError("batch parse failed")
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


def bn_stats(model):
    """Running mean and variance of every BN layer, as numpy."""
    import torch.nn as nn
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if mod.running_mean is None or mod.running_var is None:
                continue
            out[name] = (mod.running_mean.detach().cpu().numpy().copy(),
                         mod.running_var.detach().cpu().numpy().copy())
    return out


def displacement(src, adp, eps=1e-5):
    """Standardized shift of the running means, averaged over channels then
    over layers. Label-free. Zero exactly when the statistics did not move."""
    per = []
    for name, (mu_s, var_s) in src.items():
        if name not in adp:
            continue
        mu_a = adp[name][0]
        per.append(float(np.mean(np.abs(mu_a - mu_s) / np.sqrt(var_s + eps))))
    return (float(np.mean(per)) if per else float("nan")), per


def adapt(base_model, window, device, cond, order, mom):
    """Identical to script 18's adapt() except that BN momentum is a parameter
    instead of the fixed 0.1. Same steps, same lr, same quantile, same order."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = mom
            if cond == "filtered":
                mod.requires_grad_(True)
                if mod.weight is not None: params.append(mod.weight)
                if mod.bias is not None:   params.append(mod.bias)
    if cond == "stats":
        with torch.no_grad():
            for s in range(STEPS):
                fwd(m, window[order[s % len(order)]], device)
        return m
    opt = torch.optim.Adam(params, lr=LR)
    for s in range(STEPS):
        lo, _ = fwd(m, window[order[s % len(order)]], device)
        ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def preflight_momentum_zero(model, window, device, src):
    """Verify, on this machine and this torch version, that BN momentum 0.0
    freezes the running statistics.

    The whole design rests on it: prediction runs in eval mode, so momentum is
    the only thing controlling how far the running statistics move, and the
    m=0 stats-only cell is supposed to reproduce the frozen model exactly. If
    this torch build instead treats momentum 0.0 as a cumulative moving average
    (the documented behaviour of momentum=None), the m=0 cell silently becomes
    a full-window average and every row below it is mislabelled.

    Cheap: one adapt at m=0 and one prediction pass."""
    print("  [PREFLIGHT] checking that BN momentum 0.0 freezes running stats")
    rng = np.random.default_rng(0)
    order = list(rng.permutation(len(window)))
    m0 = adapt(model, window, device, "stats", order, 0.0)
    D, _ = displacement(src, bn_stats(m0))
    adp = bn_stats(m0)
    exact = all(np.array_equal(src[n][0], adp[n][0]) and
                np.array_equal(src[n][1], adp[n][1]) for n in src if n in adp)
    y0, p0 = predict_on_batches(model, window[:5], device)
    y1, p1 = predict_on_batches(m0, window[:5], device)
    same = np.array_equal(p0, p1)
    del m0
    if exact and same and D == 0.0:
        print(f"      OK: statistics bit-identical, predictions identical, "
              f"displacement {D:.1e}")
        return
    sys.exit(
        "[STOP] BN momentum 0.0 did NOT freeze the running statistics on this\n"
        f"  build. bit-identical stats: {exact}; identical predictions: {same};\n"
        f"  displacement: {D}.\n"
        "  Do not run the grid: the m=0 row would not be the frozen model and\n"
        "  prediction M-A could not be evaluated. Check whether this torch\n"
        "  version treats momentum 0.0 as cumulative averaging, and if so use a\n"
        "  small non-zero floor (for example 1e-8) as the bottom of the grid,\n"
        "  recording the change before rerunning.")


def spearman(x, y):
    def _midrank(a):
        a = np.asarray(a, float)
        o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[o[j + 1]] == a[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = _midrank(x), _midrank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def load_ck(p):
    return json.load(open(p)) if os.path.exists(p) else {"done": {}}


def save_ck(p, c):
    json.dump(c, open(p, "w"), indent=1)


def report(ck, moms, conds, K):
    done = ck["done"]
    print("\n==== E5: PARTITION EFFECT VERSUS BN MOMENTUM ====")
    print(f"  {'cond':>9} {'m':>6} {'disp':>9} {'dA':>9} {'dU':>9} {'dA/|dU|':>9}")
    rows = []
    for cond in conds:
        for m in moms:
            da, du, dp = [], [], []
            for w in range(N_WINDOWS):
                base = done.get(f"w{w}_frozen")
                if not base:
                    continue
                for k in range(K):
                    v = done.get(f"w{w}_{cond}_m{m}_{k}")
                    if not v:
                        continue
                    da.append(v["affected"] - base["affected"])
                    du.append(v["unaffected"] - base["unaffected"])
                    dp.append(v["displacement"])
            if not da:
                continue
            A, U, D = np.mean(da) * 100, np.mean(du) * 100, np.mean(dp)
            ratio = (A / abs(U)) if abs(U) > 1e-9 else float("inf")
            rows.append((cond, m, D, A, U, len(da)))
            print(f"  {cond:>9} {m:>6} {D:9.5f} {A:+9.2f} {U:+9.2f} "
                  + (f"{ratio:9.2f}" if np.isfinite(ratio) else f"{'--':>9}"))

    print("\n==== PRE-COMMITTED PREDICTIONS ====")
    s0 = [r for r in rows if r[0] == "stats" and r[1] == 0.0]
    if s0:
        _, _, D, A, U, n = s0[0]
        ok = abs(A) < 0.005 and abs(U) < 0.005 and abs(D) < 1e-9
        print(f"  M-A stats at m=0: dA {A:+.3f}  dU {U:+.3f}  disp {D:.2e}  "
              f"[{'HOLDS' if ok else 'FAILS, STOP'}]")
        if not ok:
            print("      The m=0 stats-only model must be the frozen model.")
            print("      A non-zero value here invalidates every other row.")
    sr = [r for r in rows if r[0] == "stats" and r[1] > 0]
    if len(sr) >= 3:
        us = [abs(r[4]) for r in sr]; as_ = [r[3] for r in sr]
        mono_u = all(us[i] <= us[i + 1] + 1e-9 for i in range(len(us) - 1))
        print(f"  M-B |dU| monotone in m (stats): {[f'{u:.2f}' for u in us]}  "
              f"[{'HOLDS' if mono_u else 'FAILS'}]")
        gu = us[-1] / us[0] if us[0] > 1e-9 else float("inf")
        ga = as_[-1] / as_[0] if as_[0] > 1e-9 else float("inf")
        print(f"  M-C growth from smallest to largest non-zero m: "
              f"dA x{ga:.2f}, |dU| x{gu:.2f}  "
              f"[{'HOLDS' if ga < gu else 'FAILS'}]")
    if len(rows) >= 5:
        rho = spearman([r[2] for r in rows], [r[4] for r in rows])
        verdict = ("HOLDS" if rho < -0.8 else
                   "WEAK, keep the caution framing" if rho < -0.5 else
                   "FAILS, drop the early-warning framing")
        print(f"  M-D displacement vs dU, Spearman {rho:+.3f} over "
              f"{len(rows)} cells  [{verdict}]")
    f0 = [r for r in rows if r[0] == "filtered" and r[1] == 0.0]
    if f0:
        _, _, D, A, U, n = f0[0]
        print(f"  M-E filtered at m=0: dA {A:+.2f}  dU {U:+.2f}  disp {D:.2e}  "
              f"[{'HOLDS' if (A > 0.5 and abs(U) < 0.5) else 'FAILS'}]")
        print("      This is the gradient term with no statistic movement.")
    json.dump({"rows": [{"cond": r[0], "momentum": r[1], "displacement": r[2],
                         "delta_affected": r[3], "delta_unaffected": r[4],
                         "n_units": r[5]} for r in rows]},
              open(OUT_JSON, "w"), indent=1)
    print(f"\n  raw: {CKPT}   summary: {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--moms", default="", help="comma separated override")
    ap.add_argument("--conds", default="", help="comma separated override")
    ap.add_argument("--report", action="store_true",
                    help="re-print from the checkpoint without running")
    ap.add_argument("--selftest", action="store_true",
                    help="run only the momentum-zero preflight and exit")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the momentum-zero check (not recommended)")
    a = ap.parse_args()

    moms = [float(x) for x in a.moms.split(",") if x.strip()] or MOMS
    conds = [c.strip() for c in a.conds.split(",") if c.strip()] or CONDS

    for f in (PART_JSON, PART_SHA):
        if not os.path.isfile(_resolve(f)):
            sys.exit(f"[STOP] {f} not found (searched {', '.join(_SEARCH)}).")
    recorded = open(_resolve(PART_SHA)).read().split()[0].strip().lower()
    actual = sha256_file(_resolve(PART_JSON))
    if recorded != actual:
        sys.exit(f"[STOP] {PART_JSON} does not match {PART_SHA}.\n"
                 f"  recorded {recorded}\n  actual   {actual}\n"
                 f"  The partition changed after it was committed.")
    part = json.load(open(_resolve(PART_JSON)))
    aff, una = set(part["affected"]), set(part["unaffected"])

    ck = load_ck(_resolve(CKPT))
    if a.report:
        return report(ck, moms, conds, a.K)

    print("=== BN MOMENTUM AND DISPLACEMENT (declared post-hoc) ===")
    print(f"    partition verified against {PART_SHA} ({actual[:16]}...)")
    print(f"    affected {len(aff)}, unaffected {len(una)}")
    print(f"    frozen config lr={LR:.0e}, steps={STEPS}, q={QUANT}, "
          f"momentum swept")
    print(f"    grid: {moms} x {conds} x K={a.K} x {N_WINDOWS} windows")
    print(f"    predictions M-A to M-E are recorded in this file's docstring\n")
    if ck["done"]:
        print(f"[RESUME] {len(ck['done'])} units done\n")

    model, loader, device = build(a.size, TEST_WEEK)
    src = bn_stats(model)
    t0 = time.time()
    for w in range(N_WINDOWS):
        win = collect(loader, w * WINDOW, WINDOW, label=f"W-47 w{w+1}")
        y, p = predict_on_batches(model, win, device)
        if w == 0:
            assert_anchor(float((y == p).mean()), ANCHOR_W47_W1, tol=0.0,
                          name="W-47 window 1 frozen (Table I anchor)")
            print(f"  [ANCHOR OK] window 1 frozen = {float((y==p).mean())!r}")
        if w == 0 and not a.skip_preflight:
            preflight_momentum_zero(model, win, device, src)
            if a.selftest:
                return
        ma, mu = np.isin(y, list(aff)), np.isin(y, list(una))
        base = {"affected": float((y[ma] == p[ma]).mean()),
                "unaffected": float((y[mu] == p[mu]).mean()),
                "n_affected": int(ma.sum()), "n_unaffected": int(mu.sum())}
        ck["done"][f"w{w}_frozen"] = base; save_ck(_resolve(CKPT), ck)
        print(f"  window {w+1} frozen: affected {base['affected']:.4f}, "
              f"unaffected {base['unaffected']:.4f}")
        for cond in conds:
            for mom in moms:
                for k in range(a.K):
                    key = f"w{w}_{cond}_m{mom}_{k}"
                    if key in ck["done"]:
                        continue
                    rng = np.random.default_rng(1000 * w + k)
                    order = list(rng.permutation(len(win)))
                    m = adapt(model, win, device, cond, order, mom)
                    D, per = displacement(src, bn_stats(m))
                    y2, p2 = predict_on_batches(m, win, device)
                    ck["done"][key] = {
                        "affected": float((y2[ma] == p2[ma]).mean()),
                        "unaffected": float((y2[mu] == p2[mu]).mean()),
                        "displacement": D,
                        "displacement_per_layer": per}
                    save_ck(_resolve(CKPT), ck); del m
                    v = ck["done"][key]
                    print(f"    {cond:>9} m={mom:<5} k={k}: "
                          f"dA {(v['affected']-base['affected'])*100:+6.2f}p  "
                          f"dU {(v['unaffected']-base['unaffected'])*100:+6.2f}p"
                          f"  disp {D:.5f}  ({(time.time()-t0)/60:.1f}m)")
        del win
    report(ck, moms, conds, a.K)


if __name__ == "__main__":
    main()
