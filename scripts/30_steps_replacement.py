#!/usr/bin/env python3
r"""
30_steps_replacement.py -- is it momentum, or total statistic replacement?

Every run in this study fixes S=50 steps, so momentum m and the replaced
fraction rho(m,S) = 1 - (1-m)^S have moved together in all of them. The title
claim attributes the single-service collapse to momentum. This separates them
by varying S at fixed m, so that pairs with different m have matching rho.

  (m=0.05, S=50)  rho = 0.923   service intact   (already measured: -6.63)
  (m=0.07, S=35)  rho = 0.921   matched rho, higher m        <- new
  (m=0.07, S=50)  rho = 0.973   service collapsed (already measured: -21.54)
  (m=0.05, S=70)  rho = 0.972   matched rho, lower m         <- new

PRE-COMMITTED PREDICTIONS, recorded before the run:
  S-A  Replacement controls it. (0.05, 70) collapses, losing more than 15
       points on the service, and (0.07, 35) stays intact, losing less than 10.
       Then the operational parameter is total replacement, the recommendation
       becomes "bound rho", not "sweep momentum", and the title must say
       statistic replacement.
  S-B  Momentum controls it. (0.05, 70) stays intact and (0.07, 35) collapses.
       The title stands and the result is more surprising, because rho would
       then not be the controlling variable despite Eq. (4).
  S-C  Neither cleanly. Both new points land between 10 and 15, or both
       collapse, or both stay intact. Then the two are not separable with this
       design and the paper must say the attribution is unresolved and retitle
       to the neutral form.

  Decided now: S-A is the outcome I expect, and under S-A the manuscript's
  headline is rewritten rather than defended.

  Caveat recorded before the run: Eq. (4) is exact for a stationary input. In
  the filtered condition the affine parameters of earlier layers change during
  adaptation, so deeper layers see a drifting input and their rho is only
  approximate. The stats condition has no such drift, which is why it is run
  alongside.

WHAT IT MEASURES: per-class recall on the report week, so the service can be
tracked directly, plus the partition effects. Same adaptation code and seeds as
scripts 24 and 27.

Run:
    python scripts/30_steps_replacement.py --size S --K 3
    python scripts/30_steps_replacement.py --size S --report
Output: steps_replacement_progress.json, steps_replacement.json
"""
import argparse, copy, hashlib, json, os, sys, time, warnings
import numpy as np
from tta_guards import guarded_eval, assert_anchor

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
WINDOW     = 200
N_WINDOWS  = 3
N_CLASSES  = 103
WORST      = 66                      # the service tracked through the paper
BIG_DROP   = 10.0

LR, QUANT = 1e-3, 0.5                # frozen config, not retuned
PAIRS = [(0.07, 35), (0.05, 70)]     # the two new cells
REFERENCE = {(0.05, 50): -6.63, (0.07, 50): -21.54}   # from script 27
COND = "filtered"

PART_JSON = "class_partition.json"
PART_SHA  = "class_partition.sha256"
CKPT      = "steps_replacement_progress.json"
OUT_JSON  = "steps_replacement.json"
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


def rho(m, S):
    return 1.0 - (1.0 - m) ** S


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


def per_class(y, p):
    rec = np.full(N_CLASSES, np.nan); sup = np.zeros(N_CLASSES, dtype=np.int64)
    for c in range(N_CLASSES):
        m = (y == c); n = int(m.sum()); sup[c] = n
        if n:
            rec[c] = float((p[m] == c).mean())
    return rec, sup


def adapt(base_model, window, device, order, mom, steps):
    """Script 27's adapt() with the step count exposed."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = mom
            mod.requires_grad_(True)
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias is not None:   params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)
    for s in range(steps):
        lo, _ = fwd(m, window[order[s % len(order)]], device)
        ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def load_ck(p):
    return json.load(open(p)) if os.path.exists(p) else {"done": {}}


def save_ck(p, c):
    json.dump(c, open(p, "w"), indent=1)


def report(ck, pairs, K, una):
    done = ck["done"]
    print("\n==== SERVICE RECALL AGAINST (m, S) AT MATCHED REPLACEMENT ====")
    print(f"  {'m':>6} {'S':>4} {'rho':>7} {'service':>9} {'dU':>8} "
          f"{'flows<-10':>10}  source")
    rows = []
    allp = [(m, s) for (m, s) in list(REFERENCE) + list(pairs)]
    for (m, s) in sorted(set(allp), key=lambda t: (rho(*t), t[0])):
        if (m, s) in REFERENCE and f"w0_m{m}_S{s}_0" not in done:
            print(f"  {m:>6} {s:>4} {rho(m,s):7.3f} {REFERENCE[(m,s)]:+9.2f} "
                  f"{'':>8} {'':>10}  script 27")
            rows.append((m, s, rho(m, s), REFERENCE[(m, s)], None))
            continue
        v66, vdu = [], []
        fl = {k: 0.0 for k in range(K)}     # summed over windows, per seed,
        for w in range(N_WINDOWS):          # to match script 27's convention
            b = done.get(f"w{w}_frozen")
            if not b:
                continue
            br = np.array(b["recall"], float); sw = np.array(b["support"], float)
            idx = [c for c in una if sw[c] > 0]
            for k in range(K):
                u = done.get(f"w{w}_m{m}_S{s}_{k}")
                if not u:
                    continue
                r = np.array(u["recall"], float); d = (r - br) * 100
                v66.append(d[WORST])
                vdu.append((np.sum(r[idx] * sw[idx]) / np.sum(sw[idx])
                            - np.sum(br[idx] * sw[idx]) / np.sum(sw[idx])) * 100)
                fl[k] += sum(sw[c] for c in idx
                             if not np.isnan(d[c]) and d[c] < -BIG_DROP)
        if not v66:
            continue
        print(f"  {m:>6} {s:>4} {rho(m,s):7.3f} {np.mean(v66):+9.2f} "
              f"{np.mean(vdu):+8.2f} "
              f"{int(np.mean(list(fl.values()))):>10,}  this run")
        rows.append((m, s, rho(m, s), float(np.mean(v66)), float(np.mean(vdu))))

    print("\n==== PRE-COMMITTED PREDICTIONS ====")
    got = {(m, s): v for m, s, _, v, _ in rows}
    a, b = got.get((0.05, 70)), got.get((0.07, 35))
    if a is None or b is None:
        print("  both new cells are needed before S-A to S-C can be read")
        return
    print(f"  (0.05, 70), rho {rho(0.05,70):.3f}: service {a:+.2f}p")
    print(f"  (0.07, 35), rho {rho(0.07,35):.3f}: service {b:+.2f}p")
    if a < -15 and b > -10:
        print("  S-A HOLDS. Total statistic replacement controls the collapse,")
        print("  not momentum. Rewrite the headline in terms of rho, change the")
        print("  operator recommendation to bounding rho, and retitle.")
    elif a > -10 and b < -15:
        print("  S-B HOLDS. Momentum controls it and rho does not, which Eq. (4)")
        print("  does not predict. The title stands and this becomes a finding")
        print("  in its own right.")
    else:
        print("  S-C. The two are not separable with this design. State the")
        print("  attribution as unresolved and use the neutral title.")
    json.dump({"rows": [{"m": m, "S": s, "rho": r, "service": v,
                         "delta_unaffected": u} for m, s, r, v, u in rows]},
              open(OUT_JSON, "w"), indent=1)
    print(f"\n  raw: {CKPT}   summary: {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--pairs", default="", help="e.g. 0.07:35,0.05:70")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    pairs = PAIRS
    if a.pairs.strip():
        pairs = []
        for tok in a.pairs.split(","):
            if not tok.strip():
                continue
            mo, st = tok.split(":")
            pairs.append((float(mo), int(st)))

    for f in (PART_JSON, PART_SHA):
        if not os.path.isfile(_resolve(f)):
            sys.exit(f"[STOP] {f} not found (searched {', '.join(_SEARCH)}).")
    rec = open(_resolve(PART_SHA)).read().split()[0].strip().lower()
    act = sha256_file(_resolve(PART_JSON))
    if rec != act:
        sys.exit(f"[STOP] {PART_JSON} does not match {PART_SHA}.")
    una = sorted(set(json.load(open(_resolve(PART_JSON)))["unaffected"]))

    ck = load_ck(_resolve(CKPT))
    if a.report:
        return report(ck, pairs, a.K, una)

    print("=== STEPS VERSUS MOMENTUM AT MATCHED REPLACEMENT (post-hoc) ===")
    print(f"    partition verified ({act[:16]}...)")
    print(f"    condition {COND}, lr={LR:.0e}, q={QUANT}")
    for (m, s) in pairs:
        print(f"    new cell m={m}, S={s}, rho={rho(m,s):.3f}")
    print("    predictions S-A to S-C are in this file's docstring\n")
    if ck["done"]:
        print(f"[RESUME] {len(ck['done'])} units done\n")

    model, loader, device = build(a.size, TEST_WEEK)
    t0 = time.time()
    for w in range(N_WINDOWS):
        win = collect(loader, w * WINDOW, WINDOW, label=f"W-47 w{w+1}")
        y, p = predict_on_batches(model, win, device)
        if w == 0:
            assert_anchor(float((y == p).mean()), ANCHOR_W47_W1, tol=0.0,
                          name="W-47 window 1 frozen (Table I anchor)")
            print(f"  [ANCHOR OK] window 1 frozen = {float((y==p).mean())!r}")
        r0, sup = per_class(y, p)
        ck["done"][f"w{w}_frozen"] = {"recall": r0.tolist(),
                                      "support": sup.tolist()}
        save_ck(_resolve(CKPT), ck)
        for (m, s) in pairs:
            for k in range(a.K):
                key = f"w{w}_m{m}_S{s}_{k}"
                if key in ck["done"]:
                    continue
                rng = np.random.default_rng(1000 * w + k)
                order = list(rng.permutation(len(win)))
                mm = adapt(model, win, device, order, m, s)
                y2, p2 = predict_on_batches(mm, win, device)
                r2, _ = per_class(y2, p2)
                ck["done"][key] = {"recall": r2.tolist()}
                save_ck(_resolve(CKPT), ck); del mm
                print(f"    m={m} S={s} k={k}: service "
                      f"{(r2[WORST]-r0[WORST])*100:+7.2f}p  "
                      f"({(time.time()-t0)/60:.1f}m)")
        del win
    report(ck, pairs, a.K, una)


if __name__ == "__main__":
    main()
