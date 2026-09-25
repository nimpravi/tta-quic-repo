#!/usr/bin/env python3
r"""
27_perclass_momentum.py -- does the single-service collapse survive at m=0.03?

WHY THIS EXISTS. The momentum grid showed filtered adaptation at m=0.03 giving
Delta_U = -0.16 against -2.79 at the m=0.1 the study uses, with a slightly
larger gain on the affected classes. That makes the partition-level collateral
damage largely an artifact of the default momentum. But the paper's strongest
single fact is per-class, not partition-level: one undrifted service falling
from 0.972 to 0.755 on 52,866 flows while aggregate accuracy, macro-F1 and every
collapse diagnostic improved. The grid never looked at individual classes, so
that service's fate at m=0.03 is unknown, and the framing of the revision
depends on it.

PRE-COMMITTED PREDICTIONS, recorded before the run:
  C-A  the worst-hit undrifted class still loses more than 10 points at m=0.03.
       The harm concentrates rather than disappearing: a partition mean of
       -0.16 is consistent with a few classes falling hard and the rest
       improving slightly. If this holds, the caution claim survives and gets
       sharper, because the aggregate hides the damage even better at the
       better momentum.
  C-B  the number of undrifted classes losing more than 10 points at m=0.03 is
       lower than at m=0.1 but not zero.
  C-C  the correlation between a class's per-class change at m=0.1 and at
       m=0.03 is positive and strong, Spearman above +0.6. The same classes are
       hurt at both settings; momentum scales the damage rather than moving it.

  What each outcome means, decided now rather than after seeing the numbers:
    C-A holds   -> the revision keeps the per-class catastrophe as its centre
                   and adds momentum as the mechanism and the fix.
    C-A fails   -> the per-class catastrophe is itself a momentum artifact. The
                   paper's finding becomes "the shipped default is wrong", which
                   is a different paper: title, abstract and framing all change.
                   Report it, do not rescue it.
    C-C fails   -> momentum moves the damage between classes rather than
                   scaling it, which no part of Section IV-A anticipates and
                   which would need its own explanation before anything is
                   claimed.

WHAT IT DOES: evaluation passes only. For each of m in {0.1, 0.03} it adapts as
script 18 does, then records per-class recall on every one of the 102 known
services, alongside the frozen per-class recall. No new adaptation semantics,
no retuning, no new configuration.

Run:
    python scripts/27_perclass_momentum.py --size S --K 3
Output: perclass_momentum_progress.json, perclass_momentum.json
"""
import argparse, copy, hashlib, json, os, sys, time, warnings
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
WINDOW     = 200
N_WINDOWS  = 3
N_CLASSES  = 103                      # 102 known services plus _unknown

LR, STEPS, QUANT = 1e-3, 50, 0.5      # frozen episodic config, not retuned
MOMS  = [0.1, 0.03]
COND  = "filtered"
BIG_DROP = 10.0                       # points, the partition rule's threshold

PART_JSON = "class_partition.json"
PART_SHA  = "class_partition.sha256"
CKPT      = "perclass_momentum_progress.json"
OUT_JSON  = "perclass_momentum.json"
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
    return ds, model, ds.get_test_dataloader(), device


def class_names(ds):
    """Index -> service name. cesnet-datazoo exposes these on ds.class_info,
    which is what 18_nondrifted_control.py uses; the attribute probes below
    are kept as a fallback for other versions of the package."""
    try:
        names = [str(x) for x in ds.class_info.target_names]
        if len(names) == N_CLASSES:
            return names
        print(f"  [note] class_info gave {len(names)} names, expected "
              f"{N_CLASSES}; falling back")
    except Exception as e:
        print(f"  [note] ds.class_info unavailable ({e}); falling back")
    for attr in ("known_apps", "app_names", "classes", "labels"):
        v = getattr(ds, attr, None)
        if v is not None:
            try:
                return [str(x) for x in list(v)]
            except Exception:
                pass
    print("  [note] class names unavailable; using indices")
    return [f"class_{i}" for i in range(N_CLASSES)]


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
    """Recall and support for every label slot. NaN where support is zero."""
    rec = np.full(N_CLASSES, np.nan)
    sup = np.zeros(N_CLASSES, dtype=np.int64)
    for c in range(N_CLASSES):
        m = (y == c)
        n = int(m.sum())
        sup[c] = n
        if n:
            rec[c] = float((p[m] == c).mean())
    return rec, sup


def adapt(base_model, window, device, order, mom):
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = mom
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


def spearman(x, y):
    def _mr(a):
        a = np.asarray(a, float)
        o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float); i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[o[j + 1]] == a[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = _mr(x), _mr(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def load_ck(p):
    return json.load(open(p)) if os.path.exists(p) else {"done": {}}


def save_ck(p, c):
    json.dump(c, open(p, "w"), indent=1)


def report(ck, moms, K, names, una):
    done = ck["done"]
    # pooled per-class delta per momentum, averaged over windows and seeds
    delta = {}
    support = np.zeros(N_CLASSES)
    for m in moms:
        acc = []
        for w in range(N_WINDOWS):
            b = done.get(f"w{w}_frozen")
            if not b:
                continue
            support = np.maximum(support, np.array(b["support"], float))
            for k in range(K):
                v = done.get(f"w{w}_m{m}_{k}")
                if not v:
                    continue
                acc.append(np.array(v["recall"], float)
                           - np.array(b["recall"], float))
        if acc:
            # label slot 102 is the zero-support _unknown and is NaN in every
            # unit, so nanmean warns on an empty slice. Expected, not an error.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                delta[m] = np.nanmean(np.vstack(acc), axis=0) * 100

    # Per-window worst, which is the aggregation a worst-case claim needs.
    # Averaging a class's delta across windows before ranking dilutes a class
    # that collapses in one window and not another, so the pooled table below
    # understates the damage at every momentum. Both views are printed.
    print("\n==== WORST UNDRIFTED CLASS, PER WINDOW ====")
    print(f"  {'win':>4} {'m':>6} {'class':>12} {'support':>9} {'delta':>8} "
          f"{'#<-10':>6} {'flows in those':>15}")
    for w in range(N_WINDOWS):
        b = done.get(f"w{w}_frozen")
        if not b:
            continue
        br = np.array(b["recall"], float)
        sw = np.array(b["support"], float)
        for m in moms:
            ds = [np.array(done[f"w{w}_m{m}_{k}"]["recall"], float) - br
                  for k in range(K) if f"w{w}_m{m}_{k}" in done]
            if not ds:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d = np.nanmean(np.vstack(ds), axis=0) * 100
            cells = [(d[c], c) for c in sorted(una)
                     if sw[c] > 0 and not np.isnan(d[c])]
            if not cells:
                continue
            wv, wc = min(cells)
            big = [(v, c) for v, c in cells if v < -BIG_DROP]
            nm = names[wc] if wc < len(names) else f"class_{wc}"
            print(f"  {w:>4} {m:>6} {nm[:12]:>12} {sw[wc]:>9,.0f} {wv:+8.2f} "
                  f"{len(big):>6} {sum(sw[c] for _, c in big):>15,.0f}")
    print("  Read this table, not the pooled one, for any worst-case claim.")
    print("  Check the support column: a large loss on a few hundred flows is")
    print("  a handful of flows changing label, not a service failing.")

    # Momentum curve for whichever class is worst at the largest momentum, and
    # the exposed-flow count. This is what locates a threshold, if there is one.
    allm = sorted(delta)
    if len(allm) >= 3:
        ref = allm[-1]
        cand = [(delta[ref][c], c) for c in sorted(una)
                if not np.isnan(delta[ref][c])]
        _, wc = min(cand)
        nm = names[wc] if wc < len(names) else f"class_{wc}"
        deep = delta[ref][wc]
        print(f"\n==== MOMENTUM CURVE FOR {nm} ====")
        print(f"  {'m':>7} {'delta':>8} {'% of deepest':>13} {'flows<-10':>11}")
        for m in allm:
            tot = 0
            for w in range(N_WINDOWS):
                b = done.get(f"w{w}_frozen")
                if not b:
                    continue
                sw = np.array(b["support"], float)
                br = np.array(b["recall"], float)
                ds = [np.array(done[f"w{w}_m{m}_{k}"]["recall"], float) - br
                      for k in range(K) if f"w{w}_m{m}_{k}" in done]
                if not ds:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    d = np.nanmean(np.vstack(ds), axis=0) * 100
                tot += sum(sw[c] for c in sorted(una)
                           if sw[c] > 0 and not np.isnan(d[c]) and d[c] < -BIG_DROP)
            pct = 100 * delta[m][wc] / deep if deep != 0 else float("nan")
            print(f"  {m:>7} {delta[m][wc]:+8.2f} {pct:12.1f}% {int(tot):>11,}")
        print("  A percentage that jumps between two adjacent momenta is a")
        print("  threshold, not a slope, and the paper should say which it is.")

    if 0.1 not in delta or 0.03 not in delta:
        print("\n  C-A to C-C need both m=0.1 and m=0.03 in this run; skipped.")
        print("  The tables above are complete for the momenta that were run.")
        return

    d10, d03 = delta[0.1], delta[0.03]
    uset = sorted(una)
    print("\n==== PER-CLASS CHANGE ON THE UNDRIFTED PARTITION ====")
    print("  Averaged over windows and seeds. A class that collapses in one")
    print("  window only is diluted here; the per-window table above is the")
    print("  one to quote.")
    print(f"  {'class':<28} {'support':>9} {'m=0.1':>9} {'m=0.03':>9}")
    order = sorted(uset, key=lambda c: (d10[c] if not np.isnan(d10[c]) else 0))
    for c in order[:12]:
        nm = names[c] if c < len(names) else f"class_{c}"
        print(f"  {nm[:28]:<28} {support[c]:9,.0f} {d10[c]:+9.2f} {d03[c]:+9.2f}")

    worst = order[0]
    nm = names[worst] if worst < len(names) else f"class_{worst}"
    print("\n==== PRE-COMMITTED PREDICTIONS ====")
    print(f"  worst undrifted class at m=0.1: {nm} "
          f"({support[worst]:,.0f} flows)")
    print(f"    m=0.1  {d10[worst]:+.2f} points")
    print(f"    m=0.03 {d03[worst]:+.2f} points")
    ca = d03[worst] < -BIG_DROP
    print(f"  C-A worst undrifted class still loses more than {BIG_DROP:.0f} "
          f"points at m=0.03  [{'HOLDS' if ca else 'FAILS'}]")
    if ca:
        print("      The caution claim survives. The harm concentrates rather")
        print("      than disappearing, and the aggregate hides it better at")
        print("      the better momentum. Keep the per-class catastrophe as")
        print("      the centre of the revision.")
    else:
        print("      The per-class catastrophe is itself a momentum artifact.")
        print("      The finding becomes that the shipped default is wrong.")
        print("      That is a different paper: title, abstract and framing")
        print("      all change. Report it, do not rescue it.")

    n10 = int(sum(1 for c in uset if d10[c] < -BIG_DROP))
    n03 = int(sum(1 for c in uset if d03[c] < -BIG_DROP))
    cb = (n03 < n10) and n03 > 0
    print(f"  C-B undrifted classes losing more than {BIG_DROP:.0f} points: "
          f"{n10} at m=0.1, {n03} at m=0.03  "
          f"[{'HOLDS' if cb else 'FAILS'}]")

    pair = [(d10[c], d03[c]) for c in uset
            if not np.isnan(d10[c]) and not np.isnan(d03[c])]
    rho = spearman([a for a, _ in pair], [b for _, b in pair])
    print(f"  C-C Spearman between the two momenta over {len(pair)} undrifted "
          f"classes: {rho:+.3f}  [{'HOLDS' if rho > 0.6 else 'FAILS'}]")
    if rho <= 0.6:
        print("      Momentum moves the damage between classes rather than")
        print("      scaling it. Nothing in Section IV-A anticipates this and")
        print("      it needs its own explanation before anything is claimed.")

    json.dump({"delta_m0.1": d10.tolist(), "delta_m0.03": d03.tolist(),
               "support": support.tolist(), "unaffected": uset,
               "worst_class_index": int(worst), "worst_class_name": nm,
               "n_big_drop_m0.1": n10, "n_big_drop_m0.03": n03,
               "spearman_between_momenta": rho},
              open(OUT_JSON, "w"), indent=1)
    print(f"\n  raw: {CKPT}   summary: {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--moms", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    moms = [float(x) for x in a.moms.split(",") if x.strip()] or MOMS

    for f in (PART_JSON, PART_SHA):
        if not os.path.isfile(_resolve(f)):
            sys.exit(f"[STOP] {f} not found (searched {', '.join(_SEARCH)}).")
    recorded = open(_resolve(PART_SHA)).read().split()[0].strip().lower()
    actual = sha256_file(_resolve(PART_JSON))
    if recorded != actual:
        sys.exit(f"[STOP] {PART_JSON} does not match {PART_SHA}.\n"
                 f"  recorded {recorded}\n  actual   {actual}")
    part = json.load(open(_resolve(PART_JSON)))
    una = set(part["unaffected"])

    ck = load_ck(_resolve(CKPT))
    if a.report:
        # Class names are cached on the first real run so that --report does
        # not have to load the dataset and the model just to print labels.
        names = ck.get("class_names") or [f"class_{i}" for i in range(N_CLASSES)]
        if "class_names" not in ck:
            print("  [note] no cached class names in the checkpoint; printing")
            print("         indices. They are cached by the first full run.")
        return report(ck, moms, a.K, names, una)
    ds, model, loader, device = build(a.size, TEST_WEEK)
    names = class_names(ds)
    ck["class_names"] = names
    save_ck(_resolve(CKPT), ck)

    print("=== PER-CLASS EFFECT VERSUS MOMENTUM (declared post-hoc) ===")
    print(f"    partition verified against {PART_SHA} ({actual[:16]}...)")
    print(f"    condition {COND}, lr={LR:.0e}, steps={STEPS}, q={QUANT}")
    print(f"    momenta {moms}, K={a.K}, {N_WINDOWS} windows")
    print("    predictions C-A to C-C are in this file's docstring\n")
    if ck["done"]:
        print(f"[RESUME] {len(ck['done'])} units done\n")

    t0 = time.time()
    for w in range(N_WINDOWS):
        win = collect(loader, w * WINDOW, WINDOW, label=f"W-47 w{w+1}")
        y, p = predict_on_batches(model, win, device)
        if w == 0:
            assert_anchor(float((y == p).mean()), ANCHOR_W47_W1, tol=0.0,
                          name="W-47 window 1 frozen (Table I anchor)")
            print(f"  [ANCHOR OK] window 1 frozen = {float((y==p).mean())!r}")
        rec, sup = per_class(y, p)
        ck["done"][f"w{w}_frozen"] = {"recall": rec.tolist(),
                                      "support": sup.tolist()}
        save_ck(_resolve(CKPT), ck)
        for mom in moms:
            for k in range(a.K):
                key = f"w{w}_m{mom}_{k}"
                if key in ck["done"]:
                    continue
                rng = np.random.default_rng(1000 * w + k)
                order = list(rng.permutation(len(win)))
                m = adapt(model, win, device, order, mom)
                y2, p2 = predict_on_batches(m, win, device)
                r2, _ = per_class(y2, p2)
                ck["done"][key] = {"recall": r2.tolist()}
                save_ck(_resolve(CKPT), ck); del m
                d = (r2 - rec) * 100
                du = [d[c] for c in sorted(una)
                      if not np.isnan(d[c])]
                print(f"    m={mom:<5} k={k}: undrifted mean "
                      f"{np.mean(du):+6.2f}p, worst {min(du):+6.2f}p, "
                      f"{sum(1 for x in du if x < -BIG_DROP)} classes below "
                      f"-{BIG_DROP:.0f}  ({(time.time()-t0)/60:.1f}m)")
        del win
    report(ck, moms, a.K, names, una)


if __name__ == "__main__":
    main()
