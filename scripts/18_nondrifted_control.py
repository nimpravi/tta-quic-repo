#!/usr/bin/env python3
r"""
18_nondrifted_control.py -- Experiment C of the pre-registration: what
label-free adaptation COSTS when nothing has drifted.

WHY:
  Every number in this study so far measures what adaptation recovers on
  traffic that was damaged by the certificate change. An operator does not get
  to run adaptation only on damaged traffic. If adaptation is deployed
  continuously it also runs on traffic that never drifted, and any accuracy it
  destroys there is a real cost that must be set against the recovery.

THREE MODES, in this order:

  --c1         Period-level control. The frozen configuration (stats-only and
               filtered) applied to three windows of W-2022-45 before the
               event manifests, and three windows of W-2022-44. Recovery here
               is expected to be near zero or negative. Whatever it is, it is
               the price of running adaptation on undamaged traffic.

  --partition  Builds the affected/unaffected class partition from W-2022-45
               and W-2022-46 ONLY, by the pre-registered rule: a class is
               affected if its frozen per-class recall on W-2022-46 is more
               than 10 points below its recall on the pre-shift windows of
               W-2022-45. Writes class_partition.json. W-2022-47 is not
               touched. The partition must then be hashed and committed
               before --c2 will run.

  --c2         Per-partition report on W-2022-47. Refuses to start unless
               class_partition.sha256 exists and matches class_partition.json,
               so the partition cannot be adjusted after seeing a report-week
               number.

               Since the threshold addendum, --c2 also records PER-CLASS flow
               and correct counts for every unit, frozen and adapted. The
               partition aggregates it already wrote are unchanged, so the
               artifact is a strict superset of the recorded one and
               21_verify_all.py keeps passing. The per-class field makes every
               alternative partition threshold an offline arithmetic exercise
               rather than another model run, and it is what
               scripts/22_threshold_sweep.py consumes.

TWO THINGS THE PRE-REGISTRATION DID NOT SPECIFY, HANDLED LITERALLY:
  1. No minimum class support. The rule is applied as written. Classes with
     zero support in either period have undefined recall and are placed in a
     third "undetermined" bucket rather than silently assigned. Per-class
     support is written into the partition file so that any affected class
     resting on thin support is visible rather than hidden. Applying a support
     threshold now would be a discretionary choice the locked document does
     not authorize; if one is wanted it is a post-hoc sensitivity analysis and
     must be labeled as such.
  2. Which W-2022-46 windows. The first three (batches 0 to 599) are used, for
     symmetry with the three pre-shift W-2022-45 windows.

W-2022-44 CAVEAT, printed at run time: the public weights were trained on
W-2022-44, so its test flows overlap the model's training data and its frozen
accuracy is inflated by memorization. It is run because it is pre-registered,
and it is reported with this caveat attached rather than as a clean control.

Run:
    python scripts/18_nondrifted_control.py --c1        --size S --K 3
    python scripts/18_nondrifted_control.py --partition --size S
    # hash class_partition.json into class_partition.sha256, commit both
    python scripts/18_nondrifted_control.py --c2        --size S --K 3
"""
import argparse, copy, hashlib, json, os, sys, time
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
REF_WEEK   = "W-2022-45"
TUNE_WEEK  = "W-2022-46"
TEST_WEEK  = "W-2022-47"
BATCH      = 256

WINDOW      = 200
N_WINDOWS   = 3
LR, STEPS, QUANT, BN_MOM = 1e-3, 50, 0.5, 0.1     # frozen episodic config
RECALL_DROP = 0.10                                 # pre-registered rule
C3_COST     = 0.5                                  # pre-registered kill rule

PART_JSON   = "class_partition.json"
PART_SHA    = "class_partition.sha256"
C1_CKPT     = "nondrifted_c1_progress.json"
C2_CKPT     = "nondrifted_c2_progress.json"

ANCHOR_W47_W1 = 0.72239013671875
ANCHOR_W45 = [0.955947265625, 0.95069580078125, 0.95899658203125]


# --- input path resolution -------------------------------------------------
# Reads search the working directory first, then results/raw and results, so
# these scripts keep working after the raw artifacts are moved into
# results/raw/. Writes are unaffected and still land in the working
# directory, so a rerun never overwrites a committed artifact in place.
_SEARCH = [".", "results/raw", "results"]


def _resolve(name):
    """Return an existing path for `name`, or `name` itself if not found so
    that the caller's own missing-file handling still runs."""
    if os.path.isabs(name) or os.path.isfile(name):
        return name
    for d in _SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return name


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
    return ds, model, ds.get_test_dataloader(), device


def class_names(ds):
    """Index -> application name, from the dataset's own encoder. Falls back to
    plain indices rather than inventing names."""
    try:
        ci = ds.class_info
        names = list(ci.target_names)
        prov = dict(ci.provider_mapping)
        return names, prov
    except Exception as e:
        print(f"  [note] class names unavailable ({e}); using indices")
        return None, None


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


def collect(loader, skip, n, label=""):
    out = []
    for i, b in enumerate(loader):
        if i < skip: continue
        out.append(b)
        if len(out) >= n: break
    if len(out) < n:
        print(f"  [WARN]{' '+label if label else ''} wanted {n} from {skip}, "
              f"got {len(out)}")
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


def accuracy_on_batches(model, batches, device):
    y, p = predict_on_batches(model, batches, device)
    return float((y == p).mean())


def adapt(base_model, window, device, cond, order):
    import torch, torch.nn as nn, torch.nn.functional as F
    if cond == "frozen":
        return base_model
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = BN_MOM
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


def per_class_counts(y, p, n_classes):
    """Flows per class and correct predictions per class, as plain ints."""
    total = np.zeros(n_classes, dtype="int64")
    correct = np.zeros(n_classes, dtype="int64")
    np.add.at(total, y, 1)
    np.add.at(correct, y[y == p], 1)
    return total.tolist(), correct.tolist()


def load_ck(p):
    return json.load(open(p)) if os.path.exists(p) else {"done": {}}


def save_ck(p, c):
    json.dump(c, open(p, "w"), indent=1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- C1
def do_c1(args):
    ck = load_ck(C1_CKPT)
    print("=== EXPERIMENT C1: adaptation applied to NON-DRIFTED traffic ===")
    print(f"    frozen config lr={LR:.0e}, steps={STEPS}, q={QUANT}\n")
    for week in (REF_WEEK, TRAIN_WEEK):
        if week == TRAIN_WEEK:
            print(f"\n  [CAVEAT] {TRAIN_WEEK} is the week the public weights were "
                  f"trained on. Its test flows overlap the model's training "
                  f"data, so the frozen accuracy below is inflated by "
                  f"memorization. Reported with this caveat, not as a clean "
                  f"control.")
        print(f"\n--- {week} ---")
        ds, model, loader, device = build(args.size, week)
        for w in range(N_WINDOWS):
            win = collect(loader, w * WINDOW, WINDOW, label=f"{week} w{w+1}")
            kf = f"{week}_w{w}_frozen"
            if kf in ck["done"]:
                fr = ck["done"][kf]["accuracy"]
            else:
                fr = accuracy_on_batches(model, win, device)
                if week == REF_WEEK:
                    assert_anchor(fr, ANCHOR_W45[w], tol=0.0,
                                  name=f"W-45 window {w+1} frozen")
                ck["done"][kf] = {"accuracy": fr}; save_ck(C1_CKPT, ck)
            print(f"  window {w+1}: frozen = {fr:.4f}")
            for cond in ("stats", "filtered"):
                for k in range(args.K):
                    key = f"{week}_w{w}_{cond}_{k}"
                    if key in ck["done"]: continue
                    rng = np.random.default_rng(1000 * w + k)
                    order = list(rng.permutation(len(win)))
                    t0 = time.time()
                    m = adapt(model, win, device, cond, order)
                    a = accuracy_on_batches(m, win, device)
                    ck["done"][key] = {"accuracy": a, "frozen": fr,
                                       "recovery": (a - fr) * 100}
                    save_ck(C1_CKPT, ck); del m
                    print(f"    {cond:>8} k={k}: {a:.4f}  "
                          f"{(a-fr)*100:+.2f}p  ({(time.time()-t0)/60:.1f}m)")
            del win

    print("\n==== C1 RESULT: change in accuracy on non-drifted traffic ====")
    print(f"  {'period':>12} {'window':>7} {'frozen':>8} {'stats':>14} {'filtered':>14}")
    worst = {}
    for week in (REF_WEEK, TRAIN_WEEK):
        for w in range(N_WINDOWS):
            fr = ck["done"].get(f"{week}_w{w}_frozen", {}).get("accuracy")
            if fr is None: continue
            row = []
            for cond in ("stats", "filtered"):
                v = [ck["done"][f"{week}_w{w}_{cond}_{k}"]["recovery"]
                     for k in range(args.K)
                     if f"{week}_w{w}_{cond}_{k}" in ck["done"]]
                row.append((np.mean(v), np.std(v)) if v else None)
                if v:
                    worst[cond] = min(worst.get(cond, 99), float(np.mean(v)))
            def f(t): return f"{t[0]:+.2f} ± {t[1]:.2f}" if t else "n/a"
            print(f"  {week:>12} {w+1:>7} {fr:8.4f} {f(row[0]):>14} {f(row[1]):>14}")
    print("\n  [C3] pre-registered rule: if adaptation reduces accuracy on "
          f"non-drifted traffic by more than {C3_COST} points, that cost is "
          "stated in the abstract, not only in the limitations section.")
    for cond, v in worst.items():
        verdict = ("FIRES: state in the abstract" if v < -C3_COST
                   else "does not fire")
        print(f"    worst {cond:>8} change on any non-drifted window: "
              f"{v:+.2f}p  -> C3 {verdict}")


# --------------------------------------------------------------- partition
def per_class_recall(model, loader, device, n_batches, n_classes, label):
    correct = np.zeros(n_classes, dtype="int64")
    total = np.zeros(n_classes, dtype="int64")
    t0 = time.time()
    for i, b in enumerate(loader):
        if i >= n_batches: break
        y, p = predict_on_batches(model, [b], device)
        np.add.at(total, y, 1)
        np.add.at(correct, y[y == p], 1)
        if (i + 1) % 200 == 0:
            print(f"    {label}: {i+1}/{n_batches} batches "
                  f"({(time.time()-t0)/60:.1f}m)")
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(total > 0, correct / np.maximum(total, 1), np.nan)
    return rec, total


def do_partition(args):
    if os.path.exists(_resolve(PART_JSON)) and not args.force:
        sys.exit(f"[STOP] {PART_JSON} already exists. Rebuilding it after a "
                 f"report-week number exists would defeat its purpose. Use "
                 f"--force only if no --c2 run has ever been performed.")
    n = N_WINDOWS * WINDOW
    print("=== BUILDING THE CLASS PARTITION (W-2022-45 and W-2022-46 ONLY) ===")
    print(f"    rule: affected if recall on {TUNE_WEEK} is more than "
          f"{RECALL_DROP*100:.0f} points below recall on the pre-shift windows "
          f"of {REF_WEEK}")
    print(f"    windows: first {n} batches of each\n")
    ds, model, l45, device = build(args.size, REF_WEEK)
    names, prov = class_names(ds)
    n_classes = len(names) if names else 200
    r45, t45 = per_class_recall(model, l45, device, n, n_classes, "W-45")
    _, _, l46, _ = build(args.size, TUNE_WEEK)
    r46, t46 = per_class_recall(model, l46, device, n, n_classes, "W-46")

    affected, unaffected, undetermined = [], [], []
    rows = {}
    for c in range(n_classes):
        if t45[c] == 0 or t46[c] == 0:
            bucket = "undetermined"
        elif r46[c] < r45[c] - RECALL_DROP:
            bucket = "affected"
        else:
            bucket = "unaffected"
        {"affected": affected, "unaffected": unaffected,
         "undetermined": undetermined}[bucket].append(c)
        rows[str(c)] = {
            "name": names[c] if names and c < len(names) else str(c),
            "provider": (prov.get(names[c]) if names and prov and c < len(names)
                         else None),
            "recall_W45": None if np.isnan(r45[c]) else round(float(r45[c]), 6),
            "recall_W46": None if np.isnan(r46[c]) else round(float(r46[c]), 6),
            "support_W45": int(t45[c]), "support_W46": int(t46[c]),
            "bucket": bucket}

    out = {"rule": f"affected if recall_W46 < recall_W45 - {RECALL_DROP}",
           "reference_period": REF_WEEK, "comparison_period": TUNE_WEEK,
           "batches_per_period": n, "min_support_threshold": None,
           "note": ("No minimum support was pre-registered, so none is "
                    "applied. Classes with zero support in either period have "
                    "undefined recall and are 'undetermined'. Per-class "
                    "support is included so thin evidence is visible."),
           "affected": affected, "unaffected": unaffected,
           "undetermined": undetermined, "classes": rows}
    json.dump(out, open(PART_JSON, "w"), indent=1)

    print(f"\n  affected     : {len(affected)} classes, "
          f"{int(t47_support(t46, affected)):,} W-46 flows")
    print(f"  unaffected   : {len(unaffected)} classes")
    print(f"  undetermined : {len(undetermined)} classes")
    if names and prov:
        from collections import Counter
        cnt = Counter(rows[str(c)]["provider"] or "(none)" for c in affected)
        print("\n  providers represented in the affected set "
              "(reported, not assumed):")
        for p, k in cnt.most_common(10):
            print(f"    {p}: {k} class(es)")
    thin = [c for c in affected if t45[c] < 100 or t46[c] < 100]
    if thin:
        print(f"\n  [note] {len(thin)} affected class(es) rest on fewer than "
              f"100 flows in one of the two periods; their recall estimates "
              f"are noisy and this is visible in {PART_JSON}.")
    h = sha256_file(_resolve(PART_JSON))
    print(f"\n  {PART_JSON} written.")
    print(f"  SHA-256: {h}")
    print(f"\n  NEXT, before --c2 will run:")
    print(f"    1. write that hash into {PART_SHA}")
    print(f"    2. commit both files")
    print(f"  --c2 verifies the hash and refuses otherwise, so the partition "
          f"cannot be adjusted after a report-week number exists.")


def t47_support(t, idx):
    return sum(int(t[c]) for c in idx)


# --------------------------------------------------------------- C2
def do_c2(args):
    if not os.path.exists(_resolve(PART_JSON)):
        sys.exit(f"[STOP] {PART_JSON} not found. Run --partition first.")
    if not os.path.exists(_resolve(PART_SHA)):
        sys.exit(f"[STOP] {PART_SHA} not found. Hash {PART_JSON}, write the "
                 f"digest into {PART_SHA}, and commit both before running "
                 f"--c2. Current digest: {sha256_file(_resolve(PART_JSON))}")
    recorded = open(_resolve(PART_SHA)).read().split()[0].strip().lower()
    actual = sha256_file(_resolve(PART_JSON))
    if recorded != actual:
        sys.exit(f"[STOP] {PART_JSON} does not match {PART_SHA}.\n"
                 f"  recorded {recorded}\n  actual   {actual}\n"
                 f"  The partition changed after it was committed. Stop and "
                 f"establish why before recording any number.")
    part = json.load(open(_resolve(PART_JSON)))
    aff = set(part["affected"]); una = set(part["unaffected"])
    n_classes = max(int(c) for c in part["classes"]) + 1
    print("=== EXPERIMENT C2: per-partition effect on W-2022-47 ===")
    print(f"    partition verified against {PART_SHA} ({actual[:16]}...)")
    print(f"    affected {len(aff)} classes, unaffected {len(una)}, "
          f"undetermined {len(part['undetermined'])}\n")

    ck = load_ck(C2_CKPT)
    ds, model, loader, device = build(args.size, TEST_WEEK)
    for w in range(N_WINDOWS):
        win = collect(loader, w * WINDOW, WINDOW, label=f"W-47 w{w+1}")
        y, p = predict_on_batches(model, win, device)
        if w == 0:
            assert_anchor(float((y == p).mean()), ANCHOR_W47_W1, tol=0.0,
                          name="W-47 window 1 frozen (Table I anchor)")
            print(f"  [ANCHOR OK] window 1 frozen = {float((y==p).mean())!r}")
        ma = np.isin(y, list(aff)); mu = np.isin(y, list(una))
        base = {"affected": float((y[ma] == p[ma]).mean()) if ma.any() else None,
                "unaffected": float((y[mu] == p[mu]).mean()) if mu.any() else None,
                "n_affected": int(ma.sum()), "n_unaffected": int(mu.sum())}
        tot, cor = per_class_counts(y, p, n_classes)
        base["per_class_total"] = tot
        base["per_class_correct"] = cor
        base["n_classes"] = n_classes
        base["class_partition_sha256"] = actual
        ck["done"][f"w{w}_frozen"] = base; save_ck(C2_CKPT, ck)
        print(f"  window {w+1} frozen: affected {base['affected']:.4f} "
              f"({base['n_affected']:,} flows), unaffected "
              f"{base['unaffected']:.4f} ({base['n_unaffected']:,} flows)")
        for cond in ("stats", "filtered"):
            for k in range(args.K):
                key = f"w{w}_{cond}_{k}"
                # A unit recorded before the per-class field existed is
                # recomputed rather than skipped, so a stale checkpoint
                # cannot leave the artifact half populated.
                if key in ck["done"] and "per_class_correct" in ck["done"][key]:
                    continue
                rng = np.random.default_rng(1000 * w + k)
                order = list(rng.permutation(len(win)))
                m = adapt(model, win, device, cond, order)
                y2, p2 = predict_on_batches(m, win, device)
                # The window and its order are identical, so the label vector
                # must be too. If it is not, the per-class totals recorded for
                # the frozen unit do not describe this unit and nothing below
                # is comparable.
                if not np.array_equal(y, y2):
                    sys.exit(f"[STOP] {key}: label vector differs from the "
                             f"frozen pass over the same window. The loader is "
                             f"not returning a stable order; per-class counts "
                             f"cannot be aligned.")
                _, cor2 = per_class_counts(y2, p2, n_classes)
                ck["done"][key] = {
                    "affected": float((y2[ma] == p2[ma]).mean()),
                    "unaffected": float((y2[mu] == p2[mu]).mean()),
                    "per_class_correct": cor2}
                save_ck(C2_CKPT, ck); del m
                print(f"    {cond:>8} k={k}: affected "
                      f"{(ck['done'][key]['affected']-base['affected'])*100:+.2f}p"
                      f"   unaffected "
                      f"{(ck['done'][key]['unaffected']-base['unaffected'])*100:+.2f}p")
        del win

    print("\n==== C2 RESULT: change in accuracy by class partition ====")
    print(f"  {'window':>7} {'condition':>10} {'affected':>14} {'unaffected':>14}")
    worst_un = 99.0
    for w in range(N_WINDOWS):
        base = ck["done"].get(f"w{w}_frozen")
        if not base: continue
        for cond in ("stats", "filtered"):
            v = [ck["done"][f"w{w}_{cond}_{k}"] for k in range(args.K)
                 if f"w{w}_{cond}_{k}" in ck["done"]]
            if not v: continue
            da = np.mean([x["affected"] for x in v]) - base["affected"]
            du = np.mean([x["unaffected"] for x in v]) - base["unaffected"]
            worst_un = min(worst_un, du * 100)
            print(f"  {w+1:>7} {cond:>10} {da*100:13.2f}p {du*100:13.2f}p")
    print(f"\n  [C3] worst change on the UNAFFECTED partition: "
          f"{worst_un:+.2f}p")
    if worst_un < -C3_COST:
        print(f"  *** C3 FIRES *** adaptation costs more than {C3_COST} points "
              f"on traffic that did not drift. Per the pre-registration this "
              f"is stated in the abstract, not only in the limitations.")
    else:
        print(f"  C3 does not fire (threshold {-C3_COST:+.2f}p).")

    print(f"\n  Per-class counts recorded for every unit in {C2_CKPT}. Run")
    print(f"  scripts/22_threshold_sweep.py for the threshold sweep and the")
    print(f"  continuous drift-magnitude analysis. The pre-registered 10-point")
    print(f"  partition results above are unchanged and remain primary.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--c1", action="store_true")
    ap.add_argument("--partition", action="store_true")
    ap.add_argument("--c2", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if sum([a.c1, a.partition, a.c2]) != 1:
        sys.exit("[STOP] pass exactly one of --c1, --partition, --c2.")
    (do_c1 if a.c1 else do_partition if a.partition else do_c2)(a)


if __name__ == "__main__":
    main()
