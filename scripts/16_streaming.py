#!/usr/bin/env python3
r"""
16_streaming.py -- Experiment A of the streaming/delayed-label pre-registration:
what label-free adaptation delivers when flows are classified as they arrive.

WHY:
  The recorded results are episodic and transductive: the model adapts on a
  window and is then evaluated on that same window, so every flow is
  classified by a model that has already seen it, and the whole 409,600-flow
  window must be held in memory before any flow in it is classified. The
  stream-order audit measured what those windows are in wall clock: 11.19,
  4.46 and 8.12 hours. Buffering is therefore not a matter of seconds of
  latency but of most of a day. This script measures the deployable
  alternative.

CONDITIONS (all label-free; the model is never given a label):
  frozen              no adaptation. Reference for every streaming number.
  causal-stats        per batch: predict in evaluation mode using only
                      statistics accumulated from strictly earlier batches,
                      then absorb this batch with a forward pass. No gradients.
  causal-filtered     as causal-stats, plus one filtered entropy gradient step
                      on the batch after it has been predicted. HEADLINE.
  batchtrans-filtered predict with batch normalization in training mode, so
                      this batch's own statistics normalize it, then take the
                      gradient step. Requires buffering 2048 flows before
                      classifying any of them. The difference from
                      causal-filtered is the measured value of that buffer.
  reset-200           causal-filtered with the model reset to the pretrained
                      weights every 200 batches, bounding how far the adapted
                      state can travel.

CAUSALITY:
  In the three causal conditions the prediction for batch b uses only
  information from batches strictly before b. That is the property that makes
  a streaming number deployable, and it is why prediction happens before
  adaptation rather than after.

ANCHOR:
  The frozen condition over batches 0 to 199 of W-2022-47 must reproduce the
  recorded Table I window 1 accuracy 0.72239013671875 exactly. It is asserted
  at zero tolerance. A mismatch means the streaming harness does not agree
  with the existing record and nothing it produces should be believed.

STATE AUDITING:
  Every prediction pass saves and restores module training flags, and is
  additionally wrapped in the strict state guard on the first batch and every
  --guard-every batches thereafter. The guard hashes every parameter and
  batch-normalization buffer, which is too slow to run on all 3,227 batches of
  a full-week pass; --guard-every 1 forces it on every batch for a paranoid
  run. An evaluation-mode forward under no_grad cannot update running
  statistics or parameters, so the sampled guard is checking the code, not the
  arithmetic.

ORDER OF OPERATIONS ENFORCED BY THIS SCRIPT:
  --tune runs on W-2022-46 only and writes streaming_config.json.
  --report refuses to run until that file exists. W-2022-47 cannot be touched
  before the configuration is selected and recorded.

Run:
    python scripts/16_streaming.py --tune   --size S
    python scripts/16_streaming.py --report --size S
Resumable: streaming_ckpt_<tag>.pt after every --ckpt-every batches.
"""
import argparse, copy, json, os, sys, time
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TUNE_WEEK  = "W-2022-46"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
AUDIT_JSON = "streaming_order_audit.json"
CONFIG_JSON = "streaming_config.json"
OUT_JSON   = "streaming_results.json"

BN_MOM        = 0.1
TUNE_BATCHES  = 600                      # pre-registered
GRID_LR       = [1e-4, 1e-3]             # pre-registered
GRID_QUANT    = [0.5, 1.0]               # pre-registered
TIE_POINTS    = 0.05                     # ties break to the smaller lr
RESET_PERIOD  = 200
REPORT_WINDOWS = [(0, 200), (200, 400), (400, 600)]
ANCHOR_W47_W1 = 0.72239013671875

# Conditions and how far each is run on the report week. The split is a
# compute allocation fixed in the pre-registration, not a choice made after
# seeing which conditions looked better.
FULL_WEEK_CONDS = ["frozen", "causal-filtered"]
WINDOWS_ONLY_CONDS = ["causal-stats", "batchtrans-filtered", "reset-200"]


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


def bn_prepare(base_model, with_grad):
    import torch.nn as nn
    m = copy.deepcopy(base_model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = BN_MOM
            if with_grad:
                mod.requires_grad_(True)
                if mod.weight is not None: params.append(mod.weight)
                if mod.bias is not None:   params.append(mod.bias)
    return m, params


def predict_causal(model, batch, device, guard):
    """Predict using running statistics only, i.e. information from strictly
    earlier batches. Saves and restores training flags; optionally proves
    it under the strict state guard."""
    import torch
    def _run():
        was = {n: mod.training for n, mod in model.named_modules()}
        model.eval()
        with torch.no_grad():
            lo, y = fwd(model, batch, device)
            p = lo.argmax(1).cpu().numpy()
        for n, mod in model.named_modules():
            mod.train(was[n])
        return p, y
    return guarded_eval(model, _run) if guard else _run()


def entropy_step(m, params, opt, batch, device, quant):
    import torch, torch.nn.functional as F
    lo, y = fwd(m, batch, device)
    ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
    if quant < 1.0:
        sel = ent <= torch.quantile(ent.detach(), quant)
        loss = ent[sel].mean() if sel.any() else ent.mean()
    else:
        loss = ent.mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return lo, y


def stream(base_model, loader, device, cond, lr, quant, n_batches,
           guard_every, ckpt_path, ckpt_every, resume):
    """Prequential pass. Returns per-batch (correct, n) arrays."""
    import torch
    grads = cond in ("causal-filtered", "batchtrans-filtered", "reset-200")
    reset_every = RESET_PERIOD if cond == "reset-200" else 0

    def fresh():
        if cond == "frozen":
            return copy.deepcopy(base_model), None, None
        m, params = bn_prepare(base_model, with_grad=grads)
        o = torch.optim.Adam(params, lr=lr) if grads else None
        return m, params, o

    m, params, opt = fresh()
    correct, ntot, start = [], [], 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        if st["cond"] == cond and st["lr"] == lr and st["quant"] == quant:
            m.load_state_dict(st["model"])
            if opt is not None and st.get("opt"):
                opt.load_state_dict(st["opt"])
            correct, ntot = list(st["correct"]), list(st["n"])
            start = len(correct)
            print(f"  [RESUME] {cond}: {start} batches already streamed")
        else:
            print(f"  [note] {ckpt_path} is for a different condition/config; "
                  f"ignoring it")

    t0 = time.time()
    for i, b in enumerate(loader):
        if i >= n_batches: break
        if i < start: continue                       # fast-forward on resume
        if reset_every and i % reset_every == 0 and i > 0:
            m, params, opt = fresh()
        guard = (i == 0) or (guard_every > 0 and i % guard_every == 0)

        if cond == "batchtrans-filtered":
            # One forward in training mode: this batch's own statistics
            # normalize it, the prediction is read from that forward, then the
            # gradient step follows.
            lo, y = entropy_step(m, params, opt, b, device, quant)
            p = lo.detach().argmax(1).cpu().numpy()
        else:
            p, y = predict_causal(m, b, device, guard)
            if cond == "causal-stats":
                with torch.no_grad():
                    fwd(m, b, device)                # absorb: statistics only
            elif cond in ("causal-filtered", "reset-200"):
                entropy_step(m, params, opt, b, device, quant)
            elif cond != "frozen":
                raise ValueError(cond)

        correct.append(int((p == np.asarray(y)).sum()))
        ntot.append(int(len(y)))

        if ckpt_path and ckpt_every and (i + 1) % ckpt_every == 0:
            torch.save({"cond": cond, "lr": lr, "quant": quant,
                        "correct": correct, "n": ntot,
                        "model": m.state_dict(),
                        "opt": opt.state_dict() if opt is not None else None},
                       ckpt_path)
        if (i + 1) % 200 == 0:
            acc = sum(correct) / max(1, sum(ntot))
            print(f"    {cond}: {i+1}/{n_batches} batches, "
                  f"running acc {acc:.4f} ({(time.time()-t0)/60:.1f}m)")
    return np.asarray(correct), np.asarray(ntot)


def acc_range(correct, ntot, a, b):
    a, b = max(0, a), min(len(correct), b)
    if b <= a: return None
    return float(correct[a:b].sum()) / float(ntot[a:b].sum())


def load_day_map():
    if not os.path.isfile(_resolve(AUDIT_JSON)):
        return None
    with open(_resolve(AUDIT_JSON)) as f: audit = json.load(f)
    a = audit.get(TEST_WEEK)
    if not a or a.get("capped"): return None
    return {d: di for d, di in a["day_map_from_indices"].items() if di.get("flows")}


def do_tune(args):
    nb = args.smoke or TUNE_BATCHES
    smoke = args.smoke is not None
    if smoke:
        print(f"*** SMOKE RUN: {nb} batches instead of {TUNE_BATCHES}. "
              f"{CONFIG_JSON} will NOT be written and no configuration is "
              f"selected. This run cannot be mistaken for tuning. ***\n")
    print(f"=== EXPERIMENT A TUNING on {TUNE_WEEK} "
          f"(first {nb} batches) ===")
    print(f"    grid lr x quantile = {GRID_LR} x {GRID_QUANT}, "
          f"condition causal-filtered")
    print(f"    {TEST_WEEK} IS NOT TOUCHED BY THIS MODE\n")
    model, loader, device = build(args.size, TUNE_WEEK)
    print(f"device={device}\n")
    results = {}
    for lr in GRID_LR:
        for q in GRID_QUANT:
            tag = f"tune_lr{lr:.0e}_q{q}"
            c, n = stream(model, loader, device, "causal-filtered", lr, q,
                          nb, args.guard_every,
                          (None if smoke else f"streaming_ckpt_{tag}.pt"),
                          args.ckpt_every, not smoke)
            a = acc_range(c, n, 0, nb)
            results[f"lr={lr:.0e},q={q}"] = {"lr": lr, "quant": q, "accuracy": a}
            print(f"  lr={lr:.0e} q={q}: streaming accuracy {a:.4f}\n")
    # frozen reference on the same batches
    c, n = stream(model, loader, device, "frozen", 0, 0, nb,
                  args.guard_every, None, 0, False)
    frozen = acc_range(c, n, 0, nb)
    print(f"  frozen reference on the same batches: {frozen:.4f}\n")

    best = max(results.values(), key=lambda r: (round(r["accuracy"], 6), -r["lr"]))
    near = [r for r in results.values()
            if r["accuracy"] >= best["accuracy"] - TIE_POINTS / 100.0]
    sel = min(near, key=lambda r: r["lr"])          # ties to the smaller lr
    if smoke:
        print("\n  [SMOKE] plumbing exercised; nothing written. Rerun without "
              "--smoke to select the configuration.")
        return
    cfg = {"lr": sel["lr"], "quant": sel["quant"],
           "tune_week": TUNE_WEEK, "tune_batches": TUNE_BATCHES,
           "tune_accuracy": sel["accuracy"], "frozen_reference": frozen,
           "grid": results, "reset_period": RESET_PERIOD,
           "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(CONFIG_JSON, "w") as f: json.dump(cfg, f, indent=1)
    print("==== SELECTED (frozen from here on) ====")
    print(f"  lr = {sel['lr']:.0e}, quantile = {sel['quant']}  "
          f"(W-46 streaming accuracy {sel['accuracy']:.4f} vs frozen {frozen:.4f}, "
          f"{(sel['accuracy']-frozen)*100:+.2f}p)")
    print(f"  written to {CONFIG_JSON}")
    if sel["accuracy"] <= frozen:
        print("\n  [A-K1] The selected configuration does NOT beat the frozen "
              "model on the tuning week. Per the pre-registration the report "
              "week is still run and reported, but the claim is killed: the "
              "paper states that under strict causality label-free adaptation "
              "does not deliver a usable recovery on this event, and says so "
              "in the abstract.")
    print(f"\n  Record this configuration in RESULTS.md, then run --report.")


def do_report(args):
    if not os.path.isfile(_resolve(CONFIG_JSON)):
        sys.exit(f"[STOP] {CONFIG_JSON} not found. Run --tune first. The "
                 f"report week is not touched before the configuration is "
                 f"selected on {TUNE_WEEK} and recorded.")
    with open(_resolve(CONFIG_JSON)) as f: cfg = json.load(f)
    lr, quant = cfg["lr"], cfg["quant"]
    print(f"=== EXPERIMENT A REPORT on {TEST_WEEK} ===")
    print(f"    frozen config from {CONFIG_JSON}: lr={lr:.0e}, q={quant} "
          f"(selected {cfg['selected_at']})\n")

    day_map = load_day_map()
    model, loader, device = build(args.size, TEST_WEEK)
    print(f"device={device}")
    n_week = args.max_batches
    if n_week is None:
        if day_map:
            n_week = max(di["last_batch_index"] for di in day_map.values()) + 1
        else:
            sys.exit("[STOP] no uncapped audit found; pass --max-batches or "
                     "run scripts/14_stream_order_audit.py first.")
    print(f"    full-week length = {n_week:,} batches\n")

    out = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f: out = json.load(f)

    conds = ([c for c in FULL_WEEK_CONDS] +
             [c for c in WINDOWS_ONLY_CONDS])
    for cond in conds:
        if cond in out and not args.force:
            print(f"  {cond}: already in {OUT_JSON}; skipping")
            continue
        nb = n_week if cond in FULL_WEEK_CONDS else REPORT_WINDOWS[-1][1]
        print(f"\n  --- {cond} over {nb:,} batches ---")
        c, n = stream(model, loader, device, cond, lr, quant, nb,
                      args.guard_every, f"streaming_ckpt_{cond}.pt",
                      args.ckpt_every, True)
        rec = {"condition": cond, "lr": lr, "quant": quant,
               "n_batches": int(len(c)), "n_flows": int(n.sum()),
               "accuracy_overall": acc_range(c, n, 0, len(c)),
               "per_window": {}, "per_day": {}}
        for wi, (a, b) in enumerate(REPORT_WINDOWS):
            rec["per_window"][f"window_{wi+1}"] = acc_range(c, n, a, b)
        if cond == "frozen":
            w1 = rec["per_window"]["window_1"]
            assert_anchor(w1, ANCHOR_W47_W1, tol=0.0,
                          name="W-47 window 1 frozen (Table I anchor)")
            print(f"  [ANCHOR OK] frozen window 1 = {w1!r} reproduces the "
                  f"Table I record exactly")
        if day_map and cond in FULL_WEEK_CONDS:
            for d, di in day_map.items():
                a, b = di["first_batch_index"], di["last_batch_index"] + 1
                rec["per_day"][d] = acc_range(c, n, a, b)
        out[cond] = rec
        with open(OUT_JSON, "w") as f: json.dump(out, f, indent=1)
        print(f"  {cond}: overall {rec['accuracy_overall']:.4f}, "
              f"windows " + " / ".join(
                  f"{v:.4f}" for v in rec["per_window"].values()))

    print("\n==== EXPERIMENT A RESULT ====")
    fr = out.get("frozen", {})
    fw = fr.get("per_window", {})
    def cell(v, w=8):
        return f"{v:{w}.4f}" if v is not None else " " * (w - 3) + "n/a"

    print(f"{'condition':>22} {'w1':>8} {'w2':>8} {'w3':>8} {'3-window':>10} "
          f"{'full week':>10}")
    for cond in conds:
        if cond not in out: continue
        r = out[cond]
        pw = r["per_window"]
        vals = [pw.get(f"window_{i}") for i in (1, 2, 3)]
        three = (float(np.mean(vals)) if all(v is not None for v in vals)
                 else None)
        full = r["accuracy_overall"] if cond in FULL_WEEK_CONDS else None
        print(f"{cond:>22} " + " ".join(cell(v) for v in vals)
              + f" {cell(three, 10)} {cell(full, 10)}")
    print("\n  recovery over frozen, in points:")
    for cond in conds:
        if cond not in out or cond == "frozen": continue
        pw, fwd_ = out[cond]["per_window"], fw
        vals = [(pw[f"window_{i}"] - fwd_[f"window_{i}"]) * 100
                for i in (1, 2, 3)
                if pw.get(f"window_{i}") is not None
                and fwd_.get(f"window_{i}") is not None]
        if not vals: continue
        below = [i + 1 for i, v in enumerate(vals) if v < 0]
        note = (f"   [A-K2] window(s) {below} BELOW the frozen baseline; "
                f"no aggregate may be presented without this fact adjacent"
                if below else "")
        print(f"    {cond:>22}: " + " / ".join(f"{v:+.2f}" for v in vals)
              + f"   mean {np.mean(vals):+.2f}p{note}")
    if "causal-filtered" in out and "frozen" in out:
        a = out["causal-filtered"]["accuracy_overall"]
        b = out["frozen"]["accuracy_overall"]
        print(f"\n  full-week causal-filtered {a:.4f} vs frozen {b:.4f} "
              f"= {(a-b)*100:+.2f}p over {out['frozen']['n_batches']:,} batches")
        if out["causal-filtered"].get("per_day"):
            print("  per day (causal-filtered minus frozen, points):")
            for d, v in out["causal-filtered"]["per_day"].items():
                f0 = out["frozen"]["per_day"].get(d)
                if f0 is None: continue
                print(f"    {d}: {v:.4f} vs {f0:.4f}  {(v-f0)*100:+.2f}p")
    print(f"\n  raw: {OUT_JSON}")
    print("  The episodic headline (+3.06p) is reported beside these numbers, "
          "labeled as requiring the whole window in memory before any flow in "
          "it is classified: 11.19, 4.46 and 8.12 hours of traffic for the "
          "three windows respectively.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="override the full-week length; for smoke tests")
    ap.add_argument("--guard-every", type=int, default=100,
                    help="run the strict state guard every N batches "
                         "(1 = every batch, 0 = first batch only)")
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--smoke", type=int, default=None,
                    help="tuning smoke test over N batches; writes no config")
    ap.add_argument("--force", action="store_true",
                    help="recompute conditions already present in the output")
    args = ap.parse_args()
    if args.tune == args.report:
        sys.exit("[STOP] pass exactly one of --tune or --report.")
    (do_tune if args.tune else do_report)(args)


if __name__ == "__main__":
    main()
