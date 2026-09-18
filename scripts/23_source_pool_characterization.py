#!/usr/bin/env python3
r"""
23_source_pool_characterization.py -- why the source day matters.

DECLARED POST-HOC. Follow-up to ADDENDUM_source_day_sweep.md. It measures
properties of each source pool and adds no adaptation, no training and no new
recovery number. Nothing it produces revises a pre-registered value.

WHAT PROMPTED IT:
  The sweep showed the gradient term's contribution collapsing from +7.11 on
  the Monday of W-2022-45 to +0.59 by that Sunday, and sitting at +1.30 on the
  training week W-2022-44. Neither "cleaner is better" nor "more recent is
  better" fits: the training week is the cleanest data the model will ever see
  and it gives almost nothing.

THE HYPOTHESIS THIS TESTS:
  Filtered entropy minimization behaves like self-training on the lowest-
  entropy half of each batch. Its value should then depend on two properties
  of the source pool, both measurable here:

    PURITY    the accuracy of the model's own predictions on the half the
              filter keeps. These are the targets the gradient actually
              pushes toward. Uses labels, so it is a diagnostic an operator
              could not compute. Marked as such.
    HEADROOM  how much entropy there is to minimize in the first place. On
              memorized data the predictions are already saturated, so there
              is little gradient regardless of purity. Label-free.

  Prediction, recorded before the run:
    P-A  W-2022-44 has high purity and LOW headroom  -> small gradient term.
    P-B  early W-2022-45 has high purity and useful headroom -> large term.
    P-C  late W-2022-45 and W-2022-46 keep headroom but lose purity, because
         the confident predictions now include confidently wrong ones on the
         drifted classes -> small term again.
    P-D  across all source days, the gradient term correlates positively with
         purity, and the W-2022-44 point sits off that line, low on headroom.

  If instead the gradient term tracks purity alone with W-2022-44 on the line,
  P-A is wrong and the headroom half of the account should be dropped. If it
  tracks neither, the three-regime story is wrong and the sweep stays an
  unexplained measurement.

WHAT IT COMPUTES, per source day, over the same 200-batch pool the sweep used:
  frozen accuracy; the prediction-entropy distribution; and, splitting at each
  batch's own median entropy exactly as the q=0.5 filter does, the accuracy
  and mean entropy of the kept half and of the dropped half.

Run:
    python scripts/23_source_pool_characterization.py --size S
    python scripts/23_source_pool_characterization.py --size S --days 20221107
Output: source_pool_characterization.json. One forward pass per day.
"""
import argparse, json, os, sys, time
import numpy as np
from tta_guards import guarded_eval

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
BATCH      = 256
POOL_BATCHES = 200
QUANT      = 0.5                       # the filter the method uses
OUT_JSON   = "source_pool_characterization.json"

DAYS = ["20221031", "20221107", "20221108", "20221109", "20221110",
        "20221111", "20221112", "20221113", "20221114", "20221118",
        "20221120"]

_SEARCH = [".", "results/raw", "results"]


def _resolve(name):
    if os.path.isabs(name) or os.path.isfile(name):
        return name
    for d in _SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return name


def build(size, period_name, dates=None):
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
              train_period_name=TRAIN_WEEK, test_period_name=period_name,
              batch_size=BATCH, train_workers=0, test_workers=0,
              use_packet_histograms=True,
              ppi_transform=tr.get("ppi_transform"),
              flowstats_transform=tr.get("flowstats_transform"),
              flowstats_phist_transform=tr.get("flowstats_phist_transform"))
    if dates is not None:
        kw["test_dates"] = list(dates)
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


def characterize(model, loader, device, n_batches):
    """One evaluation-mode pass. Splits each batch at its own median entropy,
    exactly as the q=0.5 filter does, and reports both halves."""
    import torch, torch.nn.functional as F

    def _run():
        was = {n: m.training for n, m in model.named_modules()}
        model.eval()
        ents, corr, kept = [], [], []
        nb = 0
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= n_batches:
                    break
                lo, y = fwd(model, b, device)
                e = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
                sel = e <= torch.quantile(e, QUANT)
                p = lo.argmax(1).cpu().numpy()
                ents.append(e.cpu().numpy())
                corr.append((p == np.asarray(y)).astype(np.int8))
                kept.append(sel.cpu().numpy())
                nb += 1
        for n, m in model.named_modules():
            m.train(was[n])
        return (np.concatenate(ents), np.concatenate(corr),
                np.concatenate(kept), nb)

    return guarded_eval(model, _run)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def gradient_terms():
    """src-tent minus src-stats per source day, from the sweep artifact plus
    the recorded post-drift values. Returns {} if the artifact is absent."""
    p = _resolve("predrift_label_progress.json")
    out = {}
    if os.path.isfile(p):
        from collections import defaultdict
        agg = defaultdict(list)
        for v in json.load(open(p))["done"].values():
            agg[(v["source_day"], v["capacity"])].append(np.mean(v["recoveries"]))
        for (day, cap), vals in agg.items():
            if cap == "src-tent" and (day, "src-stats") in agg:
                out[day] = (float(np.mean(vals))
                            - float(np.mean(agg[(day, "src-stats")])))
    # recorded Experiment B post-drift values
    for day, (ss, st) in {"20221114": (2.16, 2.66), "20221118": (2.74, 2.92),
                          "20221120": (2.21, 2.84)}.items():
        out.setdefault(day, st - ss)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--days", default="", help="comma separated subset")
    ap.add_argument("--batches", type=int, default=POOL_BATCHES)
    args = ap.parse_args()

    days = [d.strip() for d in args.days.split(",") if d.strip()] or DAYS
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    if res:
        print(f"[RESUME] {len(res)} days already characterized\n")

    print("=== SOURCE POOL CHARACTERIZATION (declared post-hoc) ===")
    print("    no adaptation, no training, one forward pass per day")
    print(f"    pool = first {args.batches} batches of each day, filter q={QUANT}")
    print("    PURITY uses labels and is a diagnostic, not an operator signal\n")

    t0 = time.time()
    for day in days:
        if day in res:
            continue
        model, loader, device = build(args.size, f"DAY-{day}", dates=[day])
        e, c, k, nb = characterize(model, loader, device, args.batches)
        if nb < args.batches:
            print(f"  [note] {day}: {nb} batches, short of {args.batches}")
        res[day] = {
            "n_batches": int(nb), "n_flows": int(len(e)),
            "frozen_accuracy": float(c.mean()),
            "entropy_mean": float(e.mean()),
            "entropy_q": {q: float(np.quantile(e, q / 100))
                          for q in (10, 25, 50, 75, 90)},
            "kept_purity": float(c[k].mean()), "kept_entropy": float(e[k].mean()),
            "kept_n": int(k.sum()),
            "dropped_purity": float(c[~k].mean()),
            "dropped_entropy": float(e[~k].mean()),
        }
        json.dump(res, open(OUT_JSON, "w"), indent=1)
        r = res[day]
        print(f"  {day}: acc {r['frozen_accuracy']:.4f}  "
              f"entropy mean {r['entropy_mean']:.4f}  "
              f"kept purity {r['kept_purity']:.4f}  "
              f"kept entropy {r['kept_entropy']:.4f}  "
              f"({(time.time()-t0)/60:.1f}m)")
        del model, loader

    gt = gradient_terms()
    print("\n==== SOURCE POOL PROPERTIES vs THE GRADIENT TERM ====")
    print(f"  {'day':>10} {'acc':>7} {'H mean':>8} {'H kept':>8} "
          f"{'purity':>8} {'gradient':>9}")
    rows = []
    for day in sorted(res):
        r = res[day]
        g = gt.get(day)
        rows.append((day, r["frozen_accuracy"], r["entropy_mean"],
                     r["kept_entropy"], r["kept_purity"], g))
        print(f"  {day:>10} {r['frozen_accuracy']:7.4f} {r['entropy_mean']:8.4f} "
              f"{r['kept_entropy']:8.4f} {r['kept_purity']:8.4f} "
              + (f"{g:9.2f}" if g is not None else f"{'n/a':>9}"))

    have = [r for r in rows if r[5] is not None]
    if len(have) >= 5:
        print("\n  Spearman against the gradient term:")
        for i, name in ((4, "kept purity (P-D)"), (3, "kept entropy, headroom"),
                        (2, "pool entropy mean"), (1, "frozen accuracy")):
            print(f"    {name:<28} {spearman([r[i] for r in have], [r[5] for r in have]):+.3f}")
        w44 = [r for r in have if r[0] == "20221031"]
        if w44:
            r = w44[0]
            others = [x for x in have if x[0] != "20221031"]
            print(f"\n  P-A check, the W-2022-44 point:")
            print(f"    purity {r[4]:.4f} vs others "
                  f"[{min(x[4] for x in others):.4f}, {max(x[4] for x in others):.4f}]")
            print(f"    kept entropy {r[3]:.4f} vs others "
                  f"[{min(x[3] for x in others):.4f}, {max(x[3] for x in others):.4f}]")
            print(f"    gradient {r[5]:+.2f}")
            print("    P-A expects high purity AND low kept entropy here, with a")
            print("    small gradient term: high purity but nothing left to minimize.")
    print(f"\n  raw: {OUT_JSON}")


if __name__ == "__main__":
    main()
