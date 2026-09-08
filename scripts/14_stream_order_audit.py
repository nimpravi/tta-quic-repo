#!/usr/bin/env python3
r"""
14_stream_order_audit.py -- A0 of the streaming/delayed-label pre-registration.

WHAT THIS ANSWERS:
  The streaming experiments assume that consuming the test dataloader in its
  natural order approximates consuming traffic in arrival order. That is not
  the same as claiming the stream is sorted to the individual flow. This
  script measures what the stream actually is, at the resolution the
  experiments operate at.

  Two orderings are measured, because a flow has two times: TIME_FIRST, when
  it began, and TIME_LAST, when it ended and its record was exported. Neither
  is assumed to be the sort key.

  The decisive statistic is DISPLACEMENT: for each flow, how many positions it
  would move if the whole stream were sorted by time. Displacement is measured
  in flow positions, not seconds, because seconds are rate dependent. Backbone
  traffic varies several-fold over a day, so the same disorder in seconds
  reaches much further in batch units at 3am than at 3pm. Positions do not
  have that problem: if the 99.9th percentile displacement is well under one
  batch, then the disorder cannot meaningfully change which flows share a
  batch, and adaptation and prediction are unaffected.

  The script also records the true number of batches per period (the existing
  pipeline never measured it: count_available_batches() is called with a cap
  of 650, so the recorded n_avail=650 is the cap, not the length of the week)
  and the exact batch index at which each dataset day begins.

  NO MODEL IS LOADED, NO ADAPTATION HAPPENS, NO LABEL IS CONSULTED.

TIME HANDLING:
  The toolchain maps TIME_ columns through datetime.fromtimestamp(), giving
  NAIVE datetimes in the running machine's LOCAL time. This script recovers
  epoch MICROSECONDS without assuming a resolution (pandas may deliver
  nanosecond, microsecond, millisecond or second resolution, and an int64
  view means something different in each case; an earlier version of this
  script assumed nanoseconds and was wrong by a factor of 1000). Sub-second
  precision is preserved rather than truncated, and the script reports
  whether any exists.
    - Every DURATION reported here is exact; a constant offset cancels.
    - Every ABSOLUTE time may be shifted from UTC by the local offset of the
      machine that ran it. Absolute times are labeled "local".
    - Day boundaries come from the dataset's own per-date tables, not from
      timestamps. The dataset's days are in the capture site's local time,
      a third timezone again.
  W-2022-44 contains a US daylight-saving transition and is warned about.

CRITERIA, all reported, none substituted for another:
    prereg       TIME_FIRST non-decreasing for >= 0.99 of flow pairs
    batch_order  per-batch median time non-decreasing for >= 0.999 of pairs
    displacement 99.9th percentile displacement < one batch (2048 positions)
  Which of these gates the streaming experiments is a decision for the
  pre-registration document, to be made BEFORE it is hash-locked.

Run:
    python scripts/14_stream_order_audit.py --size S --max-batches 20   # smoke
    python scripts/14_stream_order_audit.py --size S                    # audit
Output: streaming_order_audit.json. A capped run is marked capped and exits 2.
"""
import argparse, json, os, sys, time, warnings
import numpy as np

DATA_DIR   = "./data/CESNET-QUIC22/"
TRAIN_WEEK = "W-2022-44"
BATCH      = 256
TEST_BATCH = 2048
OUT_JSON   = "streaming_order_audit.json"
DEFAULT_PERIODS = ["W-2022-46", "W-2022-47"]
US = 1_000_000

GATE_NON_DECREASING = 0.99          # pre-registered, on TIME_FIRST
BATCH_ORDER_MIN     = 0.999
DISPLACEMENT_P999_MAX = TEST_BATCH  # one batch
REPORT_WINDOWS = [(0, 200), (200, 400), (400, 600)]

EPOCH_MIN_US = 1666915200 * US   # 2022-10-28
EPOCH_MAX_US = 1669766400 * US   # 2022-11-30
DST_PERIODS = {"W-2022-44"}
WANT_FIELDS = ["TIME_FIRST", "TIME_LAST"]


def build(size, period):
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import MM_CESNET_V2_Weights
    transforms = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44.transforms
    ds = CESNET_QUIC22(DATA_DIR, size=size)
    kw = dict(dataset=ds, apps_selection=AppSelection.ALL_KNOWN,
              train_period_name=TRAIN_WEEK, test_period_name=period,
              batch_size=BATCH, train_workers=0, test_workers=0,
              use_packet_histograms=True, return_other_fields=True,
              ppi_transform=transforms.get("ppi_transform"),
              flowstats_transform=transforms.get("flowstats_transform"),
              flowstats_phist_transform=transforms.get("flowstats_phist_transform"))
    kw = {k: v for k, v in kw.items() if v is not None}
    cfg = DatasetConfig(**kw)
    # other_fields is not part of TestDataParams, so narrowing it cannot change
    # the index cache key, which flows are selected, or their order.
    try:
        cfg.other_fields = list(WANT_FIELDS)
    except Exception as e:
        print(f"  [note] could not narrow other_fields ({e})")
    ds.set_dataset_config_and_initialize(cfg)
    return ds, ds.get_test_dataloader(), cfg


def day_map_from_indices(ds, cfg):
    from cesnet_datazoo.constants import INDICES_TABLE_FIELD
    tables = np.asarray(ds.dataset_indices.test_known_indices[INDICES_TABLE_FIELD])
    dates = list(cfg.test_dates)
    if tables.size and int(tables.max()) >= len(dates):
        raise RuntimeError("table id beyond the configured test_dates")
    counts = np.bincount(tables, minlength=len(dates)).astype("int64")
    out, cum = {}, 0
    for d, n in zip(dates, counts.tolist()):
        if n == 0:
            out[d] = {"flows": 0, "note": "no known-class flows"}
            continue
        f0, f1 = cum, cum + n - 1
        out[d] = {"flows": int(n), "first_flow_index": int(f0),
                  "last_flow_index": int(f1),
                  "first_batch_index": int(f0 // TEST_BATCH),
                  "last_batch_index": int(f1 // TEST_BATCH)}
        cum += n
    return out, int(cum)


def epoch_us(col, seen):
    """Epoch microseconds, resolution-agnostic and precision-preserving."""
    import pandas as pd
    arr = np.asarray(col.to_numpy())
    seen.add(str(arr.dtype))
    if arr.dtype.kind != "M":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = np.asarray(pd.to_datetime(pd.Series(arr)).to_numpy())
    return arr.astype("datetime64[us]").astype("int64")


def ordering(t_us, label):
    d = np.diff(t_us)
    back = -d[d < 0]
    n = len(t_us)
    out = {f"{label}_inversions": int(back.size),
           f"{label}_non_decreasing_fraction": 1.0 - back.size / max(1, n - 1)}
    if back.size:
        out[f"{label}_backward_step_seconds"] = {
            str(p): round(float(np.percentile(back, p)) / US, 3)
            for p in (50, 90, 99, 99.9, 100)}
    else:
        out[f"{label}_backward_step_seconds"] = {}
    return out, back


def displacement(t_us, label):
    """How far each flow would move under a stable global sort by time.
    Stable so that equal timestamps keep their original order and do not
    inflate the statistic."""
    order = np.argsort(t_us, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    disp = np.abs(rank - np.arange(len(order))).astype("int64")
    return {
        f"{label}_displacement_positions": {
            str(p): int(np.percentile(disp, p))
            for p in (50, 90, 99, 99.9, 100)},
        f"{label}_displacement_p999_batches": round(
            float(np.percentile(disp, 99.9)) / TEST_BATCH, 4),
        f"{label}_displacement_max_batches": round(
            float(disp.max()) / TEST_BATCH, 4),
        f"{label}_fraction_displaced_beyond_one_batch": float(
            (disp >= TEST_BATCH).mean()),
    }, disp


def audit_period(size, period, max_batches, do_day_map):
    print(f"\n=== AUDITING {period} ===")
    if period in DST_PERIODS:
        print("  [WARNING] contains a US daylight-saving transition; up to one "
              "hour of repeated local wall clock appears as backward steps.")
    ds, loader, cfg = build(size, period)

    day_map, day_total = {}, None
    if do_day_map:
        try:
            day_map, day_total = day_map_from_indices(ds, cfg)
            print(f"  day map: {len(day_map)} dates, {day_total:,} flows, "
                  f"{int(np.ceil(day_total/TEST_BATCH)):,} batches expected")
        except Exception as e:
            print(f"  [note] day map unavailable ({e})")

    tf_p, tl_p, sizes, bspan, bmed = [], [], [], [], []
    seen = set()
    t0 = time.time()
    for i, b in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        other_fields_df, x_ppi, x_flowstats, labels = b
        if "TIME_FIRST" not in other_fields_df.columns:
            sys.exit("[STOP] no TIME_FIRST column; return_other_fields did not "
                     "take effect.")
        tf = epoch_us(other_fields_df["TIME_FIRST"], seen)
        tl = (epoch_us(other_fields_df["TIME_LAST"], seen)
              if "TIME_LAST" in other_fields_df.columns else None)
        if i == 0:
            lo, hi = int(tf.min()), int(tf.max())
            if not (EPOCH_MIN_US <= lo <= EPOCH_MAX_US
                    and EPOCH_MIN_US <= hi <= EPOCH_MAX_US):
                import datetime as _d
                sys.exit("[STOP] recovered timestamps fall outside the "
                         "collection period, so the time conversion is wrong "
                         "and every ordering number would be meaningless.\n"
                         f"  first batch: "
                         f"{_d.datetime.fromtimestamp(lo/US, _d.timezone.utc)} to "
                         f"{_d.datetime.fromtimestamp(hi/US, _d.timezone.utc)}\n"
                         f"  dtypes seen: {sorted(seen)}")
            if tl is None:
                print("  [note] TIME_LAST unavailable.")
        tf_p.append(tf)
        if tl is not None:
            tl_p.append(tl)
        sizes.append(len(tf))
        bspan.append(int(tf.max() - tf.min()))
        bmed.append(int(np.median(tf)))
        if (i + 1) % 200 == 0:
            print(f"  {i+1} batches, {sum(sizes):,} flows "
                  f"({(time.time()-t0)/60:.1f}m)")

    if not tf_p:
        sys.exit(f"[STOP] {period} yielded no batches.")

    tf = np.concatenate(tf_p)
    tl = np.concatenate(tl_p) if tl_p else None
    n = len(tf)
    capped = max_batches is not None

    sub_second = int((tf % US != 0).sum())

    ord_tf, back_tf = ordering(tf, "time_first")
    ord_tl = ({}, None)
    if tl is not None:
        ord_tl, _ = ordering(tl, "time_last")
    else:
        ord_tl = {}

    disp_tf, dtf = displacement(tf, "time_first")
    disp_tl = {}
    if tl is not None:
        disp_tl, _ = displacement(tl, "time_last")

    med = np.asarray(bmed)
    binv = int((np.diff(med) < 0).sum())
    bfrac = 1.0 - binv / max(1, len(med) - 1)
    span_arr = np.asarray(bspan) / US

    import datetime as _d
    def loc(us):
        return _d.datetime.fromtimestamp(int(us) / US, _d.timezone.utc)\
                 .isoformat().replace("+00:00", "")

    cum = np.cumsum([0] + sizes)
    windows = {}
    for wi, (b0, b1) in enumerate(REPORT_WINDOWS):
        if b1 > len(sizes):
            continue
        f0, f1 = int(cum[b0]), int(cum[b1]) - 1
        windows[f"window_{wi+1}"] = {
            "batches": [b0, b1 - 1], "flows": [f0, f1],
            "start_local": loc(tf[f0]), "end_local": loc(tf[f1]),
            "span_hours": round((int(tf[f1]) - int(tf[f0])) / US / 3600.0, 2)}
    win_days = {}
    for wi, (b0, b1) in enumerate(REPORT_WINDOWS):
        f0, f1 = b0 * TEST_BATCH, b1 * TEST_BATCH - 1
        win_days[f"window_{wi+1}"] = sorted(
            d for d, di in day_map.items() if di.get("flows")
            and not (f1 < di["first_flow_index"] or f0 > di["last_flow_index"]))

    dur = ((tl - tf) / US) if tl is not None else None
    if dur is not None:
        dur = dur[dur >= 0]

    res = {
        "period": period, "size": size, "capped": capped,
        "max_batches_requested": max_batches,
        "time_column_dtypes_seen": sorted(seen),
        "sub_second_timestamps": sub_second,
        "timestamp_resolution_note": (
            "no sub-second component present; timestamps are whole seconds"
            if sub_second == 0 else
            f"{sub_second:,} of {n:,} flows carry a sub-second component"),
        "absolute_times_are": "local wall clock; may be offset from UTC. "
                              "Durations are exact.",
        "n_batches": len(sizes), "n_flows": int(n),
        "distinct_batch_sizes": sorted(set(sizes)),
        "day_map_from_indices": day_map, "day_map_total_flows": day_total,
        "report_window_days": win_days,
        "first_time_local": loc(tf[0]), "last_time_local": loc(tf[-1]),
        "span_hours": round((int(tf[-1]) - int(tf[0])) / US / 3600.0, 2),
        "batch_wall_clock_span_seconds": {
            "p50": round(float(np.percentile(span_arr, 50)), 1),
            "p90": round(float(np.percentile(span_arr, 90)), 1),
            "min": round(float(span_arr.min()), 1),
            "max": round(float(span_arr.max()), 1)},
        "batch_median_inversions": binv,
        "batch_median_in_order_fraction": bfrac,
        "flow_duration_seconds": ({str(p): round(float(np.percentile(dur, p)), 3)
                                   for p in (50, 90, 99, 99.9, 100)}
                                  if dur is not None and dur.size else {}),
        "report_windows": windows,
        "prereg_gate_threshold": GATE_NON_DECREASING,
        "batch_order_criterion": BATCH_ORDER_MIN,
        "batch_order_pass": bool(bfrac >= BATCH_ORDER_MIN),
        "displacement_criterion_positions": DISPLACEMENT_P999_MAX,
    }
    res.update(ord_tf); res.update(ord_tl); res.update(disp_tf); res.update(disp_tl)
    res["prereg_gate_pass"] = bool(
        res["time_first_non_decreasing_fraction"] >= GATE_NON_DECREASING)
    key = "time_last" if tl is not None else "time_first"
    res["displacement_pass"] = bool(
        res[f"{key}_displacement_positions"]["99.9"] < DISPLACEMENT_P999_MAX)
    res["displacement_key_used"] = key

    print(f"  dtype {', '.join(res['time_column_dtypes_seen'])}   "
          f"{res['timestamp_resolution_note']}")
    print(f"  batches / flows    : {res['n_batches']:,} / {res['n_flows']:,}")
    print(f"  span (local)       : {res['first_time_local']} to "
          f"{res['last_time_local']}  ({res['span_hours']} h)")
    bs = res["batch_wall_clock_span_seconds"]
    print(f"  one batch spans    : p50 {bs['p50']} s (min {bs['min']}, "
          f"max {bs['max']}) of wall clock")
    if day_map:
        print("  day boundaries (dataset's own per-date tables):")
        for d, di in day_map.items():
            if not di.get("flows"):
                print(f"    {d}: none"); continue
            print(f"    {d}: batches {di['first_batch_index']:>5} to "
                  f"{di['last_batch_index']:>5}, {di['flows']:>10,} flows")
        print(f"  report windows fall in: "
              + "; ".join(f"{w} -> {', '.join(v) or 'beyond period'}"
                          for w, v in win_days.items()))
    for lbl in ("time_first", "time_last"):
        if f"{lbl}_non_decreasing_fraction" not in res:
            continue
        print(f"  {lbl.upper()}")
        print(f"    non-decreasing   : "
              f"{res[f'{lbl}_non_decreasing_fraction']:.6f}"
              + (f"   [pre-registered gate {GATE_NON_DECREASING}] -> "
                 f"{'PASS' if res['prereg_gate_pass'] else 'FAIL'}"
                 if lbl == "time_first" else ""))
        pc = res[f"{lbl}_backward_step_seconds"]
        if pc:
            print(f"    backward steps s : p50 {pc['50']}, p90 {pc['90']}, "
                  f"p99 {pc['99']}, p99.9 {pc['99.9']}, max {pc['100']}")
        dp = res[f"{lbl}_displacement_positions"]
        print(f"    displacement pos : p50 {dp['50']}, p90 {dp['90']}, "
              f"p99 {dp['99']}, p99.9 {dp['99.9']}, max {dp['100']}"
              f"   (batch = {TEST_BATCH})")
        print(f"    beyond one batch : "
              f"{res[f'{lbl}_fraction_displaced_beyond_one_batch']:.2e} of flows")
    if res["flow_duration_seconds"]:
        pc = res["flow_duration_seconds"]
        print(f"  flow duration s    : p50 {pc['50']}, p90 {pc['90']}, "
              f"p99 {pc['99']}, max {pc['100']}")
    print(f"  BATCH ORDER        : per-batch median in order for "
          f"{bfrac:.6f} of pairs -> "
          f"{'PASS' if res['batch_order_pass'] else 'FAIL'} "
          f"[{BATCH_ORDER_MIN}]")
    print(f"  DISPLACEMENT       : p99.9 on {key} = "
          f"{res[f'{key}_displacement_positions']['99.9']} positions "
          f"({res[f'{key}_displacement_p999_batches']} batches) -> "
          f"{'PASS' if res['displacement_pass'] else 'FAIL'} "
          f"[< {DISPLACEMENT_P999_MAX}]")
    for w, wi in windows.items():
        print(f"  {w}: batches {wi['batches'][0]}-{wi['batches'][1]}, "
              f"{wi['span_hours']} h, {wi['start_local']} to {wi['end_local']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--periods", default=",".join(DEFAULT_PERIODS))
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--no-day-map", action="store_true")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    out = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            out = json.load(f)
        print(f"[RESUME] {args.out} has: {', '.join(out)}")

    for period in [p.strip() for p in args.periods.split(",") if p.strip()]:
        if period in out and not out[period].get("capped"):
            print(f"\n=== {period} already audited (uncapped); skipping ===")
            continue
        out[period] = audit_period(args.size, period, args.max_batches,
                                   not args.no_day_map)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)

    print("\n==== A0 SUMMARY ====")
    capped, fails = [], []
    for period, r in out.items():
        if r.get("capped"):
            capped.append(period)
        bad = [k for k in ("batch_order_pass", "displacement_pass")
               if r.get(k) is False]
        if bad:
            fails.append(f"{period} ({', '.join(bad)})")
        k = r.get("displacement_key_used", "time_first")
        print(f"  {period}: {r['n_batches']:,} batches, {r['n_flows']:,} flows"
              f"{'   [CAPPED, not the audit]' if r.get('capped') else ''}")
        print(f"      prereg TIME_FIRST {r['time_first_non_decreasing_fraction']:.6f} "
              f"{'PASS' if r['prereg_gate_pass'] else 'FAIL'}"
              f"   batch order {r['batch_median_in_order_fraction']:.6f} "
              f"{'PASS' if r['batch_order_pass'] else 'FAIL'}"
              f"   displacement p99.9 {r[f'{k}_displacement_positions']['99.9']} "
              f"{'PASS' if r['displacement_pass'] else 'FAIL'}")
    print(f"\n  written to {args.out}")

    if capped:
        print(f"\n  [CAPPED RUN] {', '.join(capped)}. Smoke test, not the audit.")
        sys.exit(2)
    if fails:
        print(f"\n  [CRITERIA FAILED] {'; '.join(fails)}")
        print("  The stream does not preserve batch composition. The streaming "
              "design must be re-specified before any streaming number is "
              "generated.")
        sys.exit(1)
    print("\n  [BATCH-RESOLUTION CRITERIA PASSED]")
    if any(not r["prereg_gate_pass"] for r in out.values()):
        print("  The pre-registered flow-level gate did NOT pass. Read it with "
              "the displacement result: the stream is chronological at batch "
              "resolution with sub-batch jitter, which is a different claim "
              "from flow-exact sorting and is the claim the experiments need. "
              "Whether the gate is restated accordingly is a decision for the "
              "pre-registration, to be made before it is hash-locked.")
        sys.exit(1)


if __name__ == "__main__":
    main()
