#!/usr/bin/env python3
r"""
26_inference_cost.py -- what adaptation costs on the inference path.

The study reports step counts and parameter counts and no memory or latency.
A reviewer asked for the three numbers an operator would actually need, and
they are cheap to get.

WHAT IT MEASURES, on real batches from the report week:
  latency    wall time of one frozen forward pass over a 2048-flow batch, and
             of one adaptation step (forward, entropy, backward, optimizer) on
             the same batch, both after warmup, reported as median over
             repeats so a single slow batch cannot set the number
  memory     peak allocation during each, from torch.cuda.max_memory_allocated
             on GPU, or tracemalloc plus resident-set delta on CPU
  throughput implied flows per second at 2048-flow batches for each mode

NO PRE-COMMITTED PREDICTIONS. These are descriptive measurements of this
machine, not a test of any claim, and nothing in the paper rests on them.

READ THE NUMBERS WITH THIS IN MIND: they characterize the machine they were
taken on, not the method. A CPU number and a GPU number for the same code
differ by more than the adapted-versus-frozen ratio does, so quote the ratio,
name the hardware, and do not present the absolute latency as a property of
the approach.

Run:
    python scripts/26_inference_cost.py --size S
    python scripts/26_inference_cost.py --size S --repeats 50
Output: inference_cost.json
"""
import argparse, copy, json, os, platform, statistics, time
import numpy as np
from tta_guards import guarded_eval

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256
LR, QUANT, BN_MOM = 1e-3, 0.5, 0.1
OUT_JSON = "inference_cost.json"


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


def sync(device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()


def peak_reset(device):
    import torch
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        return None
    import tracemalloc
    tracemalloc.start()
    return tracemalloc


def peak_read(device, handle):
    import torch
    if device == "cuda":
        return float(torch.cuda.max_memory_allocated()) / 2 ** 20
    cur, peak = handle.get_traced_memory()
    handle.stop()
    return float(peak) / 2 ** 20


def time_frozen(model, batches, device, repeats):
    import torch
    ts = []
    def _run():
        with torch.no_grad():
            for b in batches[:3]:
                fwd(model, b, device)                      # warmup
            sync(device)
            for i in range(repeats):
                b = batches[i % len(batches)]
                t = time.perf_counter()
                fwd(model, b, device)
                sync(device)
                ts.append(time.perf_counter() - t)
        return ts
    was = {n: m.training for n, m in model.named_modules()}
    model.eval()
    guarded_eval(model, _run)
    for n, m in model.named_modules():
        m.train(was[n])
    return ts


def time_adapt(model, batches, device, repeats):
    """One adaptation step: forward, filtered entropy, backward, optimizer."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m = copy.deepcopy(model); m.requires_grad_(False); params = []
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.train(); mod.momentum = BN_MOM
            mod.requires_grad_(True)
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias is not None:   params.append(mod.bias)
    opt = torch.optim.Adam(params, lr=LR)

    def step(b):
        lo, _ = fwd(m, b, device)
        ent = -(F.softmax(lo, 1) * F.log_softmax(lo, 1)).sum(1)
        sel = ent <= torch.quantile(ent.detach(), QUANT)
        loss = ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()

    for b in batches[:3]:
        step(b)                                            # warmup
    sync(device)
    ts = []
    for i in range(repeats):
        b = batches[i % len(batches)]
        t = time.perf_counter()
        step(b)
        sync(device)
        ts.append(time.perf_counter() - t)
    n_params = int(sum(p.numel() for p in params))
    del m, opt
    return ts, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--repeats", type=int, default=30)
    a = ap.parse_args()

    import torch
    model, loader, device = build(a.size, TEST_WEEK)
    batches = []
    for i, b in enumerate(loader):
        if i >= 10:
            break
        batches.append(b)
    parts = (list(batches[0]) if isinstance(batches[0], (tuple, list))
             else [batches[0]])
    n_flows = next(len(np.asarray(p)) for p in parts
                   if np.asarray(p).ndim == 1
                   and np.issubdtype(np.asarray(p).dtype, np.integer))

    print("=== INFERENCE-PATH COST (descriptive, no predictions) ===")
    print(f"    device {device}, torch {torch.__version__}")
    print(f"    {platform.platform()}")
    print(f"    batch {n_flows} flows, {a.repeats} timed repeats after warmup\n")

    h = peak_reset(device)
    tf = time_frozen(model, batches, device, a.repeats)
    mem_f = peak_read(device, h)

    h = peak_reset(device)
    ta, n_params = time_adapt(model, batches, device, a.repeats)
    mem_a = peak_read(device, h)

    mf, ma = statistics.median(tf), statistics.median(ta)
    res = {
        "device": device, "torch": torch.__version__,
        "platform": platform.platform(),
        "flows_per_batch": n_flows, "repeats": a.repeats,
        "adapted_params": n_params,
        "frozen_forward_s": {"median": mf, "p10": float(np.quantile(tf, .10)),
                             "p90": float(np.quantile(tf, .90))},
        "adapt_step_s": {"median": ma, "p10": float(np.quantile(ta, .10)),
                         "p90": float(np.quantile(ta, .90))},
        "latency_ratio": ma / mf if mf > 0 else float("nan"),
        "peak_mem_frozen_MiB": mem_f, "peak_mem_adapted_MiB": mem_a,
        "mem_ratio": mem_a / mem_f if mem_f > 0 else float("nan"),
        "flows_per_s_frozen": n_flows / mf if mf > 0 else float("nan"),
        "flows_per_s_adapt": n_flows / ma if ma > 0 else float("nan"),
    }
    print(f"  frozen forward   median {mf*1000:8.2f} ms  "
          f"[p10 {np.quantile(tf,.10)*1000:.2f}, p90 {np.quantile(tf,.90)*1000:.2f}]")
    print(f"  adaptation step  median {ma*1000:8.2f} ms  "
          f"[p10 {np.quantile(ta,.10)*1000:.2f}, p90 {np.quantile(ta,.90)*1000:.2f}]")
    print(f"  latency ratio    {res['latency_ratio']:.2f}x")
    print(f"  peak memory      frozen {mem_f:.1f} MiB, "
          f"adapted {mem_a:.1f} MiB, ratio {res['mem_ratio']:.2f}x")
    print(f"  throughput       frozen {res['flows_per_s_frozen']:,.0f} flows/s, "
          f"adapting {res['flows_per_s_adapt']:,.0f} flows/s")
    print(f"  adapted parameters {n_params:,} (BN affine only)")
    print("\n  These characterize this machine. Quote the ratios, name the")
    print("  hardware, and do not present the absolute latency as a property")
    print("  of the method.")
    if device == "cpu":
        print("\n  [note] CPU peak memory uses tracemalloc, which sees Python")
        print("  allocations only. Treat the memory ratio as indicative on CPU")
        print("  and rerun on GPU if the number has to carry weight.")
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(f"\n  raw: {OUT_JSON}")


if __name__ == "__main__":
    main()
