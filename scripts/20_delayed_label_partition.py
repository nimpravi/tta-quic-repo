#!/usr/bin/env python3
r"""
20_delayed_label_partition.py -- POST-HOC extension of Experiment C2.

THIS IS NOT PRE-REGISTERED. Say so wherever it is reported.

  The locked pre-registration specifies the affected/unaffected partition
  analysis for the label-free conditions only (Section 5, C2). This script
  applies the same partition to the supervised delayed-label conditions of
  Experiment B. It was written after C2 showed that label-free adaptation
  gains about seven points on drifted traffic while destroying two and a half
  to five and a half points on traffic that did not drift, which makes the
  same question about the supervised baselines unavoidable: does retraining
  on delayed labels carry the same collateral cost, or not?

  Nothing here can change a pre-registered outcome. B-K1 has already fired,
  the partition is already hash-locked, and every configuration used below was
  frozen before W-2022-47 was touched. What this adds is a decomposition of
  numbers that already exist. It is reported in its own subsection, labeled
  post-hoc, and its result is not used to revise any kill-rule verdict.

WHAT IT COMPUTES:
  For each label delay and each capacity, the change in accuracy on the
  affected and unaffected class partitions of the three W-2022-47 report
  windows, alongside the label-free controls that share the same source day.

VERIFICATION BUILT IN:
  1. class_partition.json is checked against class_partition.sha256; a
     mismatch aborts.
  2. Window 1 frozen accuracy is asserted against 0.72239013671875.
  3. The per-partition frozen baselines are cross-checked against
     nondrifted_c2_progress.json when that file is present, so the two runs
     are tied together rather than merely consistent-looking.
  4. Every row's support-weighted net is printed beside the recorded
     Experiment B number for the same condition. The decomposition must
     reconstruct the recorded value; if it does not, something is wrong and
     the output says so rather than the reader having to notice.

Run:
    python scripts/20_delayed_label_partition.py --size S
    python scripts/20_delayed_label_partition.py --size S --combined
Resumable via delayed_label_partition_progress.json.
"""
import argparse, copy, datetime as dt, hashlib, json, os, sys, time
import numpy as np
from tta_guards import guarded_eval, assert_anchor

DATA_DIR   = "./data/CESNET-QUIC22/"
MODEL_DIR  = "./models/"
TRAIN_WEEK = "W-2022-44"
TEST_WEEK  = "W-2022-47"
BATCH      = 256

EVAL_DAY   = "20221121"
DELTAS     = [1, 3, 7]
POOL_BATCHES = 200
REPORT_WINDOWS = [(0, 200), (200, 400), (400, 600)]
BN_MOM     = 0.1
TTA_LR, TTA_STEPS, TTA_QUANT = 1e-3, 50, 0.5   # frozen episodic config

PART_JSON = "class_partition.json"
PART_SHA  = "class_partition.sha256"
CONFIG_JSON = "delayed_label_config.json"
C2_CKPT   = "nondrifted_c2_progress.json"
B_CKPT    = "delayed_label_progress.json"
CKPT      = "delayed_label_partition_progress.json"

ANCHOR_W47_W1 = 0.72239013671875
# Recorded Experiment B per-window recoveries, for the reconstruction check.
RECORDED_B = {
 (1,"src-stats"):[2.17,2.07,2.38], (1,"src-tent"):[3.05,2.78,2.67],
 (1,"matched"):[14.11,13.41,13.13],(1,"head"):[11.54,10.86,10.82],
 (1,"full"):[14.90,14.03,13.68],   (1,"matched+tta"):[12.52,12.43,11.38],
 (3,"src-stats"):[2.65,2.63,2.95], (3,"src-tent"):[3.25,2.90,2.60],
 (3,"matched"):[14.04,13.34,12.89],(3,"head"):[11.86,11.14,10.84],
 (3,"full"):[15.55,14.65,13.95],   (3,"matched+tta"):[12.38,12.15,11.33],
 (7,"src-stats"):[2.19,1.92,2.36], (7,"src-tent"):[3.00,2.52,2.46],
 (7,"matched"):[12.57,11.93,11.61],(7,"head"):[11.85,11.09,10.70],
 (7,"full"):[14.27,13.39,12.88],   (7,"matched+tta"):[11.07,11.09,10.15]}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()


def shift_day(day, delta):
    return (dt.datetime.strptime(day,"%Y%m%d").date()
            - dt.timedelta(days=delta)).strftime("%Y%m%d")


def build(size, period_name, dates=None):
    import torch
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection
    from cesnet_models.models import mm_cesnet_v2, MM_CESNET_V2_Weights
    w = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
    model = mm_cesnet_v2(weights=w, model_dir=MODEL_DIR); model.eval()
    tr = w.transforms
    ds = CESNET_QUIC22(DATA_DIR, size=size)
    kw = dict(dataset=ds, apps_selection=AppSelection.ALL_KNOWN,
              train_period_name=TRAIN_WEEK, test_period_name=period_name,
              batch_size=BATCH, train_workers=0, test_workers=0,
              use_packet_histograms=True,
              ppi_transform=tr.get("ppi_transform"),
              flowstats_transform=tr.get("flowstats_transform"),
              flowstats_phist_transform=tr.get("flowstats_phist_transform"))
    if dates is not None: kw["test_dates"] = list(dates)
    kw = {k:v for k,v in kw.items() if v is not None}
    cfg = DatasetConfig(**kw); ds.set_dataset_config_and_initialize(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, ds.get_test_dataloader(), device


def fwd(model, batch, device):
    import torch
    parts = list(batch) if isinstance(batch,(tuple,list)) else [batch]
    ppi=fs=y=None
    for p in parts:
        a=np.asarray(p)
        if a.ndim==3: ppi=a
        elif a.ndim==2 and a.shape[1]>0: fs=a
        elif a.ndim==1 and np.issubdtype(a.dtype,np.integer): y=a
    if ppi is None or fs is None or y is None:
        raise RuntimeError("batch parse failed")
    return model((torch.as_tensor(ppi).float().to(device),
                  torch.as_tensor(fs).float().to(device))), y


def collect(loader, skip, n, label=""):
    out=[]
    for i,b in enumerate(loader):
        if i<skip: continue
        out.append(b)
        if len(out)>=n: break
    if len(out)<n:
        print(f"  [WARN]{' '+label if label else ''} wanted {n} from {skip}, got {len(out)}")
    return out


def predict(model, batches, device):
    import torch
    def _run():
        was={n:m.training for n,m in model.named_modules()}
        model.eval(); ys,ps=[],[]
        with torch.no_grad():
            for b in batches:
                lo,y=fwd(model,b,device)
                ps.append(lo.argmax(1).cpu().numpy()); ys.append(y)
        for n,m in model.named_modules(): m.train(was[n])
        return np.concatenate(ys), np.concatenate(ps)
    return guarded_eval(model,_run)


def find_head(m):
    import torch.nn as nn
    last=None
    for name,mod in m.named_modules():
        if isinstance(mod,nn.Linear): last=(name,mod)
    if last is None: raise RuntimeError("no Linear layer found")
    return last


def retrain(base, pool, device, capacity, lr, steps, order):
    """Identical to 17_delayed_label.py. Supervised capacities use labels;
    src-stats and src-tent use none."""
    import torch, torch.nn as nn, torch.nn.functional as F
    m=copy.deepcopy(base); m.requires_grad_(False); params=[]
    if capacity in ("src-stats","src-tent"):
        for mod in m.modules():
            if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
                mod.train(); mod.momentum=BN_MOM
                if capacity=="src-tent":
                    mod.requires_grad_(True)
                    if mod.weight is not None: params.append(mod.weight)
                    if mod.bias is not None:   params.append(mod.bias)
        if capacity=="src-stats":
            with torch.no_grad():
                for s in range(steps): fwd(m, pool[order[s%len(order)]], device)
            return m
        opt=torch.optim.Adam(params,lr=lr)
        for s in range(steps):
            lo,_=fwd(m,pool[order[s%len(order)]],device)
            ent=-(F.softmax(lo,1)*F.log_softmax(lo,1)).sum(1)
            sel=ent<=torch.quantile(ent.detach(),TTA_QUANT)
            loss=ent[sel].mean() if sel.any() else ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return m
    if capacity=="matched":
        for mod in m.modules():
            if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
                mod.requires_grad_(True); mod.train(); mod.momentum=BN_MOM
                if mod.weight is not None: params.append(mod.weight)
                if mod.bias is not None:   params.append(mod.bias)
    elif capacity=="head":
        _,head=find_head(m); head.requires_grad_(True)
        params=list(head.parameters())
    elif capacity=="full":
        m.train(); m.requires_grad_(True)
        for mod in m.modules():
            if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)): mod.momentum=BN_MOM
        params=[p for p in m.parameters() if p.requires_grad]
    else:
        raise ValueError(capacity)
    opt=torch.optim.Adam(params,lr=lr)
    for s in range(steps):
        lo,y=fwd(m,pool[order[s%len(order)]],device)
        loss=F.cross_entropy(lo,torch.as_tensor(y).long().to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def tta_on_window(model, window, device, order):
    import torch, torch.nn as nn, torch.nn.functional as F
    m=copy.deepcopy(model); m.requires_grad_(False); params=[]
    for mod in m.modules():
        if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
            mod.requires_grad_(True); mod.train(); mod.momentum=BN_MOM
            if mod.weight is not None: params.append(mod.weight)
            if mod.bias is not None:   params.append(mod.bias)
    opt=torch.optim.Adam(params,lr=TTA_LR)
    for s in range(TTA_STEPS):
        lo,_=fwd(m,window[order[s%len(order)]],device)
        ent=-(F.softmax(lo,1)*F.log_softmax(lo,1)).sum(1)
        sel=ent<=torch.quantile(ent.detach(),TTA_QUANT)
        loss=ent[sel].mean() if sel.any() else ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def part_acc(y, p, mask):
    return float((y[mask]==p[mask]).mean()) if mask.any() else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--K", type=int, default=1,
                    help="orderings per unit; 1 is enough for a descriptive "
                         "post-hoc decomposition and is declared as such")
    ap.add_argument("--combined", action="store_true",
                    help="also decompose matched+tta, which shows whether "
                         "stacking adaptation re-introduces the collateral cost")
    ap.add_argument("--capacities",
                    default="src-stats,src-tent,matched,head,full")
    ap.add_argument("--deltas", default="1,3,7")
    a=ap.parse_args()

    print("=" * 72)
    print("POST-HOC ANALYSIS. NOT PRE-REGISTERED.")
    print("Reported in its own subsection, labeled post-hoc. It decomposes")
    print("numbers that already exist and revises no kill-rule verdict.")
    print("=" * 72 + "\n")

    for f in (PART_JSON, PART_SHA, CONFIG_JSON):
        if not os.path.exists(f): sys.exit(f"[STOP] {f} not found.")
    rec=open(PART_SHA).read().split()[0].strip().lower()
    act=sha256_file(PART_JSON)
    if rec!=act:
        sys.exit(f"[STOP] {PART_JSON} does not match {PART_SHA}.\n"
                 f"  recorded {rec}\n  actual   {act}")
    part=json.load(open(PART_JSON))
    aff=np.array(part["affected"]); una=np.array(part["unaffected"])
    caps=json.load(open(CONFIG_JSON))["per_capacity"]
    print(f"partition verified ({act[:16]}...): {len(aff)} affected, "
          f"{len(una)} unaffected classes")
    for c, b in caps.items():
        lrs = "n/a" if b["lr"] is None else f"{b['lr']:.0e}"
        print(f"  {c:>9}: lr={lrs}, steps={b['steps']}")
    print()

    ck=json.load(open(CKPT)) if os.path.exists(CKPT) else {"done":{}}
    if ck["done"]: print(f"[RESUME] {len(ck['done'])} units done\n")

    base,loader,device=build(a.size,TEST_WEEK)
    print(f"device={device}")
    windows=[collect(loader,s,e-s,label=f"window {i+1}")
             for i,(s,e) in enumerate(REPORT_WINDOWS)]
    fro=[]
    for i,w in enumerate(windows):
        y,p=predict(base,w,device)
        ma=np.isin(y,aff); mu=np.isin(y,una)
        rec_i={"overall":float((y==p).mean()),
               "affected":part_acc(y,p,ma),"unaffected":part_acc(y,p,mu),
               "n_affected":int(ma.sum()),"n_unaffected":int(mu.sum()),
               "mask_aff":ma,"mask_una":mu}
        if i==0:
            assert_anchor(rec_i["overall"],ANCHOR_W47_W1,tol=0.0,
                          name="W-47 window 1 frozen (Table I anchor)")
            print(f"  [ANCHOR OK] window 1 frozen = {rec_i['overall']!r}")
        fro.append(rec_i)
    if os.path.exists(C2_CKPT):
        c2=json.load(open(C2_CKPT))["done"]
        ok=True
        for i in range(3):
            b=c2.get(f"w{i}_frozen")
            if not b: continue
            for k in ("affected","unaffected","n_affected","n_unaffected"):
                if abs(b[k]-fro[i][k])>1e-12: ok=False
        print(f"  [CROSS-CHECK] per-partition frozen baselines "
              f"{'MATCH' if ok else 'DO NOT MATCH'} {C2_CKPT}")
        if not ok: sys.exit("[STOP] the two runs disagree on the frozen "
                            "baselines. Establish why before recording.")
    for i,f in enumerate(fro):
        print(f"  window {i+1} frozen: affected {f['affected']:.4f} "
              f"({f['n_affected']:,}), unaffected {f['unaffected']:.4f} "
              f"({f['n_unaffected']:,})")
    print()

    deltas=[int(x) for x in a.deltas.split(",") if x.strip()]
    capnames=[c.strip() for c in a.capacities.split(",") if c.strip()]
    for delta in deltas:
        src=shift_day(EVAL_DAY,delta)
        print(f"--- delta = {delta} day(s): labels from {src} ---")
        _,sload,_=build(a.size,f"DAY-{src}",dates=[src])
        pool=collect(sload,0,POOL_BATCHES,label=f"source {src}")
        print(f"  pool: {len(pool)} batches ({len(pool)*2048:,} flows)")
        todo=list(capnames)+(["matched+tta"] if a.combined else [])
        for cap in todo:
            for k in range(a.K):
                key=f"d{delta}_{cap}_{k}"
                if key in ck["done"]: continue
                t0=time.time()
                basecap = cap.replace("+tta","")
                cfg=caps[basecap]
                order=list(np.random.default_rng(1000*delta+k).permutation(len(pool)))
                m=retrain(base,pool,device,basecap,cfg["lr"],cfg["steps"],order)
                rows=[]
                for i,w in enumerate(windows):
                    mm=m
                    if cap.endswith("+tta"):
                        wo=list(np.random.default_rng(1000*i).permutation(len(w)))
                        mm=tta_on_window(m,w,device,wo)
                    y,p=predict(mm,w,device)
                    rows.append({
                        "overall":float((y==p).mean()),
                        "affected":part_acc(y,p,fro[i]["mask_aff"]),
                        "unaffected":part_acc(y,p,fro[i]["mask_una"])})
                    if mm is not m: del mm
                ck["done"][key]={"delta":delta,"capacity":cap,"k":k,
                                 "source_day":src,"windows":rows}
                json.dump(ck,open(CKPT,"w"),indent=1)
                del m
                da=[(rows[i]["affected"]-fro[i]["affected"])*100 for i in range(3)]
                du=[(rows[i]["unaffected"]-fro[i]["unaffected"])*100 for i in range(3)]
                print(f"    {cap:>12} k={k}: affected "
                      + "/".join(f"{v:+.2f}" for v in da)
                      + "   unaffected " + "/".join(f"{v:+.2f}" for v in du)
                      + f"  ({(time.time()-t0)/60:.1f}m)")
        del pool
        print()

    bprog = None
    if os.path.exists(B_CKPT):
        bprog = json.load(open(B_CKPT)).get("done", {})
        print(f"  comparison values taken from {B_CKPT}, matched unit for unit")
    else:
        print(f"  [note] {B_CKPT} not found; comparing against the recorded "
              f"K=3 means, which will flag a K mismatch as CHECK")

    print("==== POST-HOC PARTITION DECOMPOSITION (NOT PRE-REGISTERED) ====")
    print("  change in accuracy on the three W-2022-47 report windows\n")
    print(f"  {'condition':<28}{'affected':>10}{'unaffected':>12}"
          f"{'net':>8}{'recorded':>10}{'check':>7}")
    print("  " + "-"*75)
    for delta in deltas:
        for cap in capnames+(["matched+tta"] if a.combined else []):
            keys=[k for k,v in ck["done"].items()
                  if v["delta"]==delta and v["capacity"]==cap]
            if not keys: continue
            das,dus,nets=[],[],[]
            for i in range(3):
                da=np.mean([ck["done"][k]["windows"][i]["affected"] for k in keys])
                du=np.mean([ck["done"][k]["windows"][i]["unaffected"] for k in keys])
                f=fro[i]["n_affected"]/(fro[i]["n_affected"]+fro[i]["n_unaffected"])
                da=(da-fro[i]["affected"])*100; du=(du-fro[i]["unaffected"])*100
                das.append(da); dus.append(du); nets.append(f*da+(1-f)*du)
            # Compare against the SAME ordering units this run used. The
            # recorded Experiment B means are over K=3 for several capacities
            # while this run is K=1, so comparing against those means would
            # flag a K mismatch as a discrepancy.
            rm = float("nan")
            if bprog is not None:
                per = [np.mean(bprog[f"d{delta}_{cap}_{k}"]["recoveries"])
                       for k in range(a.K)
                       if f"d{delta}_{cap}_{k}" in bprog]
                if per: rm = float(np.mean(per))
            if np.isnan(rm):
                r = RECORDED_B.get((delta, cap))
                rm = np.mean(r) if r else float("nan")
            ok="ok" if r and abs(np.mean(nets)-rm)<0.15 else ("CHECK" if r else "")
            lf=" [label-free]" if cap in ("src-stats","src-tent") else ""
            print(f"  {cap+', d='+str(delta)+'d':<28}{np.mean(das):10.2f}"
                  f"{np.mean(dus):12.2f}{np.mean(nets):8.2f}{rm:10.2f}{ok:>7}{lf}")
    print("\n  'net' is the support-weighted combination of the two partitions.")
    print("  It must reconstruct the Experiment B value for the SAME ordering")
    print("  units; 'CHECK' marks any row where it does not, within 0.15 points.")
    print("\n  For comparison, the pre-registered C2 result on the same windows:")
    print("    label-free stats-only    affected +7.18   unaffected -4.89")
    print("    label-free filtered      affected +6.87   unaffected -2.79")
    print(f"\n  raw: {CKPT}")


if __name__=="__main__":
    main()
