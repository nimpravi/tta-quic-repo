#!/usr/bin/env python3
"""
Repopulate the cached class names in 27_perclass_momentum.py's checkpoint.

27_perclass_momentum.py caches `class_names` in its checkpoint on the first
full run and reuses it for every later --report. The version that produced
the current checkpoint probed ds.known_apps / app_names / classes / labels,
none of which cesnet-datazoo defines, so it fell back to class_0 ... class_102.

class_partition.json already carries the real names for all 103 slots, so the
cache can be refilled from it without loading the dataset or the model, and
without re-running a single adaptation unit.

    python scripts/fix27_cache_class_names.py                # dry run
    python scripts/fix27_cache_class_names.py --write

Then `python scripts/27_perclass_momentum.py ... --report` prints service
names. Fix class_names() in script 27 as well, or the next full run will
overwrite the cache with indices again.
"""
import argparse, json, os, shutil, sys

SEARCH = [".", "results/raw", "results", "results/superseded"]
CKPT = "perclass_momentum_progress.json"
PART = "class_partition.json"


def find(name):
    for d in SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the checkpoint (default is a dry run)")
    a = ap.parse_args()

    cp, pp = find(CKPT), find(PART)
    if cp is None or pp is None:
        sys.exit(f"[STOP] not found: {CKPT if cp is None else ''} "
                 f"{PART if pp is None else ''} (searched {', '.join(SEARCH)})")

    part = json.load(open(pp))["classes"]
    n = len(part)
    names = [None] * n
    for k, v in part.items():
        names[int(k)] = v["name"]
    if any(x is None for x in names):
        sys.exit("[STOP] class_partition.json has gaps in its class indices")

    ck = json.load(open(cp))
    old = ck.get("class_names")
    print(f"  checkpoint {cp}")
    print(f"  partition  {pp}  ({n} classes)")
    print(f"  cached now: {'absent' if old is None else old[:3]} ...")
    print(f"  would set : {names[:3]} ... {names[-1]}")
    for i in (66, 101, 61, 62, 41):
        print(f"    {i:3d} -> {names[i]}")

    if not a.write:
        print("\n  dry run, nothing written. Re-run with --write.")
        return

    shutil.copyfile(cp, cp + ".bak")
    ck["class_names"] = names
    with open(cp, "w") as f:
        json.dump(ck, f)
    print(f"\n  written. Previous file kept at {cp}.bak")


if __name__ == "__main__":
    main()
