#!/usr/bin/env python3
r"""
diagnose_compute_determinism.py — is the ADAPTATION compute path itself
run-to-run deterministic, given the loader is already proven deterministic?

We now know (from diagnose_nondeterminism.py) the loader is bit-identical
across builds. So this script isolates the remaining question: does running
the exact same tent() adaptation twice, in one process, on the same window,
produce the same recovery?

If YES twice -> no run-to-run nondeterminism exists; the earlier "grid shift"
   was purely old-code vs new-code, nothing to seed-control.
If NO -> there is genuine nondeterminism in deepcopy/BN/threading, and we
   locate it before building error bars.

We test the single config that matters (the frozen one) plus one 5e-3 config,
since the 5e-3 row was the one that swung hardest (mixed -> all zeros).

Reuses the main script's own functions so we're testing the REAL path.
"""
import sys, importlib.util
import numpy as np

MAIN = "/mnt/user-data/outputs/leakage_clean_tent.py"

def load_main():
    spec = importlib.util.spec_from_file_location("lct", MAIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def run_once(m, size, lr, steps, q, seed=None):
    import torch
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    model, loader, device = m.build(size, m.VAL_WEEK)
    window = m.collect_window(loader, skip=0, n=m.TUNE_EVAL_BATCHES)
    fr, ad = m.tent(model, window, device, lr, steps, q)
    return fr, ad - fr

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="XS")
    args = ap.parse_args()
    m = load_main()

    configs = [
        (1e-3, 100, 0.5),   # the frozen config
        (5e-3, 100, 0.5),   # the one that swung hardest between runs
    ]

    print(f"size={args.size} — running each config TWICE in one process, no seeding\n")
    print(f"{'lr':>7} {'steps':>5} {'q':>4} | {'run A rec':>10} {'run B rec':>10} {'delta':>8}")
    print("-"*52)
    any_diff = False
    for lr, steps, q in configs:
        frA, recA = run_once(m, args.size, lr, steps, q)
        frB, recB = run_once(m, args.size, lr, steps, q)
        d = (recB - recA)*100
        if abs(d) > 1e-9: any_diff = True
        print(f"{lr:>7.0e} {steps:>5} {q:>4.1f} | {recA*100:>+9.3f}p {recB*100:>+9.3f}p {d:>+7.3f}p")

    print("\n=== now the same two, WITH seed=0 both times ===")
    print(f"{'lr':>7} {'steps':>5} {'q':>4} | {'run A rec':>10} {'run B rec':>10} {'delta':>8}")
    print("-"*52)
    for lr, steps, q in configs:
        frA, recA = run_once(m, args.size, lr, steps, q, seed=0)
        frB, recB = run_once(m, args.size, lr, steps, q, seed=0)
        d = (recB - recA)*100
        print(f"{lr:>7.0e} {steps:>5} {q:>4.1f} | {recA*100:>+9.3f}p {recB*100:>+9.3f}p {d:>+7.3f}p")

    print("\n================ VERDICT ================")
    if not any_diff:
        print("Adaptation is fully deterministic run-to-run (no-seed deltas = 0).")
        print("=> There is NO run-to-run nondeterminism. The grid 'shift' was")
        print("   entirely old-code vs new-code. Nothing to seed-control.")
        print("   Step 2 'error bars' must come from a DELIBERATE varied factor")
        print("   (e.g. different windows, or injected batch-order shuffling),")
        print("   NOT from hoping the pipeline is noisy. I'll design that.")
    else:
        print("Adaptation DIFFERS run-to-run with no seed.")
        print("=> Genuine nondeterminism (deepcopy/BN/threading). Check whether")
        print("   the seed=0 block above zeroed the deltas: if yes, seeding")
        print("   controls it; if no, it's float-threading (set torch threads=1).")
    print("========================================")

if __name__ == "__main__":
    main()
