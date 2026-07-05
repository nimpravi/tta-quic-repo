#!/usr/bin/env python3
r"""
tta_guards.py -- minimal state-audit protocol for TTA experiments.

MOTIVATION:
  This project's original pipeline was corrupted by an accuracy probe
  that called model.eval() mid-adaptation and never restored training
  mode, silently changing the algorithm (see results/RESULTS.md §7.1).
  Test-time adaptation changes model state by design, which makes it
  unusually easy for a measurement to become an intervention. These
  guards make the safeguard infrastructure rather than advice: every
  evaluation proves it did not mutate the model.

USAGE:
    from tta_guards import guarded_eval, state_fingerprint, assert_anchor

    # Wrap ANY evaluation of a model that will continue adapting:
    acc = guarded_eval(model, lambda: accuracy_on_batches(model, w, dev))

    # Assert a known bit-level anchor at runtime:
    assert_anchor(frozen_acc, 0.72239013671875, tol=0.0, name="W-47 w1 frozen")

WHAT guarded_eval CHECKS (before vs after the evaluation callable):
  1. The training flag of every module (model.eval() side effects).
  2. A checksum of every BatchNorm running_mean / running_var /
     num_batches_tracked buffer (statistic mutation).
  3. A checksum of every parameter tensor (weight mutation).
  On any mismatch it raises StateMutationError naming the offending
  module, instead of letting a distorted number into the record.
"""
import hashlib


class StateMutationError(RuntimeError):
    """An evaluation mutated model state it had no business touching."""


def _tensor_digest(t):
    import torch
    with torch.no_grad():
        return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def state_fingerprint(model):
    """Training flags + checksums of all BN buffers and all parameters."""
    import torch.nn as nn
    fp = {"modes": {}, "buffers": {}, "params": {}}
    for name, mod in model.named_modules():
        fp["modes"][name] = mod.training
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for bname, buf in mod.named_buffers(recurse=False):
                if buf is not None:
                    fp["buffers"][f"{name}.{bname}"] = _tensor_digest(buf)
    for pname, p in model.named_parameters():
        fp["params"][pname] = _tensor_digest(p)
    return fp


def _diff(before, after):
    problems = []
    for name, mode in before["modes"].items():
        if after["modes"].get(name) != mode:
            problems.append(f"training flag changed on '{name or '<root>'}': "
                            f"{mode} -> {after['modes'].get(name)}")
    for key in ("buffers", "params"):
        for name, digest in before[key].items():
            if after[key].get(name) != digest:
                problems.append(f"{key[:-1]} mutated: '{name}'")
    return problems


def guarded_eval(model, eval_callable, restore=False):
    """Run eval_callable() and PROVE the model came back unchanged.

    restore=False (default, strict): raise StateMutationError on any
      mutation. This is the right setting for pipelines, where a
      mutation means the surrounding code is wrong and the number
      about to be recorded is not what it claims to be.
    restore=True (lenient): restore the training flags that changed
      (the classic model.eval() side effect) and only raise if buffers
      or parameters were mutated, which cannot be safely restored.
    """
    before = state_fingerprint(model)
    result = eval_callable()
    after = state_fingerprint(model)
    problems = _diff(before, after)
    if problems and restore:
        flag_problems = [p for p in problems if p.startswith("training flag")]
        hard_problems = [p for p in problems if not p.startswith("training flag")]
        for name, mod in model.named_modules():
            mod.train(before["modes"][name])
        problems = hard_problems
        if flag_problems:
            import warnings
            warnings.warn("guarded_eval restored training flags mutated by "
                          "the evaluation: " + "; ".join(flag_problems))
    if problems:
        raise StateMutationError(
            "Evaluation mutated model state:\n  " + "\n  ".join(problems))
    return result


def assert_anchor(value, expected, tol=0.0, name="anchor"):
    """Assert a runtime value against a recorded provenance anchor."""
    if abs(value - expected) > tol:
        raise AssertionError(
            f"[ANCHOR MISMATCH] {name}: got {value!r}, expected {expected!r} "
            f"(tol {tol}). The pipeline is not reproducing the recorded "
            f"provenance chain; stop and investigate before recording numbers.")
    return True


if __name__ == "__main__":
    # Self-test on a tiny model: demonstrates catching the exact bug class
    # that corrupted this project's original pipeline.
    import torch, torch.nn as nn
    net = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(),
                        nn.Linear(8, 3))
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()

    def bad_probe():
        net.eval()                      # the original bug: mode side effect
        with torch.no_grad():
            return net(torch.randn(16, 4)).argmax(1).float().mean().item()

    try:
        guarded_eval(net, bad_probe)
        print("FAIL: mutation not caught")
    except StateMutationError as e:
        print("OK: strict guard caught the eval-mode side effect:")
        print("   ", str(e).splitlines()[1].strip())

    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()

    def good_probe():
        was = {n: m.training for n, m in net.named_modules()}
        net.eval()
        with torch.no_grad():
            out = net(torch.randn(16, 4)).argmax(1).float().mean().item()
        for n, m in net.named_modules():
            m.train(was[n])             # correct: restore what you changed
        return out

    guarded_eval(net, good_probe)
    print("OK: state-preserving evaluation passes the guard.")
