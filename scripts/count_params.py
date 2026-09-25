#!/usr/bin/env python3
"""count_params.py -- parameter counts for the capacity comparison.

Writes params_count.json so that 21_verify_all.py can check these numbers
without loading a model itself.

Builds the model exactly as 17_delayed_label.py does (same weights, same
model_dir) and counts the three parameter sets the three retraining
capacities actually touch. Loads no dataset, so it runs in seconds.

Run from the repo root:

    python scripts/count_params.py
"""
import torch
import torch.nn as nn
from cesnet_models.models import mm_cesnet_v2, MM_CESNET_V2_Weights

MODEL_DIR = "./models/"

weights = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
model = mm_cesnet_v2(weights=weights, model_dir=MODEL_DIR)
model.eval()


def find_head(m):
    """Last Linear module in forward order. Identical to 17_delayed_label.py."""
    last = None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            last = (name, mod)
    if last is None:
        raise RuntimeError("no Linear layer found")
    return last


BN = (nn.BatchNorm1d, nn.BatchNorm2d)

total = sum(p.numel() for p in model.parameters())

bn_affine = sum(p.numel() for mod in model.modules() if isinstance(mod, BN)
                for p in mod.parameters(recurse=False))
n_bn = sum(1 for mod in model.modules() if isinstance(mod, BN))

bn_buffers = sum(b.numel() for mod in model.modules() if isinstance(mod, BN)
                 for b in mod.buffers(recurse=False))

head_name, head = find_head(model)
head_params = sum(p.numel() for p in head.parameters())

print(f"total parameters            : {total:,}")
print(f"'matched' (BN affine)       : {bn_affine:,}"
      f"   ({100.0*bn_affine/total:.3f}% of total, {n_bn} BN modules)")
print(f"'head'    ({head_name})     : {head_params:,}"
      f"   ({100.0*head_params/total:.3f}% of total)")
print(f"'full'                      : {total:,}   (100%)")
print()
print(f"BN running buffers (not gradient-updated, but moved by 'matched'):"
      f" {bn_buffers:,}")

import json
dropout_modules = [name for name, mod in model.named_modules()
                   if isinstance(mod, (nn.Dropout, nn.Dropout1d, nn.Dropout2d))]
print(f"Dropout modules             : {dropout_modules}")

out = {"total": total, "bn_affine": bn_affine, "head": head_params,
       "head_module": head_name, "n_bn": n_bn, "bn_buffers": bn_buffers,
       "dropout_modules": dropout_modules,
       "weights": "MM_CESNET_V2_Weights.CESNET_QUIC22_Week44"}
with open("params_count.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwritten to params_count.json")
