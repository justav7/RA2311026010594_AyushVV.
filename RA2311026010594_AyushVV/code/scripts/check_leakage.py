"""
Leakage audit. Verifies that
  1. train / val / test / cross-domain class sets are pairwise disjoint;
  2. in every evaluation recording, all support windows end before the first
     query window begins (no shared samples);
  3. no support window is byte-identical to any query window.
Writes outputs/leakage_report.txt.  Usage: python scripts/check_leakage.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from osapnet.data import CWRUBank  # noqa: E402
from osapnet.utils import load_config  # noqa: E402

cfg = load_config("configs/default.yaml")
C = cfg["classes"]
sets = {k: set(C[k]) for k in ["train", "val", "test", "cross_domain"]}
lines = ["== Class-set disjointness =="]
names = list(sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        inter = sets[names[i]] & sets[names[j]]
        lines.append(f"{names[i]:>12s} & {names[j]:<12s}: {'OK (disjoint)' if not inter else 'LEAK ' + str(inter)}")
        assert not inter
bank = CWRUBank(cfg["paths"]["data_root"], [], C["val"] + C["test"] + C["cross_domain"],
                cfg["data"]["window"], cfg["data"]["train_stride"], cfg["data"]["loads"], cfg["data"]["split_ratio"])
lines.append("\n== Support (region A) vs query (region B) per recording ==")
lines += bank.leakage_report()
lines.append("\n== Duplicate-window check ==")
for c, per in bank.eval.items():
    for l, rw in per.items():
        a = {w.tobytes() for w in rw.support}
        dup = sum(w.tobytes() in a for w in rw.query)
        assert dup == 0
lines.append("No support window is identical to any query window: OK")
os.makedirs(cfg["paths"]["outputs"], exist_ok=True)
open(os.path.join(cfg["paths"]["outputs"], "leakage_report.txt"), "w").write("\n".join(lines))
print("\n".join(lines))
print("\nLEAKAGE AUDIT PASSED")
