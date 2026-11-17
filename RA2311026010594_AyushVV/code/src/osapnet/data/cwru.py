"""
CWRU bearing dataset: class registry, file loading and leakage-controlled windowing.

Beginner notes
--------------
* The CWRU data are .mat files. Each file is ONE continuous recording of one
  bearing condition (e.g. "inner-race fault, 0.007 inch") at ONE motor load
  (0, 1, 2 or 3 HP).
* We cut each recording into windows of 2048 samples (about 0.17 s at 12 kHz).
* Leakage control: for every class that is used for validation or testing we
  split each recording in time into two disjoint REGIONS.
      region "A" = first half  -> only used for SUPPORT (labelled) windows
      region "B" = second half -> only used for QUERY (test) windows
  A guard gap of one window length separates them, and windows inside a region
  do not overlap. So no query window shares a single sample with a support
  window. Training classes are completely different classes, so their windows
  can never appear at test time.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Class registry: class name -> (accelerometer channel, [file number at load 0,1,2,3])
# File numbers follow the official CWRU Bearing Data Center numbering.
# ---------------------------------------------------------------------------
CLASS_FILES: Dict[str, Tuple[str, List[int]]] = {
    # ---- 12 kHz drive-end (DE) bearing data, DE accelerometer ----
    "Normal":   ("DE", [97, 98, 99, 100]),
    "IR007":    ("DE", [105, 106, 107, 108]),
    "B007":     ("DE", [118, 119, 120, 121]),
    "OR007@6":  ("DE", [130, 131, 132, 133]),
    "OR007@3":  ("DE", [144, 145, 146, 147]),
    "OR007@12": ("DE", [156, 158, 159, 160]),
    "IR014":    ("DE", [169, 170, 171, 172]),
    "B014":     ("DE", [185, 186, 187, 188]),
    "OR014@6":  ("DE", [197, 198, 199, 200]),
    "IR021":    ("DE", [209, 210, 211, 212]),
    "B021":     ("DE", [222, 223, 224, 225]),
    "OR021@6":  ("DE", [234, 235, 236, 237]),
    "OR021@3":  ("DE", [246, 247, 248, 249]),
    "OR021@12": ("DE", [258, 259, 260, 261]),
    "IR028":    ("DE", [3001, 3002, 3003, 3004]),
    "B028":     ("DE", [3005, 3006, 3007, 3008]),
    # ---- 12 kHz fan-end (FE) bearing data, FE accelerometer (cross-domain) ----
    "FE-IR007":   ("FE", [278, 279, 280, 281]),
    "FE-IR014":   ("FE", [274, 275, 276, 277]),
    "FE-IR021":   ("FE", [270, 271, 272, 273]),
    "FE-B007":    ("FE", [282, 283, 284, 285]),
    "FE-B014":    ("FE", [286, 287, 288, 289]),
    "FE-B021":    ("FE", [290, 291, 292, 293]),
    "FE-OR007@6": ("FE", [294, 295, 296, 297]),
    "FE-OR007@3": ("FE", [298, 299, 300, 301]),
}


def find_mat_file(root: str, number: int) -> str:
    """Locate the .mat file for a CWRU file number.

    Works with the official naming ("105.mat") and with the common GitHub
    mirror naming ("105_0.mat", "130@6_0.mat", "97_Normal_0.mat").
    """
    pattern = re.compile(rf"^{number}(\.mat$|_|@)")
    hits = [p for p in glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)
            if pattern.match(os.path.basename(p))]
    if not hits:
        raise FileNotFoundError(f"CWRU file {number} not found under {root}. "
                                f"See README section 'Dataset download'.")
    hits.sort(key=len)
    return hits[0]


def load_signal(root: str, number: int, channel: str) -> np.ndarray:
    """Return the 1-D vibration signal (float32) of one recording."""
    mat = loadmat(find_mat_file(root, number))
    key = f"X{number:03d}_{channel}_time"
    if key not in mat:
        # Known CWRU quirk: the 0.028-inch files (3001-3008) store their signal under a
        # different variable number (e.g. 3001.mat -> X056_DE_time). Accept the single
        # matching channel variable when it is unambiguous.
        cands = [k for k in mat if k.endswith(f"_{channel}_time")]
        if len(cands) != 1:
            raise KeyError(f"{key} not in file {number}; keys={list(mat)}")
        key = cands[0]
    return mat[key].astype(np.float32).ravel()


def make_windows(sig: np.ndarray, start: int, stop: int, length: int, stride: int) -> np.ndarray:
    """Cut sig[start:stop] into windows; returns (n, length) and keeps window starts."""
    starts = np.arange(start, stop - length + 1, stride)
    return np.stack([sig[s:s + length] for s in starts]) if len(starts) else np.zeros((0, length), np.float32), starts


@dataclass
class RecordingWindows:
    """Windows of one (class, load) recording, already split into regions."""
    cls: str
    load: int
    support: np.ndarray            # region A windows (n_a, L)
    query: np.ndarray              # region B windows (n_b, L)
    support_starts: np.ndarray = field(repr=False, default=None)
    query_starts: np.ndarray = field(repr=False, default=None)


class CWRUBank:
    """Holds all windows needed for one experiment.

    Parameters
    ----------
    root        : folder that contains the CWRU .mat files (any sub-folder layout)
    train_cls   : classes used for (meta-)training -> overlapping windows allowed
    eval_cls    : classes used for validation / test -> region A/B split
    window      : window length in samples (2048)
    train_stride: stride for training windows (overlap = data augmentation)
    loads       : motor loads to include (0..3)
    """

    def __init__(self, root: str, train_cls: List[str], eval_cls: List[str],
                 window: int = 2048, train_stride: int = 512, loads=(0, 1, 2, 3),
                 split_ratio: float = 0.5):
        overlap = set(train_cls) & set(eval_cls)
        if overlap:
            raise ValueError(f"Train and evaluation classes overlap: {overlap}")
        self.window = window
        self.loads = list(loads)
        self.train: Dict[str, Dict[int, np.ndarray]] = {}
        self.eval: Dict[str, Dict[int, RecordingWindows]] = {}
        for c in train_cls:
            ch, nums = CLASS_FILES[c]
            self.train[c] = {}
            for l in self.loads:
                sig = load_signal(root, nums[l], ch)
                w, _ = make_windows(sig, 0, len(sig), window, train_stride)
                self.train[c][l] = w
        for c in eval_cls:
            ch, nums = CLASS_FILES[c]
            self.eval[c] = {}
            for l in self.loads:
                sig = load_signal(root, nums[l], ch)
                cut = int(len(sig) * split_ratio)
                a, sa = make_windows(sig, 0, cut - window, window, window)         # guard gap = window
                b, sb = make_windows(sig, cut, len(sig), window, window)
                self.eval[c][l] = RecordingWindows(c, l, a, b, sa, sb)

    # ------------------------------------------------------------------
    def leakage_report(self) -> List[str]:
        """Programmatic proof that support and query windows never share samples."""
        lines = []
        for c, per_load in self.eval.items():
            for l, rw in per_load.items():
                a_end = (rw.support_starts.max() + self.window) if len(rw.support_starts) else 0
                b_start = rw.query_starts.min()
                ok = a_end <= b_start
                lines.append(f"{c:10s} load{l}: support windows={len(rw.support):3d} "
                             f"(last sample {a_end:6d}) | query windows={len(rw.query):3d} "
                             f"(first sample {b_start:6d}) | disjoint={ok}")
                if not ok:
                    raise AssertionError(f"Leakage detected in {c} load {l}")
        return lines


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-window standardisation (zero mean, unit variance)."""
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True) + 1e-8
    return (x - m) / s


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at a given signal-to-noise ratio (dB), per window."""
    p_signal = (x ** 2).mean(axis=-1, keepdims=True)
    p_noise = p_signal / (10 ** (snr_db / 10.0))
    return x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(p_noise)
