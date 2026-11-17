"""Configuration, reproducibility and checkpoint helpers."""
import json
import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Make runs repeatable (same seed -> same episodes, same initial weights)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def device(cfg) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def ckpt_path(cfg, model_name, seed):
    d = os.path.join(cfg["paths"]["outputs"], "checkpoints")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{model_name}_seed{seed}.pt")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def to_t(x, dev, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=dev)
