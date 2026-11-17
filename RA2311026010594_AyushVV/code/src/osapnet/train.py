"""
Training entry point.

Examples
--------
python -m osapnet.train --model osapnet  --seed 0
python -m osapnet.train --model siamese  --seed 0
python -m osapnet.train --model supervised --seed 0      # pre-training for the transfer baseline
python -m osapnet.train --model osapnet --seed 0 --episodes 20   # quick smoke test

Outputs: outputs/checkpoints/<model>_seed<k>.pt  (best validation checkpoint)
         outputs/logs/<model>_seed<k>_curve.csv  (training loss + validation curve)
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import CWRUBank, EpisodeSampler, zscore
from .engine import run_episode
from .losses import episode_loss
from .metrics import episode_metrics
from .models import build_model, uses_open_loss
from .utils import ckpt_path, device, load_config, set_seed, to_t


def validate(name, model, sampler, cfg, dev, n_episodes, k=5):
    """Mean closed-set accuracy and AUROC on VALIDATION classes (never test classes)."""
    accs, aucs = [], []
    for _ in range(n_episodes):
        ep = sampler.eval_episode(cfg["classes"]["val"], cfg["eval"]["val_n_known"], k,
                                  cfg["eval"]["q_per_class"])
        pred, score, _ = run_episode(name, model, ep, dev, cfg)
        m = episode_metrics(pred, score, ep.query_y, len(ep.known_names), None)
        accs.append(m["acc"]); aucs.append(m["auroc"])
    return float(np.mean(accs)), float(np.mean(aucs))


def train_episodic(name, model, bank, cfg, dev, seed, episodes, log):
    tc = cfg["train"]
    sampler, vsampler = EpisodeSampler(bank, seed), EpisodeSampler(bank, 10_000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, tc["lr_step"], 0.5)
    best, path = -1, ckpt_path(cfg, name, seed)
    for it in range(1, episodes + 1):
        model.train()
        k = int(sampler.rng.integers(tc["k_shot_range"][0], tc["k_shot_range"][1] + 1))
        ep = sampler.train_episode(cfg["classes"]["train"], tc["n_known"], tc["n_unknown"],
                                   k, tc["q_known"], tc["q_unknown"])
        qy = torch.as_tensor(ep.query_y, device=dev)
        out = model.episode(to_t(ep.support, dev), torch.as_tensor(ep.support_y, device=dev),
                            to_t(ep.query, dev), tc["n_known"])
        loss, logs = episode_loss(out, qy, tc["n_known"], tc["open_loss"], uses_open_loss(name))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        row = {"step": it, **logs}
        if it % tc["val_every"] == 0 or it == episodes:
            acc, auc = validate(name, model, vsampler, cfg, dev, tc["val_episodes"])
            row.update(val_acc=acc, val_auroc=auc)
            crit = 0.5 * (acc + auc)
            if crit > best:
                best = crit
                torch.save({"model": name, "state": model.state_dict(), "step": it,
                            "val_acc": acc, "val_auroc": auc}, path)
            print(f"[{name} s{seed}] ep {it:5d} loss {logs['total']:.3f} val_acc {acc:.3f} val_auroc {auc:.3f}")
        log(row)
    return path


def train_siamese(name, model, bank, cfg, dev, seed, iters, log):
    sc = cfg["siamese"]
    rng = np.random.default_rng(seed)
    vsampler = EpisodeSampler(bank, 10_000 + seed)
    pools = [np.concatenate([bank.train[c][l] for l in bank.loads]) for c in cfg["classes"]["train"]]
    opt = torch.optim.Adam(model.parameters(), lr=sc["lr"])
    best, path, B = -1, ckpt_path(cfg, name, seed), sc["batch_size"]
    for it in range(1, iters + 1):
        model.train()
        ca = rng.integers(0, len(pools), B)
        cb = ca.copy()                               # 2nd half: same class (target 1)
        cb[: B // 2] = (ca[: B // 2] + rng.integers(1, len(pools), B // 2)) % len(pools)  # 1st half: different
        xa = np.stack([pools[c][rng.integers(len(pools[c]))] for c in ca])
        xb = np.stack([pools[c][rng.integers(len(pools[c]))] for c in cb])
        y = torch.as_tensor((ca == cb).astype(np.float32), device=dev)
        logit = model(to_t(zscore(xa), dev), to_t(zscore(xb), dev))
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        row = {"step": it, "total": loss.item()}
        if it % sc["val_every"] == 0 or it == iters:
            acc, auc = validate(name, model, vsampler, cfg, dev, cfg["train"]["val_episodes"])
            row.update(val_acc=acc, val_auroc=auc)
            if 0.5 * (acc + auc) > best:
                best = 0.5 * (acc + auc)
                torch.save({"model": name, "state": model.state_dict(), "step": it,
                            "val_acc": acc, "val_auroc": auc}, path)
            print(f"[siamese s{seed}] it {it:5d} loss {loss.item():.3f} val_acc {acc:.3f} val_auroc {auc:.3f}")
        log(row)
    return path


def train_supervised(name, model, bank, cfg, dev, seed, epochs, log):
    sc = cfg["supervised"]
    X = np.concatenate([np.concatenate([bank.train[c][l] for l in bank.loads]) for c in cfg["classes"]["train"]])
    Y = np.concatenate([[i] * sum(len(bank.train[c][l]) for l in bank.loads)
                        for i, c in enumerate(cfg["classes"]["train"])])
    X = zscore(X)
    rng = np.random.default_rng(seed)
    vsampler = EpisodeSampler(bank, 10_000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=sc["lr"])
    best, path, step = -1, ckpt_path(cfg, name, seed), 0
    for e in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(X))
        for i in range(0, len(X), sc["batch_size"]):
            idx = perm[i:i + sc["batch_size"]]
            loss = F.cross_entropy(model(to_t(X[idx], dev)), torch.as_tensor(Y[idx], device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        acc, auc = validate("transfer", model, vsampler, cfg, dev, cfg["train"]["val_episodes"])
        log({"step": e, "total": loss.item(), "val_acc": acc, "val_auroc": auc})
        if 0.5 * (acc + auc) > best:
            best = 0.5 * (acc + auc)
            torch.save({"model": name, "state": model.state_dict(), "step": e,
                        "val_acc": acc, "val_auroc": auc}, path)
        print(f"[supervised s{seed}] epoch {e} loss {loss.item():.3f} val_acc {acc:.3f} val_auroc {auc:.3f}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="override number of episodes/iterations/epochs")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    set_seed(a.seed)
    dev = device(cfg)
    t0 = time.time()
    bank = CWRUBank(cfg["paths"]["data_root"], cfg["classes"]["train"], cfg["classes"]["val"],
                    cfg["data"]["window"], cfg["data"]["train_stride"], cfg["data"]["loads"],
                    cfg["data"]["split_ratio"])
    model = build_model(a.model, cfg).to(dev)
    os.makedirs(os.path.join(cfg["paths"]["outputs"], "logs"), exist_ok=True)
    log_path = os.path.join(cfg["paths"]["outputs"], "logs", f"{a.model}_seed{a.seed}_curve.csv")
    fields = ["step", "total", "ce", "ent", "margin", "val_acc", "val_auroc"]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        log = lambda row: w.writerow(row)  # noqa: E731
        if a.model == "siamese":
            p = train_siamese(a.model, model, bank, cfg, dev, a.seed, a.episodes or cfg["siamese"]["iterations"], log)
        elif a.model == "supervised":
            p = train_supervised(a.model, model, bank, cfg, dev, a.seed, a.episodes or cfg["supervised"]["pretrain_epochs"], log)
        else:
            p = train_episodic(a.model, model, bank, cfg, dev, a.seed, a.episodes or cfg["train"]["episodes"], log)
    print(f"saved best checkpoint -> {p}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
