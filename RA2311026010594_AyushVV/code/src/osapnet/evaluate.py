"""
Evaluation entry point (all methods, all protocols, identical episodes).

python -m osapnet.evaluate --model osapnet --seed 0 --protocol all
python -m osapnet.evaluate --model cnn_scratch --seed 0 --protocol same
python -m osapnet.evaluate --model osapnet --seed 0 --protocol same --episodes 10   # smoke test

Protocols
  same        support & query from the same load, K in eval.shots
  cross_load  query recordings from a different motor load, K = 5
  cross_domain fan-end bearing / FE sensor classes (never seen), K = 5
  noise       white Gaussian noise at eval.noise_snr_db, K = 5
  confusion   fixed split: known = first 3 test classes, unknown = last 2, K = 5

Threshold tau is calibrated per K on VALIDATION classes using known-class
scores only (target FRR = 5 %), then frozen and applied to test episodes.

Outputs (outputs/results/):
  episodes_<model>_seed<k>.csv     one row per episode with all metrics
  scores_<model>_seed<k>.npz       raw open-set scores for ROC/PR curves
  confusion_<model>_seed<k>.json   aggregated (3 known + unknown) confusion matrix
"""
from __future__ import annotations

import argparse
import os
import zlib

import numpy as np
import pandas as pd
import torch

from .data import CWRUBank, EpisodeSampler
from .engine import run_episode
from .metrics import calibrate_threshold, episode_metrics
from .models import build_model
from .utils import ckpt_path, device, load_config, save_json, set_seed

CHECKPOINT_OF = {"transfer": "supervised"}   # transfer baseline re-uses the supervised pre-training


def _episode_seed(seed, tag):
    """Same (seed, protocol, K) -> same episodes for every method (paired comparison)."""
    return 20_000 + 1000 * seed + zlib.crc32(tag.encode()) % 997


def load_model(name, cfg, seed, dev):
    if name == "cnn_scratch":
        return None
    ck_name = CHECKPOINT_OF.get(name, name)
    model = build_model(ck_name, cfg).to(dev)
    ck = torch.load(ckpt_path(cfg, ck_name, seed), map_location=dev)
    model.load_state_dict(ck["state"])
    model.eval()
    return model


def calibrate(name, model, bank, cfg, dev, seed, k, n_ep):
    s = EpisodeSampler(bank, _episode_seed(seed, f"val-k{k}"))
    known_scores = []
    for _ in range(n_ep):
        ep = s.eval_episode(cfg["classes"]["val"], cfg["eval"]["val_n_known"], k, cfg["eval"]["q_per_class"])
        _, score, _ = run_episode(name, model, ep, dev, cfg, seed)
        known_scores.append(score[ep.query_y >= 0])
    return calibrate_threshold(np.concatenate(known_scores), cfg["eval"]["target_frr"])


def run_protocol(name, model, bank, cfg, dev, seed, pool, k, n_ep, tau, protocol="same",
                 snr=None, tag="", shuffle=True, collect=None):
    s = EpisodeSampler(bank, _episode_seed(seed, tag))
    rows = []
    for e in range(n_ep):
        ep = s.eval_episode(pool, cfg["eval"]["n_known"], k, cfg["eval"]["q_per_class"],
                            protocol=protocol, snr_db=snr, shuffle=shuffle)
        pred, score, _ = run_episode(name, model, ep, dev, cfg, seed * 100_000 + e)
        m = episode_metrics(pred, score, ep.query_y, len(ep.known_names), tau)
        rows.append({"model": name, "seed": seed, "tag": tag, "k_shot": k, "episode": e,
                     "tau": tau, **m})
        if collect is not None:
            collect.append((ep, pred, score, tau))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default="all",
                    choices=["all", "same", "cross_load", "cross_domain", "noise", "confusion"])
    ap.add_argument("--episodes", type=int, default=None, help="override episodes per setting")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    set_seed(a.seed)
    dev = device(cfg)
    E = cfg["eval"]
    n_ep = a.episodes or (E["supervised_episodes"] if a.model == "cnn_scratch" else E["episodes"])
    n_cal = max(20, n_ep // 3)
    C = cfg["classes"]
    bank = CWRUBank(cfg["paths"]["data_root"], [], C["val"] + C["test"] + C["cross_domain"],
                    cfg["data"]["window"], cfg["data"]["train_stride"], cfg["data"]["loads"],
                    cfg["data"]["split_ratio"])
    model = load_model(a.model, cfg, a.seed, dev)
    protos = ["same", "cross_load", "cross_domain", "noise", "confusion"] if a.protocol == "all" else [a.protocol]
    taus = {}

    def tau_for(k):
        if k not in taus:
            taus[k] = calibrate(a.model, model, bank, cfg, dev, a.seed, k, n_cal)
        return taus[k]

    rows, raw = [], {}
    for p in protos:
        if p == "same":
            for k in E["shots"]:
                col = []
                rows += run_protocol(a.model, model, bank, cfg, dev, a.seed, C["test"], k, n_ep,
                                     tau_for(k), tag=f"same-k{k}", collect=col)
                raw[f"same_k{k}_score"] = np.concatenate([c[2] for c in col])
                raw[f"same_k{k}_unknown"] = np.concatenate([(c[0].query_y < 0) for c in col]).astype(np.int8)
        elif p == "cross_load":
            rows += run_protocol(a.model, model, bank, cfg, dev, a.seed, C["test"], 5, n_ep, tau_for(5),
                                 protocol="cross_load", tag="cross_load-k5")
        elif p == "cross_domain":
            rows += run_protocol(a.model, model, bank, cfg, dev, a.seed, C["cross_domain"], 5, n_ep,
                                 tau_for(5), tag="cross_domain-k5")
        elif p == "noise":
            k = E["noise_shot"]
            for snr in E["noise_snr_db"]:
                rows += run_protocol(a.model, model, bank, cfg, dev, a.seed, C["test"], k, n_ep,
                                     tau_for(k), snr=snr, tag=f"noise{snr}dB-k{k}")
        elif p == "confusion":
            col = []
            run_protocol(a.model, model, bank, cfg, dev, a.seed, C["test"], 5, max(50, n_ep // 3),
                         tau_for(5), tag="confusion-k5", shuffle=False, collect=col)
            labels = C["test"][:3] + ["Unknown"]
            cm = np.zeros((4, 4), int)
            for ep, pred, score, tau in col:
                true = np.where(ep.query_y < 0, 3, ep.query_y)
                op = np.where(score > tau, 3, pred)
                for t, q in zip(true, op):
                    cm[t, q] += 1
            out = os.path.join(cfg["paths"]["outputs"], "results", f"confusion_{a.model}_seed{a.seed}.json")
            save_json({"labels": labels, "matrix": cm.tolist(),
                       "note": f"rows=true, cols=predicted; unknown rows pool {C['test'][3:]}"}, out)
    res = os.path.join(cfg["paths"]["outputs"], "results")
    os.makedirs(res, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        f = os.path.join(res, f"episodes_{a.model}_seed{a.seed}.csv")
        if a.protocol != "all" and os.path.exists(f):            # append partial runs
            old = pd.read_csv(f)
            df = pd.concat([old[~old.tag.isin(df.tag.unique())], df])
        df.to_csv(f, index=False)
        print(df.groupby("tag")[["acc", "macro_f1", "auroc", "auprc", "udr", "frr"]].mean().round(4))
    if raw:
        np.savez_compressed(os.path.join(res, f"scores_{a.model}_seed{a.seed}.npz"), **raw)
    save_json(taus, os.path.join(res, f"thresholds_{a.model}_seed{a.seed}.json"))


if __name__ == "__main__":
    main()
