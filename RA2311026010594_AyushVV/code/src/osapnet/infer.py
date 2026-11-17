"""
Inference demonstration: few-shot diagnosis with unknown-failure rejection.

Scenario: an engineer has K labelled windows of each KNOWN fault and receives a
new vibration window. The model outputs either a known fault label or
"UNKNOWN FAILURE - inspect", together with the open-set score and threshold.

python -m osapnet.infer --model osapnet --seed 0 \
       --known IR014 B014 OR014@6 --query-class IR028 --k 5
(IR028 is not in the known list, so the correct answer is UNKNOWN.)
"""
import argparse
import json
import os

import numpy as np
import torch

from .data import CWRUBank, EpisodeSampler
from .evaluate import load_model
from .utils import device, load_config, to_t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="osapnet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--known", nargs="+", default=["IR014", "B014", "OR014@6"])
    ap.add_argument("--query-class", default="IR028")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--load", type=int, default=0)
    ap.add_argument("--n-queries", type=int, default=5)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    dev = device(cfg)
    classes = sorted(set(a.known + [a.query_class]))
    bank = CWRUBank(cfg["paths"]["data_root"], [], classes, cfg["data"]["window"],
                    cfg["data"]["train_stride"], [a.load], cfg["data"]["split_ratio"])
    model = load_model(a.model, cfg, a.seed, dev)
    tau_file = os.path.join(cfg["paths"]["outputs"], "results", f"thresholds_{a.model}_seed{a.seed}.json")
    taus = json.load(open(tau_file)) if os.path.exists(tau_file) else {}
    tau = taus.get(str(a.k))
    rng = np.random.default_rng(0)
    from .data.cwru import zscore
    sx = np.concatenate([bank.eval[c][a.load].support[rng.choice(len(bank.eval[c][a.load].support), a.k, replace=False)]
                         for c in a.known])
    sy = np.repeat(np.arange(len(a.known)), a.k)
    qpool = bank.eval[a.query_class][a.load].query
    qx = qpool[rng.choice(len(qpool), a.n_queries, replace=False)]
    with torch.no_grad():
        out = model.episode(to_t(zscore(sx), dev), torch.as_tensor(sy, device=dev),
                            to_t(zscore(qx), dev), len(a.known))
    prob, score = out["prob"].cpu().numpy(), out["score"].cpu().numpy()
    print(f"Known classes (support, K={a.k}): {a.known}")
    print(f"True class of query windows   : {a.query_class}"
          f" ({'KNOWN' if a.query_class in a.known else 'UNSEEN -> should be rejected'})")
    print(f"Rejection threshold tau        : {tau if tau is not None else 'not calibrated (run evaluate first)'}")
    for i in range(len(qx)):
        best = int(prob[i].argmax())
        reject = tau is not None and score[i] > tau
        label = "UNKNOWN FAILURE - inspect" if reject else a.known[best]
        print(f"  window {i}: prediction = {label:28s} p_max = {prob[i, best]:.3f}  open-set score = {score[i]:.3f}")


if __name__ == "__main__":
    main()
