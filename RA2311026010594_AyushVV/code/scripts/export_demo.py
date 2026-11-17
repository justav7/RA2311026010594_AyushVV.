"""
Export trained OSAP-Net + Siamese (base paper) weights and real CWRU test windows
to a JSON bundle for the browser demo (FJ3_Live_Demo.html).

python scripts/export_demo.py --seed 0 --out outputs/demo_bundle.json
Batch-norm layers are folded into the preceding convolution (eval mode), so the
browser forward pass is: conv -> ReLU -> [SE] -> maxpool, x5 -> flatten -> FC.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from osapnet.data import CWRUBank, zscore  # noqa: E402
from osapnet.evaluate import load_model  # noqa: E402
from osapnet.utils import load_config  # noqa: E402


def b64(a):
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


@torch.no_grad()
def export_encoder(enc):
    """Return list of block dicts with BN folded."""
    blocks = []
    for blk in enc.features:
        conv = blk[0]
        W = conv.weight.detach().clone()          # (out, in, k)
        b = conv.bias.detach().clone()
        bn = next((m for m in blk if isinstance(m, torch.nn.BatchNorm1d)), None)
        if bn is not None:
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            W = W * s[:, None, None]
            b = (b - bn.running_mean) * s + bn.bias
        k = conv.kernel_size[0]
        pad = conv.padding
        if pad == "same":
            pl, pr = (k - 1) // 2, (k - 1) - (k - 1) // 2
        else:
            pl = pr = int(pad[0])
        se = next((m for m in blk if m.__class__.__name__ == "SEBlock1d"), None)
        d = {"W": b64(W.numpy()), "b": b64(b.numpy()), "cin": W.shape[1], "cout": W.shape[0], "k": k,
             "stride": conv.stride[0], "pl": pl, "pr": pr}
        if se is not None:
            l1, l2 = se.fc[0], se.fc[2]
            d["se"] = {"W1": b64(l1.weight.detach().numpy()), "b1": b64(l1.bias.detach().numpy()), "h": l1.out_features,
                       "W2": b64(l2.weight.detach().numpy()), "b2": b64(l2.bias.detach().numpy())}
        blocks.append(d)
    return {"blocks": blocks, "fcW": b64(enc.fc.weight.detach().numpy()), "fcb": b64(enc.fc.bias.detach().numpy()),
            "D": enc.fc.out_features, "sigmoid": enc.act.__class__.__name__ == "Sigmoid"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/demo_bundle.json")
    ap.add_argument("--n-support", type=int, default=10)
    ap.add_argument("--n-query", type=int, default=12)
    a = ap.parse_args()
    cfg = load_config("configs/default.yaml")
    dev = torch.device("cpu")
    osap = load_model("osapnet", cfg, a.seed, dev)
    siam = load_model("siamese", cfg, a.seed, dev)
    sp = F.softplus
    params = {"t_w": float(sp(osap._t_w) + 1e-4), "sigma0": float(sp(osap._sigma0) + 1e-4),
              "kappa": float(sp(osap._kappa) + 1e-4), "gamma": float(sp(osap._gamma) + 1e-4)}
    taus = {}
    for m in ["osapnet", "siamese"]:
        taus[m] = json.load(open(os.path.join(cfg["paths"]["outputs"], "results", f"thresholds_{m}_seed{a.seed}.json")))
    classes = cfg["classes"]["test"] + cfg["classes"]["cross_domain"]
    bank = CWRUBank(cfg["paths"]["data_root"], [], classes, cfg["data"]["window"], cfg["data"]["train_stride"],
                    [0], cfg["data"]["split_ratio"])
    rng = np.random.default_rng(123)
    data = {}
    for c in classes:
        rw = bank.eval[c][0]
        s = rw.support[rng.choice(len(rw.support), a.n_support, replace=False)]
        q = rw.query[rng.choice(len(rw.query), a.n_query, replace=False)]
        data[c] = {"support": b64(s), "query": b64(q)}
    # reference check values: embeddings of first support window of first class
    x = torch.tensor(zscore(bank.eval[classes[0]][0].support[:1]))
    with torch.no_grad():
        ref = {"osap": osap.encoder(x)[0, :8].tolist(), "siam": siam.encoder(x)[0, :8].tolist()}
    bundle = {"window": 2048, "classes": classes, "test": cfg["classes"]["test"], "xdomain": cfg["classes"]["cross_domain"],
              "data": data, "osap": export_encoder(osap.encoder), "osap_params": params,
              "siam": export_encoder(siam.encoder), "siam_head": {"W": b64(siam.head.weight.detach().numpy()), "b": float(siam.head.bias)},
              "tau": taus, "seed": a.seed, "ref": ref, "ref_window": b64(bank.eval[classes[0]][0].support[:1])}
    json.dump(bundle, open(a.out, "w"))
    print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB; params", params, "tau", taus)


if __name__ == "__main__":
    main()
