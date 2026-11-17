"""
Shared episode inference for every compared method.

Methods
-------
cnn_scratch   conventional supervised WDCNN trained from scratch on the K-shot
              support set of each episode (score = 1 - max softmax probability)
transfer      WDCNN pre-trained with softmax on the base classes, then a
              logistic-regression classifier fitted on the frozen support
              embeddings (score = 1 - max probability)
siamese       base paper (score = 1 - max mean similarity)
protonet / osapnet / osap_*  prototype models (score = min normalised distance)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .models import SupervisedWDCNN
from .utils import to_t


@torch.no_grad()
def _embed(encoder, x, dev, bs=256):
    return torch.cat([encoder(to_t(x[i:i + bs], dev)) for i in range(0, len(x), bs)]).cpu().numpy()


def _scratch_cnn(ep, n_way, dev, cfg, seed):
    torch.manual_seed(seed)
    net = SupervisedWDCNN(n_way, cfg["model"]["embed_dim"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["supervised"]["lr"])
    sx, sy = to_t(ep.support, dev), torch.as_tensor(ep.support_y, device=dev)
    net.train()
    for _ in range(cfg["supervised"]["scratch_steps"]):
        shift = int(np.random.randint(0, sx.shape[-1]))          # augmentation: circular shift
        xb = torch.roll(sx, shifts=shift, dims=-1) + 0.05 * torch.randn_like(sx)
        loss = F.cross_entropy(net(xb), sy)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(to_t(ep.query, dev)), 1).cpu().numpy()
    return p


def run_episode(name: str, model, ep, dev, cfg, seed: int = 0):
    """Return (pred, score, prob) as numpy arrays for one episode."""
    n_way = len(ep.known_names)
    if name == "cnn_scratch":
        p = _scratch_cnn(ep, n_way, dev, cfg, seed)
        return p.argmax(1), 1.0 - p.max(1), p
    if name == "transfer":
        model.eval()
        zs, zq = _embed(model.encoder, ep.support, dev), _embed(model.encoder, ep.query, dev)
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(zs, ep.support_y)
        p = clf.predict_proba(zq)
        return p.argmax(1), 1.0 - p.max(1), p
    model.eval()
    with torch.no_grad():
        out = model.episode(to_t(ep.support, dev), torch.as_tensor(ep.support_y, device=dev),
                            to_t(ep.query, dev), n_way)
    prob = out["prob"].cpu().numpy()
    return prob.argmax(1), out["score"].cpu().numpy(), prob
