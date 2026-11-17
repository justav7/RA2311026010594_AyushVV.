"""
Few-shot models used in the study.

  SiameseWDCNN  - base paper (Zhang et al., IEEE Access 2019): twin WDCNN,
                  L1 distance, dropout, sigmoid "same-class" probability.
  OSAPNet       - proposed Open-Set Attentive Prototypical Network. With all
                  options switched off it reduces EXACTLY to a Prototypical
                  Network (Snell et al., 2017) with a WDCNN backbone, which is
                  the established few-shot baseline.

Every model exposes the same episode interface:
    out = model.episode(support_x, support_y, query_x, n_way)
    out["logits"]  (Q, N)  class scores over the N known classes
    out["score"]   (Q,)    open-set score, LARGER = more likely UNKNOWN
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import WDCNN


# =============================================================== base paper
class SiameseWDCNN(nn.Module):
    def __init__(self, embed_dim: int = 100):
        super().__init__()
        # base-paper code: no batch-norm, sigmoid embedding of size 100
        self.encoder = WDCNN(embed_dim, use_se=False, batchnorm=False, embed_activation="sigmoid")
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(embed_dim, 1)

    def pair_logit(self, za, zb):
        return self.head(self.drop(torch.abs(za - zb))).squeeze(-1)

    def forward(self, xa, xb):
        return self.pair_logit(self.encoder(xa), self.encoder(xb))

    def episode(self, sx, sy, qx, n_way):
        zs, zq = self.encoder(sx), self.encoder(qx)
        # similarity of every query to every support window
        sim = torch.sigmoid(self.pair_logit(zq[:, None, :].expand(-1, len(zs), -1),
                                            zs[None, :, :].expand(len(zq), -1, -1)))
        onehot = F.one_hot(sy, n_way).float()                    # (S, N)
        cls_sim = sim @ onehot / onehot.sum(0, keepdim=True)     # mean similarity per class
        return {"logits": torch.log(cls_sim.clamp_min(1e-6)),
                "prob": cls_sim / cls_sim.sum(1, keepdim=True),
                "score": 1.0 - cls_sim.max(1).values}            # low similarity => unknown


# ================================================================= proposed
class OSAPNet(nn.Module):
    """Open-Set Attentive Prototypical Network.

    Components (each can be ablated):
      use_se      : SE channel attention inside the WDCNN backbone
      adaptive    : reliability-weighted prototypes + shrinkage variance
                    normalisation (defined for K = 1)
      (open-set loss is switched in the training script, see losses.py)
    """

    def __init__(self, embed_dim: int = 100, use_se: bool = True, adaptive: bool = True):
        super().__init__()
        self.encoder = WDCNN(embed_dim, use_se=use_se, batchnorm=True)
        self.adaptive = adaptive
        self.D = embed_dim
        # learnable positive scalars (stored as raw values -> softplus)
        self._t_w = nn.Parameter(torch.tensor(0.0))      # support-weight temperature
        self._sigma0 = nn.Parameter(torch.tensor(0.5))   # prior class spread  sigma0^2
        self._kappa = nn.Parameter(torch.tensor(0.5))    # prior strength (pseudo-count)
        self._gamma = nn.Parameter(torch.tensor(0.5))    # logit temperature

    @staticmethod
    def _sp(x):
        return F.softplus(x) + 1e-4

    def prototypes(self, zs, sy, n_way):
        """Return prototypes P (N,D) and per-class variance sigma^2 (N,)."""
        P, V = [], []
        for c in range(n_way):
            z = zs[sy == c]                                         # (K, D)
            mu = z.mean(0)
            if not self.adaptive:
                P.append(mu); V.append(torch.ones((), device=z.device)); continue
            # reliability weights: supports close to the class centre count more
            d = ((z - mu) ** 2).sum(1) / self.D
            a = torch.softmax(-d / self._sp(self._t_w), dim=0)       # (K,)
            p = (a[:, None] * z).sum(0)
            s2 = (a * ((z - p) ** 2).sum(1)).sum() / self.D          # weighted spread
            K = z.shape[0]
            kappa, s0 = self._sp(self._kappa), self._sp(self._sigma0)
            var = (K * s2 + kappa * s0) / (K + kappa)                # shrinkage estimate
            P.append(p); V.append(var)
        return torch.stack(P), torch.stack(V)

    def episode(self, sx, sy, qx, n_way):
        zs, zq = self.encoder(sx), self.encoder(qx)
        P, V = self.prototypes(zs, sy, n_way)
        sq = torch.cdist(zq, P) ** 2                                  # (Q, N) squared distance
        if self.adaptive:
            dn = sq / (self.D * V[None, :])                           # normalised distance
            logits = -self._sp(self._gamma) * (dn + torch.log(V)[None, :])
        else:                                                         # plain ProtoNet
            dn = sq / self.D
            logits = -sq
        return {"logits": logits, "prob": torch.softmax(logits, 1),
                "dist": dn, "score": dn.min(1).values}                # far from all => unknown


# =========================================================== supervised CNN
class SupervisedWDCNN(nn.Module):
    """Conventional supervised classifier (WDCNN + softmax)."""

    def __init__(self, n_classes: int, embed_dim: int = 100):
        super().__init__()
        self.encoder = WDCNN(embed_dim, use_se=False, batchnorm=True)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.head(self.drop(self.encoder(x)))


def build_model(name: str, cfg: dict) -> nn.Module:
    """Factory used by train / evaluate / infer scripts."""
    D = cfg["model"]["embed_dim"]
    table = {
        "siamese":         lambda: SiameseWDCNN(D),
        "protonet":        lambda: OSAPNet(D, use_se=False, adaptive=False),
        "osapnet":         lambda: OSAPNet(D, use_se=True, adaptive=True),
        "osap_no_se":      lambda: OSAPNet(D, use_se=False, adaptive=True),
        "osap_no_adaptive": lambda: OSAPNet(D, use_se=True, adaptive=False),
        "osap_no_openloss": lambda: OSAPNet(D, use_se=True, adaptive=True),
        "supervised":      lambda: SupervisedWDCNN(len(cfg["classes"]["train"]), D),
    }
    if name not in table:
        raise KeyError(f"unknown model {name}; choose from {list(table)}")
    return table[name]()


def uses_open_loss(name: str) -> bool:
    return name in ("osapnet", "osap_no_se", "osap_no_adaptive")
