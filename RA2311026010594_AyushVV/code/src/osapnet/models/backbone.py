"""
WDCNN feature extractor (Zhang et al., Sensors 2017) with optional
Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

Input : (B, 1, 2048) raw, z-scored vibration window
Output: (B, 100) embedding vector

Layer plan (length after each block):
  Conv(1->16, k=64, s=16) -> 128 -> pool -> 64
  Conv(16->32, k=3)       ->  64 -> pool -> 32
  Conv(32->64, k=2)       ->  32 -> pool -> 16   (k=2 as in the base-paper code)
  Conv(64->64, k=3)       ->  16 -> pool ->  8
  Conv(64->64, k=3, no pad)->  6 -> pool ->  3   => flatten 3*64 = 192 -> FC 100
"""
import torch
import torch.nn as nn


class SEBlock1d(nn.Module):
    """Channel attention: learn a weight in (0,1) for every feature channel.

    s = sigmoid(W2 * ReLU(W1 * GAP(x)));   y = x * s
    Channels that respond to fault impulses can be amplified and channels
    dominated by noise suppressed.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, channels), nn.Sigmoid())
        self.last_weights = None  # stored for visualisation

    def forward(self, x):
        w = self.fc(x.mean(dim=-1))          # squeeze: global average over time
        self.last_weights = w.detach()
        return x * w.unsqueeze(-1)           # excite: re-weight channels


class WDCNN(nn.Module):
    def __init__(self, embed_dim: int = 100, use_se: bool = False, batchnorm: bool = True,
                 embed_activation: str = "none"):
        super().__init__()
        spec = [  # (in, out, kernel, stride, padding)
            (1, 16, 64, 16, 24),
            (16, 32, 3, 1, "same"),
            (32, 64, 2, 1, "same"),
            (64, 64, 3, 1, "same"),
            (64, 64, 3, 1, 0),
        ]
        blocks = []
        for i, (cin, cout, k, s, p) in enumerate(spec):
            layers = [nn.Conv1d(cin, cout, k, stride=s, padding=p)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(cout))
            layers.append(nn.ReLU(inplace=True))
            if use_se and i >= 1:            # attention on blocks 2-5
                layers.append(SEBlock1d(cout))
            layers.append(nn.MaxPool1d(2, 2))
            blocks.append(nn.Sequential(*layers))
        self.features = nn.Sequential(*blocks)
        self.fc = nn.Linear(64 * 3, embed_dim)
        self.act = nn.Sigmoid() if embed_activation == "sigmoid" else nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.features(x).flatten(1)
        return self.act(self.fc(h))
