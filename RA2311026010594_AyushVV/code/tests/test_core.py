"""Unit tests: shapes, prototype maths, metric sanity and leakage guards."""
import os, sys
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from osapnet.models import WDCNN, OSAPNet, SiameseWDCNN          # noqa: E402
from osapnet.metrics import episode_metrics, calibrate_threshold  # noqa: E402
from osapnet.data.cwru import make_windows, zscore, add_noise     # noqa: E402


def test_backbone_shape():
    for se in (False, True):
        assert WDCNN(100, use_se=se)(torch.randn(4, 1, 2048)).shape == (4, 100)


def test_osapnet_episode_k1_and_k5():
    torch.manual_seed(0)
    m = OSAPNet(100).eval()
    for k in (1, 5):
        sx, sy, qx = torch.randn(3 * k, 2048), torch.arange(3).repeat_interleave(k), torch.randn(7, 2048)
        out = m.episode(sx, sy, qx, 3)
        assert out["logits"].shape == (7, 3) and out["score"].shape == (7,)
        assert torch.isfinite(out["logits"]).all()


def test_protonet_reduces_to_mean_prototype():
    m = OSAPNet(100, use_se=False, adaptive=False)
    z = torch.randn(6, 100); y = torch.tensor([0, 0, 0, 1, 1, 1])
    P, V = m.prototypes(z, y, 2)
    assert torch.allclose(P[0], z[:3].mean(0)) and torch.allclose(V, torch.ones(2))


def test_siamese_episode():
    out = SiameseWDCNN().eval().episode(torch.randn(6, 2048), torch.tensor([0, 0, 1, 1, 2, 2]), torch.randn(4, 2048), 3)
    assert out["score"].min() >= 0 and out["score"].max() <= 1


def test_metrics_perfect():
    qy = np.array([0, 1, -1, -1]); pred = np.array([0, 1, 0, 1]); score = np.array([.1, .2, .9, .8])
    m = episode_metrics(pred, score, qy, 2, tau=0.5)
    assert m["acc"] == 1 and m["auroc"] == 1 and m["udr"] == 1 and m["frr"] == 0 and m["open_acc"] == 1


def test_threshold_quantile():
    assert abs(calibrate_threshold(np.arange(101.0), 0.05) - 95.0) < 1e-9


def test_windows_do_not_cross_region():
    sig = np.arange(10000, dtype=np.float32)
    a, sa = make_windows(sig, 0, 5000 - 1000, 1000, 1000)
    b, sb = make_windows(sig, 5000, 10000, 1000, 1000)
    assert sa.max() + 1000 <= sb.min()


def test_noise_snr():
    rng = np.random.default_rng(0); x = np.sin(np.linspace(0, 200, 20000, dtype=np.float32))[None]
    y = add_noise(x, 0.0, rng)
    assert abs(10 * np.log10((x ** 2).mean() / ((y - x) ** 2).mean())) < 0.2
    assert abs(zscore(y).std() - 1) < 1e-3
