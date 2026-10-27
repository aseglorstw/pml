"""Device handling. On a GPU node (cuda) or a Mac (mps) the accelerator results are compared with CPU."""
import numpy as np
import pytest
import torch

from src.models.gbsr import GBSR
from src.models.lightgcn import LightGCN
from src.models.pugib import PUGIB, pair_features
from src.train import resolve_device

M, N = 6, 7
TRAIN = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4], [3, 0], [4, 5], [4, 2], [5, 6], [5, 1]])
SOCIAL = np.array([[0, 1], [0, 2], [1, 3], [2, 4], [4, 5]])
ACC = resolve_device("auto")


def test_resolve_device_fallbacks():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == ("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_device("auto") in ("cuda", "mps", "cpu")
    if not torch.cuda.is_available():
        assert resolve_device("cuda:0") == "cpu"


def _losses(device):
    torch.manual_seed(0)
    out = {}
    m = LightGCN(M, N, TRAIN, dim=8, social_edges=SOCIAL, device=device).to(device)
    w = torch.full((len(SOCIAL),), 0.7, device=device, requires_grad=True)
    u, p, n = (torch.tensor(x, device=device) for x in ([0, 1, 2], [0, 1, 3], [6, 4, 0]))
    loss, _ = m.bpr_loss(u, p, n, 1e-4, social_w=w)
    loss.backward()
    out["lightgcn_s"] = (loss.item(), w.grad.cpu().numpy())
    torch.manual_seed(0)
    feats = pair_features(np.random.default_rng(0).normal(size=(M, 8)), SOCIAL, np.ones(len(SOCIAL)),
                          np.ones(len(SOCIAL), bool))
    pg = PUGIB(M, N, TRAIN, SOCIAL, len(SOCIAL), np.full(len(SOCIAL), 0.9), feats, dim=8, device=device).to(device)
    out["pugib_alpha"] = pg.alpha().cpu().numpy()
    torch.manual_seed(0)
    gb = GBSR(M, N, TRAIN, SOCIAL, dim=8, device=device).to(device)
    out["gbsr_gates"] = gb.gates(sample=False).detach().cpu().numpy()
    return out


@pytest.mark.skipif(ACC == "cpu", reason="no GPU (cuda/mps) available")
def test_accelerator_matches_cpu():
    a, c = _losses(ACC), _losses("cpu")
    assert np.isclose(a["lightgcn_s"][0], c["lightgcn_s"][0], atol=1e-5)
    assert np.allclose(a["lightgcn_s"][1], c["lightgcn_s"][1], atol=1e-5)
    assert np.allclose(a["pugib_alpha"], c["pugib_alpha"], atol=1e-5)
    assert np.allclose(a["gbsr_gates"], c["gbsr_gates"], atol=1e-5)
