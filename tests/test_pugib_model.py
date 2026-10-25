import numpy as np
import torch

from src.models.pugib import PUGIB, binary_concrete, kl_bernoulli, pair_features

M, N = 5, 6
TRAIN = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4], [3, 0], [4, 5], [4, 2]])
SUPPORT = np.array([[0, 1], [0, 2], [1, 3], [2, 4], [0, 3]])  # last one is a candidate
IS_OBS = np.array([1, 1, 1, 1, 0], bool)
RHO = np.where(IS_OBS, 0.9, 0.3)


def _model(norm="static", beta=0.1):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(M, 4))
    feats = pair_features(emb, SUPPORT, np.ones(len(SUPPORT)), IS_OBS)
    torch.manual_seed(0)
    ego = torch.randn(M + N, 4) * 0.1
    return PUGIB(M, N, TRAIN, SUPPORT, IS_OBS, RHO, feats, ego, dim=4, deg_mode=norm, beta=beta,
                 tau_epochs=10), emb


def test_features_symmetric():
    emb = np.random.default_rng(1).normal(size=(M, 4))
    f1 = pair_features(emb, SUPPORT, np.arange(5.0), IS_OBS)
    f2 = pair_features(emb, SUPPORT[:, ::-1], np.arange(5.0), IS_OBS)
    assert np.allclose(f1, f2)


def test_kl_closed_form():
    a, r = torch.tensor([0.2, 0.9, 0.5]), torch.tensor([0.2, 0.3, 0.5])
    kl = kl_bernoulli(a, r)
    assert torch.allclose(kl[[0, 2]], torch.zeros(2), atol=1e-6)
    expected = 0.9 * np.log(0.9 / 0.3) + 0.1 * np.log(0.1 / 0.7)
    assert np.isclose(kl[1].item(), expected, atol=1e-5)
    assert torch.isfinite(kl_bernoulli(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))).all()


def test_concrete_mean_tracks_alpha_at_low_temperature():
    torch.manual_seed(0)
    logits = torch.full((200000,), float(np.log(0.3 / 0.7)))
    z = binary_concrete(logits, tau=0.05)
    assert abs((z > 0.5).float().mean().item() - 0.3) < 0.01


def test_prior_degree_normalization():
    m, _ = _model("prior")
    deg = np.zeros(M)
    np.add.at(deg, SUPPORT[:, 0], RHO)
    np.add.at(deg, SUPPORT[:, 1], RHO)
    s = m.backbone.social
    expect = 1 / np.sqrt(deg[s.src.numpy()] * deg[s.dst.numpy()])
    assert np.allclose(s.norm.numpy(), expect, atol=1e-6)


def test_init_from_ego_and_gradients_reach_masker():
    m, _ = _model()
    m.start_epoch(1)
    assert m.tau == 1.0
    m.start_epoch(10)
    assert np.isclose(m.tau, 0.2)
    loss, parts = m.train_step_loss(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 4]), 1e-4)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.masker.parameters())
    assert "kl" in parts and "bpr" in parts
    d = m.eval_diagnostics()
    assert 0 <= d["alpha_mean_cand"] <= 1
