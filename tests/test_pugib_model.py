import numpy as np
import torch

from src.models.pugib import PUGIB, binary_concrete, kl_bernoulli, pair_features

M, N = 5, 6
TRAIN = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4], [3, 0], [4, 5], [4, 2]])
SUP_OBS = np.array([[0, 1], [0, 2], [1, 3], [2, 4]])
SUP_CAND = np.array([[0, 3], [1, 4]])
SUPPORT = np.concatenate([SUP_OBS, SUP_CAND])
IS_OBS = np.array([1, 1, 1, 1, 0, 0], bool)
RHO = np.where(IS_OBS, 0.9, 0.3)


def _model(beta=0.1, hsic=0.0, cand=True):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(M, 4))
    sup = SUPPORT if cand else SUP_OBS
    obs = IS_OBS if cand else IS_OBS[:4]
    feats = pair_features(emb, sup, np.ones(len(sup)), obs)
    torch.manual_seed(0)
    return PUGIB(M, N, TRAIN, sup, 4, RHO[: len(sup)], feats, dim=4, beta=beta, hsic_lambda=hsic, hsic_sigma=2.5,
                 batch_size=4, tau_epochs=10, omega_c=0.25,
                 in_prec=np.array([0, 0, 0, 0, 1, 0], bool)[: len(sup)],
                 in_pest=np.zeros(len(sup), bool)), emb


def test_features_symmetric():
    emb = np.random.default_rng(1).normal(size=(M, 4))
    f1 = pair_features(emb, SUPPORT, np.arange(float(len(SUPPORT))), IS_OBS)
    f2 = pair_features(emb, SUPPORT[:, ::-1], np.arange(float(len(SUPPORT))), IS_OBS)
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


def test_channels_and_cold_start():
    m, _ = _model()
    b = m.backbone
    assert b.social.n_edges == 4 and b.social_c is not None and b.social_c.n_edges == 2
    # candidates do not enter the joint normalization: same adjacency as a model without candidates
    m0, _ = _model(cand=False)
    assert torch.allclose(b.adj.to_dense(), m0.backbone.adj.to_dense())
    assert torch.allclose(b.social.norm, m0.backbone.social.norm)
    assert abs(m.emb.weight.std().item() - 0.1) < 0.05  # cold start, no copy of ẽ


def test_gradients_hsic_kl_and_diagnostics():
    m, _ = _model(beta=0.1, hsic=5.0)
    m.start_epoch(1)
    assert m.tau == 1.0
    m.start_epoch(10)
    assert np.isclose(m.tau, 0.2)
    loss, parts = m.train_step_loss(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 4]), 1e-4)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.masker.parameters())
    assert {"bpr", "kl", "hsic"} <= set(parts)
    d = m.eval_diagnostics()
    assert 0 <= d["alpha_mean_cand"] <= 1 and "auroc_cand_prec" in d
    m2, _ = _model(cand=False)  # PU-GIB-Prune
    l2, _ = m2.train_step_loss(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 4]), 1e-4)
    assert torch.isfinite(l2) and m2.backbone.social_c is None
