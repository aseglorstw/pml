import numpy as np
import torch

from src.models.gbsr import GBSR, hsic, kernel_matrix

M, N = 5, 6
TRAIN = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4], [3, 0], [4, 5], [4, 2]])
SOCIAL = np.array([[0, 1], [0, 2], [1, 3], [2, 4]])


def _loss(detach):
    torch.manual_seed(0)
    m = GBSR(M, N, TRAIN, SOCIAL, beta=1.0, sigma=0.25, detach_mask=detach, batch_size=4, dim=4)
    loss, parts = m.train_step_loss(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 3]), torch.tensor([5, 4, 0]), 1e-4)
    loss.backward()
    return m, parts


def test_detach_switch_controls_masker_gradient():
    m, parts = _loss(detach=True)  # official PyTorch behaviour
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in [m.linear_1.weight, m.linear_2.weight])
    m, parts = _loss(detach=False)
    assert m.linear_1.weight.grad.abs().sum() > 0 and parts["hsic"] > 0


def test_one_gate_per_direction_and_hsic_identical_inputs():
    torch.manual_seed(0)
    m = GBSR(M, N, TRAIN, SOCIAL, dim=4)
    g = m.gates(sample=False)
    assert g.shape[0] == 2 * len(SOCIAL) and (g >= 0.5).all() and (g <= 1.5).all()
    x = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    K = kernel_matrix(x, 0.25)
    assert hsic(K, K, 8) > hsic(K, kernel_matrix(torch.nn.functional.normalize(torch.randn(8, 4), dim=1), 0.25), 8)


def official_dense_propagation(emb, train, social_dir, M, N, weights=None, L=3):
    """numpy replica of the official get_uu_i_matrix + masked forward (torch_version/GBSR.py)."""
    n = M + N
    A = np.zeros((n, n))
    for u, i in train:
        A[u, M + i] += 1
        A[M + i, u] += 1
    for k, (u, v) in enumerate(social_dir):
        A[u, v] += 1 if weights is None else 1  # structure; weights multiply normalized values below
    D = A.sum(1)
    D[D == 0] = 1
    An = A / np.sqrt(D)[None, :] / np.sqrt(D)[:, None]
    if weights is not None:
        for (u, v), w in zip(social_dir, weights):
            An[u, v] *= w
    e, layers = emb, [emb]
    for _ in range(L):
        e = An @ e
        layers.append(e)
    out = np.mean(layers, 0)
    return out[:M], out[M:]


def test_directed_social_matches_official_dense_adjacency():
    from src.models.lightgcn import LightGCN
    social_dir = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [3, 3], [1, 3], [3, 1], [4, 4]])  # with self-loops
    torch.manual_seed(0)
    m = LightGCN(M, N, TRAIN, dim=4, social_edges=social_dir, social_form="gbsr", social_directed=True,
                 init_std=0.01)
    emb = m.emb.weight.detach().double().numpy()
    for w in (None, np.random.default_rng(0).uniform(0.5, 1.5, len(social_dir))):
        eu, ei = m.propagate(None if w is None else torch.tensor(w, dtype=torch.float32))
        ru, ri = official_dense_propagation(emb, TRAIN, social_dir, M, N, w)
        assert np.allclose(eu.detach().numpy(), ru, atol=1e-6) and np.allclose(ei.detach().numpy(), ri, atol=1e-6)


def test_gbsr_official_options():
    social_dir = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [3, 3]])
    torch.manual_seed(0)
    g = GBSR(M, N, TRAIN, social_dir, social_form="gbsr", social_directed=True, init_std=0.01,
             eval_gate="sample", gate_mode="fixed1", hsic_items=False, detach_mask=True, batch_size=4, dim=4)
    assert torch.allclose(g.gates(sample=True), torch.ones(5))
    loss, parts = g.train_step_loss(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 3]), torch.tensor([5, 4, 0]), 1e-4)
    assert torch.isfinite(loss) and parts["hsic"] > 0
    assert abs(g.emb.weight.std().item() - 0.01) < 0.005
