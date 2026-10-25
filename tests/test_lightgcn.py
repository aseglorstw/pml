import numpy as np
import torch

from src.models.lightgcn import LightGCN

M, N = 5, 6
TRAIN = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4], [3, 0], [4, 5], [4, 2]])
SOCIAL = np.array([[0, 1], [0, 2], [1, 3], [2, 4]])


def dense_reference(emb, w, omega, L=3):
    R = np.zeros((M, N))
    R[TRAIN[:, 0], TRAIN[:, 1]] = 1
    du, di = R.sum(1), R.sum(0)
    Rn = R / np.sqrt(np.outer(du, di))
    S = np.zeros((M, M))
    S[SOCIAL[:, 0], SOCIAL[:, 1]] = w
    S[SOCIAL[:, 1], SOCIAL[:, 0]] = w
    ds = np.zeros(M)
    np.add.at(ds, SOCIAL.ravel(), 1)  # static degree: count of edges, not sum of w
    Sn = S / np.sqrt(np.outer(ds, ds))
    eu, ei = emb[:M], emb[M:]
    lu, li = [eu], [ei]
    for _ in range(L):
        eu, ei = Rn @ ei + omega * Sn @ eu, Rn.T @ eu
        lu.append(eu)
        li.append(ei)
    return np.mean(lu, 0), np.mean(li, 0)


def _model(social=True):
    torch.manual_seed(0)
    return LightGCN(M, N, TRAIN, dim=4, n_layers=3, social_edges=SOCIAL if social else None,
                    omega_s=0.5)


def test_matches_dense_equation_8():
    m = _model()
    w = torch.tensor([1.0, 0.3, 0.0, 0.7])
    eu, ei = m.propagate(w)
    ru, ri = dense_reference(m.emb.weight.detach().double().numpy(), w.numpy(), 0.5)
    assert np.allclose(eu.detach().numpy(), ru, atol=1e-5)
    assert np.allclose(ei.detach().numpy(), ri, atol=1e-5)
    # w=None means all ones
    eu1, _ = m.propagate(None)
    ru1, _ = dense_reference(m.emb.weight.detach().double().numpy(), np.ones(4), 0.5)
    assert np.allclose(eu1.detach().numpy(), ru1, atol=1e-5)


def test_zero_weights_equal_plain_lightgcn():
    m, m0 = _model(True), _model(False)
    eu, ei = m.propagate(torch.zeros(len(SOCIAL)))
    eu0, ei0 = m0.propagate()
    assert torch.allclose(eu, eu0, atol=1e-6) and torch.allclose(ei, ei0, atol=1e-6)


def test_gradient_reaches_edge_weights():
    m = _model()
    w = torch.full((len(SOCIAL),), 0.5, requires_grad=True)
    loss, _ = m.bpr_loss(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([5, 4]), 1e-4, w)
    loss.backward()
    assert w.grad is not None and w.grad.abs().sum() > 0


def gbsr_dense_reference(emb, w, L=3):
    """Official GBSR backbone: A = [[S, R], [R^T, 0]], D = row sums, Â = D^-1/2 A D^-1/2 (unmasked),
    masked graph multiplies the social entries of Â by w; E^{l+1} = Â_w E^l, mean over layers."""
    R = np.zeros((M, N))
    R[TRAIN[:, 0], TRAIN[:, 1]] = 1
    S = np.zeros((M, M))
    S[SOCIAL[:, 0], SOCIAL[:, 1]] = 1
    S[SOCIAL[:, 1], SOCIAL[:, 0]] = 1
    A = np.block([[S, R], [R.T, np.zeros((N, N))]])
    d = A.sum(1)
    An = A / np.sqrt(np.outer(d, d))
    W = np.zeros((M, M))
    W[SOCIAL[:, 0], SOCIAL[:, 1]] = w
    W[SOCIAL[:, 1], SOCIAL[:, 0]] = w
    An[:M, :M] *= W
    e, layers = emb, [emb]
    for _ in range(L):
        e = An @ e
        layers.append(e)
    out = np.mean(layers, 0)
    return out[:M], out[M:]


def test_gbsr_form_matches_official_joint_adjacency():
    torch.manual_seed(0)
    m = LightGCN(M, N, TRAIN, dim=4, n_layers=3, social_edges=SOCIAL, omega_s=0.5, social_form="gbsr")
    emb = m.emb.weight.detach().double().numpy()
    for w in (np.ones(len(SOCIAL)), np.array([1.0, 0.3, 0.0, 0.7])):
        eu, ei = m.propagate(torch.tensor(w, dtype=torch.float32))
        ru, ri = gbsr_dense_reference(emb, w)
        assert np.allclose(eu.detach().numpy(), ru, atol=1e-5) and np.allclose(ei.detach().numpy(), ri, atol=1e-5)
