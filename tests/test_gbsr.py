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
