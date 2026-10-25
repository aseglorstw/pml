import math

import numpy as np
import torch

from src.metrics import evaluate, recall_ndcg, to_csr


def test_hand_computed_recall_ndcg():
    # 3 users, 6 items. User 2 has no held-out items and must be ignored.
    truth = to_csr(np.array([[0, 1], [0, 3], [0, 5], [1, 0]]), 3, 6)
    topk = np.array([
        [1, 2, 3, 0],  # user 0: hits at ranks 1 and 3
        [4, 5, 2, 3],  # user 1: no hit
        [0, 1, 2, 3],
    ])
    m = recall_ndcg(topk, truth, ks=(2, 4))
    # K=2: user0 1 hit of 3; user1 0 of 1
    assert math.isclose(m["recall@2"], (1 / 3 + 0) / 2)
    idcg2 = 1 + 1 / math.log2(3)
    assert math.isclose(m["ndcg@2"], (1 / idcg2 + 0) / 2)
    # K=4: user0 2 of 3; DCG = 1 + 1/log2(4); IDCG = 1 + 1/log2(3) + 1/log2(4)
    assert math.isclose(m["recall@4"], (2 / 3) / 2)
    dcg = 1 + 1 / math.log2(4)
    idcg4 = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(m["ndcg@4"], (dcg / idcg4) / 2)


def test_evaluate_excludes_train_items():
    # Item 0 has the highest score but is a train item; it must not be ranked.
    scores = torch.tensor([[10.0, 1.0, 5.0, 0.0]])
    m = evaluate(lambda u: scores[u], 1, 4, truth_pairs=np.array([[0, 2]]),
                 exclude_pairs=[np.array([[0, 0]])], ks=(1,))
    assert m["recall@1"] == 1.0 and m["ndcg@1"] == 1.0
    # without exclusion item 0 wins and we miss
    m = evaluate(lambda u: scores[u], 1, 4, truth_pairs=np.array([[0, 2]]),
                 exclude_pairs=[], ks=(1,))
    assert m["recall@1"] == 0.0


def test_exclusion_does_not_leak_across_users_with_shared_scores():
    # Same (expanded) scores for both users; user 0's train item must stay rankable for user 1.
    pop = torch.tensor([5.0, 3.0, 1.0])
    m = evaluate(lambda u: pop.expand(len(u), -1), 2, 3,
                 truth_pairs=np.array([[0, 1], [1, 0]]),
                 exclude_pairs=[np.array([[0, 0]])], ks=(1,))
    assert m["recall@1"] == 1.0
