import numpy as np

from src.pugib_prep import (adjacency, build_omega, channel_estimates, fit_isotonic,
                            isin_pairs, prior_from, resource_allocation, topk_pool,
                            uninformed_prior)

# Toy graph: triangle 0-1-2, tail 2-3-4.   deg = [2, 2, 3, 2, 1]
EDGES = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]])


def test_omega_ra_and_bridges_by_hand():
    A = adjacency(EDGES, 5)
    omega = build_omega(A)
    # common neighbours: (0,1):{2} (0,2):{1} (0,3):{2} (1,2):{0} (1,3):{2} (2,4):{3}
    # (2,3) and (3,4) share no neighbour -> bridges
    assert omega.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 4]]
    r = resource_allocation(A, omega)
    assert np.allclose(r, [1 / 3, 1 / 2, 1 / 3, 1 / 2, 1 / 3, 1 / 2])
    y = isin_pairs(omega, EDGES, 5)
    assert y.tolist() == [True, True, False, True, False, False]
    bridges = EDGES[~isin_pairs(EDGES, omega, 5)]
    assert bridges.tolist() == [[2, 3], [3, 4]]
    assert (r > 0).all()  # RA is positive exactly on Omega


def test_candidate_pool_is_symmetric_union():
    pairs = np.array([[0, 1], [0, 2], [1, 2], [2, 3]])
    s = np.array([0.9, 0.8, 0.1, 0.5])
    pool = topk_pool(pairs, s, K=1, rng=np.random.default_rng(0))
    # user0 -> (0,1); user1 -> (0,1); user2 -> (0,2); user3 -> (2,3)
    # (0,2) is kept because user 2 ranks it first, even though user 0 does not.
    assert pool.tolist() == [[0, 1], [0, 2], [2, 3]]
    assert (pool[:, 0] < pool[:, 1]).all() and len(np.unique(pool, axis=0)) == len(pool)


def test_pool_symmetric_on_random_graph():
    rng = np.random.default_rng(0)
    n = 60
    p = np.array([[i, j] for i in range(n) for j in range(i + 1, n)])
    p = p[rng.random(len(p)) < 0.2]
    pool = topk_pool(p, rng.random(len(p)), K=3, rng=rng)
    assert (pool[:, 0] < pool[:, 1]).all() and len(np.unique(pool, axis=0)) == len(pool)
    # every user with candidates gets at least min(K, #candidates) of its top pairs in the pool
    deg_pool = np.bincount(pool.ravel(), minlength=n)
    deg_all = np.bincount(p.ravel(), minlength=n)
    assert (deg_pool >= np.minimum(3, deg_all)).all()


def test_isotonic_mean_equals_positive_rate():
    rng = np.random.default_rng(1)
    s = np.round(rng.normal(size=5000), 2)  # with ties
    y = (rng.random(5000) < 1 / (1 + np.exp(-2 * s))).astype(int)
    _, g = fit_isotonic(s, y)
    assert np.isclose(g.mean(), y.mean())
    assert (np.diff(g[np.argsort(s, kind="stable")]) >= -1e-12).all()


def test_channel_estimator_recovers_truth_under_separability():
    c, eps, pi = 0.6, 0.01, 0.1
    p_a = pi * c + (1 - pi) * eps
    rho = prior_from(p_a, c, pi)["rho_obs"]
    g_val = rho * c + (1 - rho) * eps  # E[g_val] under separability
    est = channel_estimates(p_a, g_val, kappa=eps / p_a)
    assert np.isclose(est["c_hat"], c) and np.isclose(est["pi_hat"], pi)
    # eps -> 0 gives the classical Elkan-Noto c = g_val
    assert np.isclose(channel_estimates(p_a, g_val, kappa=1e-12)["c_hat"], g_val)


def test_uninformed_prior_is_valid():
    out = uninformed_prior(0.03, 0.05)
    assert out["valid"] and np.isclose(out["c_hat"], 0.515)


def test_anchor_recovers_top_rate():
    from src.pugib_prep import anchor_estimates, anchor_g_top
    rng = np.random.default_rng(0)
    s = rng.random(20000)
    y = rng.random(20000) < np.where(s > 0.9, 0.4, 0.02)  # top 10% has rate 0.4
    a = anchor_g_top(s, y, np.random.default_rng(1), min_size=1000)
    assert abs(a["g_top"] - 0.4) < 0.05
    e = anchor_estimates(p_a=y.mean(), g_val=0.1, g_top=a["g_top"], kappa=0.05)
    assert e["anchor_active"] and e["c_hat"] == a["g_top"] and e["valid"]
