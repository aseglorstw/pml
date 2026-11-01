"""Figure: estimated vs true noise share (mixture-proportion estimator, scripts/mixture_prior.py).

Synthetic: worlds with c = 0.5 and false_frac in {0, 0.1, ..., 0.7}, seeds 0-2 (the interactions of a generator
seed do not depend on false_frac, so the interaction-only LightGCN of synth50 is reused); truth = share of false
ties among the training ties. Real: random ties injected into the training graph (scripts/make_noisy_real.py);
truth = injected share; estimate = (estimated noise ties on the noisy graph - on the clean graph) / ties.
Douban-Book is computed here; the Epinions numbers come from the cluster runs (results/real_epinions*_report.md
and results/logs, seed 0), listed below.
Writes results/figures/estimator_vs_truth.{pdf,png} and results/estimator_vs_truth.csv.
Usage: python scripts/estimator_figure.py
"""
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SEEDS = [0, 1, 2]
# Epinions seed 0, from the cluster (prep lines of results/logs/real_epinions*.log): (q_hat, n) per group
EPINIONS = {"clean": [(0.311, 235927), (0.492, 31870)],
            0.5: {"groups": [(0.352, 259266), (0.916, 142429)], "injected": 133898},
            1.0: {"groups": [(0.418, 298157), (0.968, 237437)], "injected": 267797}}


def run(*args):
    subprocess.run([PY, *args], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def est_noise_ties(info):
    return sum(g["q_hat"] * g["n"] for g in info["groups"].values())


def synthetic():
    rows = []
    for f in FRACS:
        name = f"synth_sweep_f{int(round(f * 100))}"
        for s in SEEDS:
            d = os.path.join(ROOT, "data/processed", name, f"seed_{s}")
            if not os.path.exists(os.path.join(d, "pugib/mixture_prior.json")):
                run("scripts/make_synthetic.py", "--name", name, "--seed", str(s), "--false-frac", str(f))
                shutil.copy(os.path.join(ROOT, f"results/checkpoints/lightgcn_synth50_seed{s}.pt"),
                            os.path.join(ROOT, f"results/checkpoints/lightgcn_{name}_seed{s}.pt"))
                run("scripts/mixture_prior.py", "--dataset", name, "--seeds", str(s))
            info = json.load(open(os.path.join(d, "pugib/mixture_prior.json")))
            g = info["groups"]
            rows.append({"source": "synthetic", "level": f, "seed": s, "true": info["truth"]["false_share"],
                         "est": est_noise_ties(info) / info["n_ties"],
                         "true_omega": info["truth"]["false_share_in_omega"], "est_omega": g.get("in_omega", {}).get("q_hat"),
                         "true_bridges": info["truth"]["false_share_bridges"], "est_bridges": g["bridges"]["q_hat"]})
    return rows


def douban():
    rows = []
    clean = json.load(open(os.path.join(ROOT, "data/processed/douban_book/seed_0/pugib/mixture_prior.json")))
    for f in (0.5, 1.0):
        name = f"douban_book_n{int(round(f * 100))}"
        p = os.path.join(ROOT, "data/processed", name, "seed_0/pugib/mixture_prior.json")
        if not os.path.exists(p):
            run("scripts/make_noisy_real.py", "--dataset", "douban_book", "--fracs", str(f), "--seeds", "0")
            run("scripts/mixture_prior.py", "--dataset", name, "--seeds", "0")
        info = json.load(open(p))
        inj = json.load(open(os.path.join(ROOT, "data/processed", name, "seed_0/meta.json")))["injected"]
        n = info["n_ties"]
        rows.append({"source": "Douban-Book", "level": f, "seed": 0, "true": inj / n,
                     "est": (est_noise_ties(info) - est_noise_ties(clean)) / n})
    return rows


def epinions():
    clean = sum(q * n for q, n in EPINIONS["clean"])
    rows = []
    for f in (0.5, 1.0):
        e = EPINIONS[f]
        n = sum(k for _, k in e["groups"])
        rows.append({"source": "Epinions", "level": f, "seed": 0, "true": e["injected"] / n,
                     "est": (sum(q * k for q, k in e["groups"]) - clean) / n})
    return rows


def main():
    syn, real = synthetic(), douban() + epinions()
    os.makedirs(os.path.join(ROOT, "results/figures"), exist_ok=True)
    with open(os.path.join(ROOT, "results/estimator_vs_truth.csv"), "w") as fh:
        keys = ["source", "level", "seed", "true", "est", "true_omega", "est_omega", "true_bridges", "est_bridges"]
        fh.write(",".join(keys) + "\n")
        for r in syn + real:
            fh.write(",".join("" if r.get(k) is None else (f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]))
                              for k in keys) + "\n")
    # --- figure (ACM single column) ---
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.plot([0, 0.8], [0, 0.8], ls="--", lw=1, color="#a3a29d", zorder=1, label="perfect estimate")
    t = np.array([[r["true"] for r in syn if r["level"] == f] for f in FRACS])
    e = np.array([[r["est"] for r in syn if r["level"] == f] for f in FRACS])
    tm, em = t.mean(1), e.mean(1)
    ax.errorbar(tm, em, yerr=[em - e.min(1), e.max(1) - em], fmt="o", ms=4, lw=1, capsize=2, color="#2a78d6",
                mec="#fcfcfb", mew=0.8, zorder=3, label="synthetic (3 seeds)")
    # Douban: open squares (drawn first, larger); Epinions: filled triangles on top - the points nearly coincide
    for src, color, mk, ms, mfc, z in (("Douban-Book", "#1baf7a", "s", 8, "none", 4), ("Epinions", "#eb6834", "^", 4.5, None, 5)):
        pts = [r for r in real if r["source"] == src]
        ax.plot([r["true"] for r in pts], [r["est"] for r in pts], mk, ms=ms, color=color, mfc=mfc or color,
                mec=color if mfc == "none" else "#0b0b0b", mew=1.3 if mfc == "none" else 0.5, ls="none",
                zorder=z, label=f"{src}, injected ties")
    ax.set_xlim(-0.02, 0.8)
    ax.set_ylim(-0.02, 0.8)
    ax.set_xlabel("true share of false ties")
    ax.set_ylabel("estimated share")
    ax.grid(True, lw=0.4, color="#e4e3df")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(x) for x in ("synthetic (3 seeds)", "Epinions, injected ties", "Douban-Book, injected ties",
                                    "perfect estimate")]
    ax.legend([h[i] for i in order], [lab[i] for i in order], loc="upper left", frameon=False, handletextpad=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, f"results/figures/estimator_vs_truth.{ext}"), dpi=200)
    print("synthetic (mean over seeds): " + ", ".join(f"{a:.3f}->{b:.3f}" for a, b in zip(tm, em)))
    print("max |est - true| synthetic per seed: %.3f" % np.abs(e - t).max())
    for r in real:
        print(f"{r['source']} +{int(r['level'] * 100)}%: injected share true {r['true']:.3f}, estimated {r['est']:.3f}")


if __name__ == "__main__":
    main()
