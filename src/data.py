"""Raw data loading and preprocessing for HetRec 2011 LastFM.

Produces implicit-feedback interactions and a symmetric, self-loop-free social graph,
with users remapped to 0..M-1 and items to 0..N-1.
"""
from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

LASTFM_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"


@dataclass
class Dataset:
    n_users: int
    n_items: int
    interactions: np.ndarray  # (nnz, 2) int64 [user, item], unique
    social: np.ndarray  # (E, 2) int64 undirected edges, u < v, unique

    def stats(self) -> dict:
        u_deg = np.bincount(self.interactions[:, 0], minlength=self.n_users)
        i_deg = np.bincount(self.interactions[:, 1], minlength=self.n_items)
        s_deg = np.bincount(self.social.ravel(), minlength=self.n_users)

        def desc(x):
            return {"min": int(x.min()), "max": int(x.max()), "mean": float(x.mean()),
                    "median": float(np.median(x))}

        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_interactions": int(len(self.interactions)),
            "n_social_edges_undirected": int(len(self.social)),
            "interaction_density": len(self.interactions) / (self.n_users * self.n_items),
            "social_density": 2 * len(self.social) / (self.n_users * (self.n_users - 1)),
            "user_interaction_degree": desc(u_deg),
            "item_interaction_degree": desc(i_deg),
            "user_social_degree": desc(s_deg),
            "users_without_social_edges": int((s_deg == 0).sum()),
        }


def download_lastfm(raw_dir: str) -> str:
    """Download and unzip LastFM into raw_dir/lastfm. Returns the extracted folder."""
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "hetrec2011-lastfm-2k.zip")
    out = os.path.join(raw_dir, "lastfm")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(LASTFM_URL, zip_path)
    if not os.path.exists(os.path.join(out, "user_artists.dat")):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
    return out


def symmetrize_edges(pairs: np.ndarray) -> np.ndarray:
    """Return unique undirected edges (u < v) without self-loops."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    und = np.unique(np.stack([lo, hi], 1), axis=0)
    return und


def build_dataset(ui: pd.DataFrame, uu: pd.DataFrame) -> Dataset:
    """ui: columns [user, item, (weight)]; uu: columns [user, friend]. Raw ids."""
    ui = ui[["user", "item"]].drop_duplicates()  # any listen count -> 1
    users = np.sort(ui["user"].unique())
    items = np.sort(ui["item"].unique())
    umap = {u: k for k, u in enumerate(users)}
    imap = {i: k for k, i in enumerate(items)}
    inter = np.stack([ui["user"].map(umap).to_numpy(), ui["item"].map(imap).to_numpy()], 1)
    inter = np.unique(inter.astype(np.int64), axis=0)

    uu = uu[uu["user"].isin(umap) & uu["friend"].isin(umap)]
    pairs = np.stack([uu["user"].map(umap).to_numpy(), uu["friend"].map(umap).to_numpy()], 1)
    social = symmetrize_edges(pairs)
    return Dataset(len(users), len(items), inter, social)


def load_lastfm(raw_dir: str) -> Dataset:
    folder = download_lastfm(raw_dir)
    ui = pd.read_csv(os.path.join(folder, "user_artists.dat"), sep="\t")
    ui = ui.rename(columns={"userID": "user", "artistID": "item", "weight": "weight"})
    uu = pd.read_csv(os.path.join(folder, "user_friends.dat"), sep="\t")
    uu = uu.rename(columns={"userID": "user", "friendID": "friend"})
    return build_dataset(ui, uu)


def save_stats(ds: Dataset, path: str) -> dict:
    s = ds.stats()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(s, f, indent=2)
    return s


# ------------------------------------------------------------------ Douban-Book (GBSR release) ---
GBSR_REPO = "https://github.com/yimutianyang/KDD24-GBSR"
_SAFE_GLOBALS = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                 ("numpy", "ndarray"), ("numpy", "dtype"), ("collections", "defaultdict"),
                 ("builtins", "list"), ("builtins", "set"), ("builtins", "dict")}


def safe_load_pickled_npy(path: str):
    """Load an object .npy (pickled) allowing only plain numpy/container types.

    np.load(allow_pickle=True) would execute arbitrary code from a downloaded file;
    this unpickler refuses every global outside a small whitelist.
    """
    import pickle

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in _SAFE_GLOBALS:
                raise pickle.UnpicklingError(f"blocked global {module}.{name}")
            if module.startswith("numpy") and module.endswith("multiarray"):
                return np.core.multiarray._reconstruct if hasattr(np, "core") else np._core.multiarray._reconstruct
            return super().find_class(module, name)

    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        hdr = (np.lib.format.read_array_header_1_0 if version == (1, 0)
               else np.lib.format.read_array_header_2_0)(f)
        assert hdr[2] == np.dtype("O"), hdr
        arr = Restricted(f).load()
    return arr.item() if isinstance(arr, np.ndarray) and arr.shape == () else arr


def load_douban_book(raw_dir: str) -> Dataset:
    """Douban-Book as released with GBSR (already filtered, implicit feedback).

    GBSR ships an 80/20 train/test split; we merge both parts and re-split with our protocol.
    """
    return load_gbsr_dataset(raw_dir, "douban_book")


def load_gbsr_dataset(raw_dir: str, name: str) -> Dataset:
    """A dataset released with GBSR (douban_book, epinions, yelp): train + test merged, re-split by us."""
    folder = os.path.join(raw_dir, "gbsr_repo", "src", "datasets", name)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"clone {GBSR_REPO} into {os.path.join(raw_dir, 'gbsr_repo', 'src')}")
    pairs = []
    for name in ["traindata.npy", "testdata.npy"]:
        d = safe_load_pickled_npy(os.path.join(folder, name))
        pairs += [(u, i) for u, items in d.items() for i in items]
    ui = pd.DataFrame(pairs, columns=["user", "item"])
    soc = safe_load_pickled_npy(os.path.join(folder, "user_users_d.npy"))
    uu = pd.DataFrame([(u, v) for u, vs in soc.items() for v in vs], columns=["user", "friend"])
    print(f"GBSR release: {len(ui)} interaction rows ({ui.drop_duplicates().shape[0]} unique), "
          f"{ui.user.nunique()} users, {ui.item.nunique()} items (max ids {ui.user.max()}, {ui.item.max()}); "
          f"{len(uu)} directed social rows")
    return build_dataset(ui, uu)
