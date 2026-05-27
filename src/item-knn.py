"""
item_knn.py — Item-based KNN baseline recommender for the Fairness-Aware Music
Recommendation project (DSCI 641).

For each user, score every candidate artist by the sum of cosine similarities
between that artist and the artists in the user's train history. Recommend the
top-K after filtering already-heard. Per the proposal: "Arctic Monkeys and
The Strokes show up in a lot of the same listening histories, the algorithm
considers them similar."

Design decisions (locked in):
  - Cosine similarity over the BINARIZED user-item matrix. Raw play counts
    let single power users dominate similarity; binarization makes similarity
    reflect catalog-wide co-occurrence, which is what we actually want for
    implicit feedback. This is also Shakespeare et al. (2020)'s setup.
  - 50 neighbors per item (standard default, not tuned).
  - Artist catalog filtered to artists with >= 5 listeners. Below that
    threshold there isn't enough co-occurrence data to produce meaningful
    similarities. ALS will use the same filter for consistency.
  - implicit.nearest_neighbours.CosineRecommender for the sparse-matrix math.

Run `python src/item_knn.py` for an end-to-end smoke test against val.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent))
from metrics import DEFAULT_K, evaluate_all  # noqa: E402

# implicit emits some FutureWarnings about NumPy that aren't our concern
warnings.filterwarnings("ignore", category=FutureWarning, module="implicit")

try:
    from implicit.nearest_neighbours import CosineRecommender
except ImportError:
    sys.exit(
        "ERROR: `implicit` not installed.\n"
        "Run: uv add implicit   (or: pip install implicit --break-system-packages)"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

K_NEIGHBORS = 50  # how many similar items to keep per item
MIN_LISTENERS = 5  # drop artists with fewer than this many unique listeners
CANDIDATE_POOL = 500  # over-recommend, then filter heard, then take top K
DEFAULT_PROCESSED_DIR = Path("processed")


# ---------------------------------------------------------------------------
# Shared catalog filter (ALS will reuse this)
# ---------------------------------------------------------------------------


def filter_sparse_artists(
    train: pd.DataFrame,
    min_listeners: int = MIN_LISTENERS,
) -> tuple[pd.DataFrame, set[int]]:
    """Drop artists with fewer than `min_listeners` unique users in train.

    Returns the filtered train DataFrame and the set of kept artist_ids.
    The set is useful for filtering val/test later if needed.
    """
    counts = train.groupby("artist_id")["user_id"].nunique()
    keep = set(counts[counts >= min_listeners].index)
    filtered = train[train["artist_id"].isin(keep)]
    print(
        f"[filter] kept {len(keep):,} of {len(counts):,} artists "
        f"(>= {min_listeners} listeners); "
        f"{len(filtered):,} of {len(train):,} interactions retained"
    )
    return filtered, keep


# ---------------------------------------------------------------------------
# Sparse matrix construction
# ---------------------------------------------------------------------------


def build_sparse_matrix(
    train: pd.DataFrame,
    binarize: bool = True,
) -> tuple[csr_matrix, dict[int, int], dict[int, int]]:
    """Build a (n_users x n_items) CSR matrix from train interactions.

    Returns the matrix plus two ID maps:
        user_to_row:   user_id -> row index in the matrix
        artist_to_col: artist_id -> column index in the matrix

    Binarize=True replaces play counts with 1.0 (presence/absence). Required
    for cosine item-similarity to behave sensibly on implicit feedback.

    Note: implicit expects (users x items) for fitting. Internally it uses
    items x users for similarity computation, but you pass the standard
    user-item matrix to .fit().
    """
    users = train["user_id"].unique()
    artists = train["artist_id"].unique()

    user_to_row = {u: i for i, u in enumerate(users)}
    artist_to_col = {a: i for i, a in enumerate(artists)}

    rows = train["user_id"].map(user_to_row).to_numpy()
    cols = train["artist_id"].map(artist_to_col).to_numpy()
    data = (
        np.ones(len(train), dtype=np.float32)
        if binarize
        else train["play_count"].to_numpy(dtype=np.float32)
    )
    n_users = len(users)
    n_artists = len(artists)
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_artists),
    )
    nnz = matrix.nnz
    sparsity = 1 - nnz / (n_users * n_artists)
    print(
        f"[matrix] {n_users:,} users x {n_artists:,} artists, "
        f"{nnz:,} nonzeros (sparsity {sparsity:.6f})"
    )
    return matrix, user_to_row, artist_to_col


# ---------------------------------------------------------------------------
# Core recommender
# ---------------------------------------------------------------------------


def recommend_item_knn(
    train: pd.DataFrame,
    eligible_users: list[int],
    k: int = DEFAULT_K,
    k_neighbors: int = K_NEIGHBORS,
    min_listeners: int = MIN_LISTENERS,
    candidate_pool: int = CANDIDATE_POOL,
) -> dict[int, list[int]]:
    """Per-user top-K recommendations from item-based KNN.

    Contract matches recommend_popularity: in (train, users), out (user -> [artists]).
    """
    filtered_train, _ = filter_sparse_artists(train, min_listeners=min_listeners)
    matrix, user_to_row, artist_to_col = build_sparse_matrix(filtered_train, binarize=True)
    col_to_artist = {v: k_ for k_, v in artist_to_col.items()}

    print(f"[fit] training CosineRecommender with K={k_neighbors} neighbors...")
    t0 = time.perf_counter()
    model = CosineRecommender(K=k_neighbors)
    model.fit(matrix, show_progress=False)
    print(f"[fit] done ({time.perf_counter() - t0:.1f}s)")

    print(f"[recommend] generating top-{k} for {len(eligible_users):,} users...")
    t0 = time.perf_counter()
    recs: dict[int, list[int]] = {}
    missing_users = 0

    for user in eligible_users:
        row_idx = user_to_row.get(user)
        if row_idx is None:
            # User had all their train interactions on rare artists that got
            # filtered out. Can't recommend; record empty list.
            missing_users += 1
            recs[user] = []
            continue

        # filter_already_liked_items=True excludes items in user's train row
        # — this IS the discovery filter, no extra work needed.
        ids, _scores = model.recommend(
            row_idx,
            matrix[row_idx],
            N=candidate_pool,
            filter_already_liked_items=True,
        )
        # Map column indices back to artist_ids and trim to K
        recs[user] = [col_to_artist[int(c)] for c in ids[:k]]

    print(f"[recommend] done ({time.perf_counter() - t0:.1f}s)")
    if missing_users:
        print(
            f"[warn] {missing_users} user(s) had no train interactions on retained "
            f"artists; recommending empty list. These contribute zero to recall/NDCG."
        )

    return recs


# ---------------------------------------------------------------------------
# Smoke test (mirrors popularity.py structure)
# ---------------------------------------------------------------------------


def _load_processed(processed_dir: Path) -> tuple:
    required = ["train", "val", "test", "artist_gender", "already_heard"]
    paths = {name: processed_dir / f"{name}.parquet" for name in required}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        sys.exit(
            f"ERROR: missing parquet files in {processed_dir.resolve()}:\n"
            + "\n".join(f"  {p.name}" for p in missing)
            + "\nRun prepare_data.py first."
        )
    return tuple(pd.read_parquet(p) for p in paths.values())


def _to_user_artist_sets(df: pd.DataFrame) -> dict[int, set[int]]:
    return cast(
        dict[int, set[int]],
        df.groupby("user_id")["artist_id"].apply(set).to_dict(),
    )


def _print_results(r: dict, name: str = "ITEM-KNN BASELINE") -> None:
    print("\n" + "=" * 60)
    print(f"{name} @ K={r['k']} (eval: val)")
    print("=" * 60)
    print(f"  Recall@{r['k']}:  {r['recall@k']:.4f}")
    print(f"  NDCG@{r['k']}:    {r['ndcg@k']:.4f}")
    print()
    print("  Exposure share (recommendations):")
    for g, frac in r["exposure_share"].items():
        print(f"    {g:<8} {frac:.4f}")
    print()
    print("  Bias disparity (positive = amplification):")
    for g, val in r["bias_disparity"].items():
        sign = "+" if val >= 0 else ""
        print(f"    {g:<8} {sign}{val:.4f}")
    print("=" * 60)


def main(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> dict:
    print(f"[load] reading parquet from {processed_dir.resolve()}")
    t0 = time.perf_counter()
    train_df, val_df, _test_df, gender_df, _heard_df = _load_processed(processed_dir)
    print(f"[load] train={len(train_df):,}  val={len(val_df):,}  ({time.perf_counter() - t0:.1f}s)")

    eligible = val_df["user_id"].unique().tolist()
    print(f"[setup] {len(eligible):,} users to evaluate against val")

    top_k = recommend_item_knn(train_df, eligible, k=DEFAULT_K)

    print("[metrics] evaluating...")
    t0 = time.perf_counter()
    train_sets = _to_user_artist_sets(train_df)
    val_sets = _to_user_artist_sets(val_df)
    gender_lookup = cast(
        dict[int, str],
        dict(zip(gender_df["artist_id"], gender_df["gender"])),
    )
    results = evaluate_all(
        top_k=top_k,
        train=train_sets,
        test=val_sets,
        gender=gender_lookup,
        k=DEFAULT_K,
    )
    print(f"[metrics] done ({time.perf_counter() - t0:.1f}s)")

    _print_results(results)
    return results


if __name__ == "__main__":
    main()
