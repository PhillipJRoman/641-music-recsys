"""
als.py — Implicit ALS recommender for the Fairness-Aware Music Recommendation
project (DSCI 641).

Matrix factorization for implicit feedback (Hu, Koren, Volinsky 2008). The
user-item matrix is factored into latent vectors; recommendations are scored
as user_vector · item_vector. Confidence is scaled by play count via the
`alpha` parameter, so the recommender treats high-play-count interactions as
stronger signals — UNLIKE item-KNN, which binarized.

Design decisions (locked in):
  - implicit.als.AlternatingLeastSquares (Hu/Koren/Volinsky formulation).
  - DO NOT binarize plays. ALS uses play count as a confidence signal via
    alpha; binarizing throws that signal away.
  - Same >= 5 listener catalog filter as item-KNN, for consistency.
  - Standard hyperparameters: factors=50, regularization=0.01, iterations=15,
    alpha=40. Tune only if the standard run looks weak.

This module exposes TWO recommendation functions:
  recommend_als(...)              -> top-K, drop-in compatible with metrics.
  als_candidate_scores(...)       -> top-N candidates with their scores,
                                     for the re-ranker to consume later.

The candidate-scores helper means the re-ranker can reuse a single trained
ALS model rather than refit it.

Run `python src/als.py` for an end-to-end smoke test against val.
"""

from __future__ import annotations

import os
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

# implicit threading: by default it uses ALL cores, which is what we want for
# a one-off run. Set OPENBLAS_NUM_THREADS=1 BEFORE importing implicit to avoid
# nested-parallelism slowdowns. (If you've already imported numpy elsewhere
# this is a no-op, but it's cheap to set.)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

warnings.filterwarnings("ignore", category=FutureWarning, module="implicit")

try:
    from implicit.als import AlternatingLeastSquares
except ImportError:
    sys.exit(
        "ERROR: `implicit` not installed.\n"
        "Run: uv add implicit   (or: pip install implicit --break-system-packages)"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FACTORS = 100
REGULARIZATION = 0.01
ITERATIONS = 15
ALPHA = 10.0
MIN_LISTENERS = 5
CANDIDATE_POOL = 100  # for both top-K filtering AND re-ranker input
DEFAULT_PROCESSED_DIR = Path("processed")


# ---------------------------------------------------------------------------
# Catalog filter + sparse matrix
# (Same logic as item_knn.py — duplicated rather than shared, by choice.)
# ---------------------------------------------------------------------------


def filter_sparse_artists(
    train: pd.DataFrame,
    min_listeners: int = MIN_LISTENERS,
) -> tuple[pd.DataFrame, set[int]]:
    """Drop artists with fewer than `min_listeners` unique users."""
    counts = train.groupby("artist_id")["user_id"].nunique()
    keep = set(counts[counts >= min_listeners].index)
    filtered = train[train["artist_id"].isin(keep)]
    print(
        f"[filter] kept {len(keep):,} of {len(counts):,} artists "
        f"(>= {min_listeners} listeners); "
        f"{len(filtered):,} of {len(train):,} interactions retained"
    )
    return filtered, keep


def build_sparse_matrix(
    train: pd.DataFrame,
    binarize: bool = False,
) -> tuple[csr_matrix, dict[int, int], dict[int, int]]:
    """Build a (n_users x n_items) CSR matrix. ALS uses raw play counts."""
    users = train["user_id"].unique()
    artists = train["artist_id"].unique()

    user_to_row = {u: i for i, u in enumerate(users)}
    artist_to_col = {a: i for i, a in enumerate(artists)}

    rows = train["user_id"].map(user_to_row).to_numpy()
    cols = train["artist_id"].map(artist_to_col).to_numpy()
    data = (
        np.ones(len(train), dtype=np.float32)
        if binarize
        else np.log1p(train["play_count"].to_numpy(dtype=np.float32))
    )

    n_users = len(users)
    n_artists = len(artists)
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_artists),
    )
    nnz = len(data)
    sparsity = 1 - nnz / (n_users * n_artists)
    print(
        f"[matrix] {n_users:,} users x {n_artists:,} artists, "
        f"{nnz:,} nonzeros (sparsity {sparsity:.6f})"
    )
    return matrix, user_to_row, artist_to_col


# ---------------------------------------------------------------------------
# Training (shared by both recommend_als and als_candidate_scores)
# ---------------------------------------------------------------------------


class _TrainedALS:
    """Bundle of a fitted ALS model + the ID maps needed to interpret it.

    Returned by `train_als`. Both recommend_als and als_candidate_scores
    work from one of these, so a single fit can power both flows.
    """

    def __init__(
        self,
        model,
        matrix: csr_matrix,
        user_to_row: dict[int, int],
        artist_to_col: dict[int, int],
    ):
        self.model = model
        self.matrix = matrix
        self.user_to_row = user_to_row
        self.col_to_artist = {v: k for k, v in artist_to_col.items()}


def train_als(
    train: pd.DataFrame,
    factors: int = FACTORS,
    regularization: float = REGULARIZATION,
    iterations: int = ITERATIONS,
    alpha: float = ALPHA,
    min_listeners: int = MIN_LISTENERS,
) -> _TrainedALS:
    """Filter, build sparse matrix, fit ALS. Returns a bundle."""
    filtered, _ = filter_sparse_artists(train, min_listeners=min_listeners)
    matrix, user_to_row, artist_to_col = build_sparse_matrix(filtered, binarize=False)

    # ALS confidence: implicit scales nonzero entries by alpha internally.
    # We pass the raw play-count matrix; implicit handles 1 + alpha * c.
    print(
        f"[fit] training ALS (factors={factors}, reg={regularization}, "
        f"iters={iterations}, alpha={alpha})..."
    )
    t0 = time.perf_counter()
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        calculate_training_loss=False,
    )
    model.fit(matrix, show_progress=False)
    print(f"[fit] done ({time.perf_counter() - t0:.1f}s)")

    return _TrainedALS(model, matrix, user_to_row, artist_to_col)


# ---------------------------------------------------------------------------
# Top-K recommender (matches the contract of popularity / item_knn)
# ---------------------------------------------------------------------------


def recommend_als(
    train: pd.DataFrame,
    eligible_users: list[int],
    k: int = DEFAULT_K,
    candidate_pool: int = CANDIDATE_POOL,
    **als_kwargs,
) -> dict[int, list[int]]:
    """Train ALS, recommend top-K per user. Drop-in for metrics.evaluate_all."""
    bundle = train_als(train, **als_kwargs)
    candidates = als_candidate_scores(bundle, eligible_users, n=candidate_pool)
    # For pure ALS, top-K is just the first k candidates (already sorted by score).
    return {user: [a for a, _s in cands[:k]] for user, cands in candidates.items()}


# ---------------------------------------------------------------------------
# Candidate scores (for the re-ranker to consume)
# ---------------------------------------------------------------------------


def als_candidate_scores(
    bundle: _TrainedALS,
    eligible_users: list[int],
    n: int = CANDIDATE_POOL,
) -> dict[int, list[tuple[int, float]]]:
    """Top-N (artist_id, score) candidates per user, sorted by score desc.

    Already-heard artists are filtered out (filter_already_liked_items=True).
    The re-ranker will consume this directly: each user's list is the
    candidate pool, ALS score is the relevance signal, and the re-ranker
    rescores against a fairness objective.
    """
    print(f"[candidates] scoring top-{n} candidates for {len(eligible_users):,} users...")
    t0 = time.perf_counter()

    out: dict[int, list[tuple[int, float]]] = {}
    missing = 0
    for user in eligible_users:
        row_idx = bundle.user_to_row.get(user)
        if row_idx is None:
            missing += 1
            out[user] = []
            continue

        ids, scores = bundle.model.recommend(
            row_idx,
            bundle.matrix[row_idx],
            N=n,
            filter_already_liked_items=True,
        )
        out[user] = [(bundle.col_to_artist[int(c)], float(s)) for c, s in zip(ids, scores)]

    print(f"[candidates] done ({time.perf_counter() - t0:.1f}s)")
    if missing:
        print(
            f"[warn] {missing} user(s) had no train interactions on retained "
            f"artists; returning empty candidate list."
        )
    return out


# ---------------------------------------------------------------------------
# Smoke test
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


def _print_results(r: dict, name: str = "ALS BASELINE") -> None:
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

    top_k = recommend_als(train_df, eligible, k=DEFAULT_K)

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
