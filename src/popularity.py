"""
popularity.py — Popularity baseline recommender for the Fairness-Aware Music
Recommendation project (DSCI 641).

Every user gets the same global popularity ranking, with their already-heard
artists filtered out. This is the no-personalization floor: any model that
can't beat this isn't doing useful work. It's also useful as a fairness
reference point — popularity bias is a known channel through which gender
bias propagates, so the popularity baseline's bias-disparity number is the
"do nothing" benchmark the re-ranker has to improve on.

Run `python src/popularity.py` from the repo root for an end-to-end smoke test:
loads processed/, recommends, evaluates against val, prints all four metrics.

The signature of `recommend_popularity` is intentionally the contract that
ALS and KNN will follow: take (train_df, eligible_users), return
dict[user_id, list[artist_id]] of length K. Drop-in compatible with the
experiment loop and with metrics.evaluate_all.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# Import metrics from the same src/ folder. Works whether this is run as
# `python src/popularity.py` (script) or imported as `from src.popularity ...`.
sys.path.insert(0, str(Path(__file__).parent))
from metrics import DEFAULT_K, evaluate_all  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Candidate pool: precompute this many most-popular artists. The per-user
# filter removes already-heard from this pool, then takes the top K from
# what's left. 500 is plenty of headroom — even a power user filters out far
# fewer than 500 of the globally-most-popular artists specifically.
CANDIDATE_POOL = 500
DEFAULT_PROCESSED_DIR = Path("processed")


# ---------------------------------------------------------------------------
# Core recommender
# ---------------------------------------------------------------------------


def rank_by_popularity(train: pd.DataFrame, n: int = CANDIDATE_POOL) -> np.ndarray:
    """Return the top-`n` artist_ids by total play count in train, ranked.

    Plain total plays is the standard popularity signal and what the proposal
    implies. Alternatives (unique users, log plays) are not used here — this
    is supposed to be the simplest possible baseline.
    """
    top = (
        train.groupby("artist_id")["play_count"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index.to_numpy()
    )
    return top


def recommend_popularity(
    train: pd.DataFrame,
    eligible_users: list[int],
    k: int = DEFAULT_K,
    candidate_pool: int = CANDIDATE_POOL,
) -> dict[int, list[int]]:
    """Per-user top-K from the global popularity pool, with already-heard removed.

    Contract (matched by ALS / KNN later):
        in:   train interactions, list of users to recommend for, K
        out:  dict[user_id, list[artist_id]] of length K each

    The per-user signal here is ONLY the already-heard filter — every user is
    scored by the same global ranking. This is what makes it the "no
    personalization" baseline.
    """
    top_artists = rank_by_popularity(train, n=candidate_pool)

    # Build per-user heard sets ONCE. This is the expensive step. The user
    # loop after this is just set lookups, which is cheap.
    heard_by_user = train.groupby("user_id")["artist_id"].apply(set).to_dict()

    recs: dict[int, list[int]] = {}
    underfilled = 0

    for user in eligible_users:
        heard = heard_by_user.get(user, set())
        user_recs = [a for a in top_artists if a not in heard][:k]

        if len(user_recs) < k:
            # Shouldn't happen with a pool of 500 unless a user has heard
            # an enormous slice of the most popular artists. Count rather
            # than crash — useful diagnostic.
            underfilled += 1

        recs[user] = user_recs

    if underfilled:
        print(
            f"[warn] {underfilled} user(s) got fewer than {k} recommendations — "
            f"consider raising candidate_pool above {candidate_pool}"
        )

    return recs


# ---------------------------------------------------------------------------
# Smoke test: load everything, recommend, evaluate, print
# ---------------------------------------------------------------------------


def _load_processed(processed_dir: Path) -> tuple:
    """Load the five parquet files prepare_data.py produces."""
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
    """Turn a [user_id, artist_id, ...] dataframe into {user: {artists}}."""
    return cast(dict[int, set[int]], df.groupby("user_id")["artist_id"].apply(set).to_dict())


def main(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> dict:
    """End-to-end smoke test against val. Returns the metrics dict."""
    print(f"[load] reading parquet from {processed_dir.resolve()}")
    t0 = time.perf_counter()
    train_df, val_df, test_df, gender_df, _heard_df = _load_processed(processed_dir)
    print(
        f"[load] train={len(train_df):,}  val={len(val_df):,}  "
        f"test={len(test_df):,}  ({time.perf_counter() - t0:.1f}s)"
    )

    # Eligible = anyone with at least one val interaction. prepare_data.py
    # already enforced the min-interactions threshold; val users are the
    # ones we can actually evaluate.
    eligible = val_df["user_id"].unique().tolist()
    print(f"[setup] {len(eligible):,} users to evaluate against val")

    print(f"[model] recommending top-{DEFAULT_K} by popularity...")
    t0 = time.perf_counter()
    top_k = recommend_popularity(train_df, eligible, k=DEFAULT_K)
    print(f"[model] done ({time.perf_counter() - t0:.1f}s)")

    print("[metrics] evaluating...")
    t0 = time.perf_counter()
    train_sets = _to_user_artist_sets(train_df)
    val_sets = _to_user_artist_sets(val_df)
    gender_lookup = dict(zip(gender_df["artist_id"], gender_df["gender"]))

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


def _print_results(r: dict) -> None:
    print("\n" + "=" * 60)
    print(f"POPULARITY BASELINE @ K={r['k']} (eval: val)")
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


if __name__ == "__main__":
    main()
