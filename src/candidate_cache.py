"""
candidate_cache.py — Persist ALS candidate (user, artist, score) tuples to
parquet so we can iterate on re-ranker variants without re-fitting ALS.

The cache is keyed only by file existence — if you change ALS hyperparameters,
delete the cache file manually and rerun. Cheap insurance against an obscure
bug class, and adequate for project-scale work.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CACHE = Path("processed") / "als_candidates.parquet"


def save_candidates(
    candidates: dict[int, list[tuple[int, float]]],
    path: Path = DEFAULT_CACHE,
) -> None:
    """Flatten {user: [(artist, score), ...]} into a long parquet table."""
    rows = []
    for user, cands in candidates.items():
        for rank, (artist, score) in enumerate(cands):
            rows.append((user, artist, float(score), rank))
    df = pd.DataFrame(rows, columns=["user_id", "artist_id", "score", "rank"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[cache] wrote {len(df):,} candidate rows to {path}")


def load_candidates(
    path: Path = DEFAULT_CACHE,
) -> dict[int, list[tuple[int, float]]]:
    """Inverse of save_candidates. Returns sorted-by-rank-asc per user."""
    df = pd.read_parquet(path)
    df = df.sort_values(["user_id", "rank"])
    out: dict[int, list[tuple[int, float]]] = {}
    for user, group in df.groupby("user_id"):
        out[int(user)] = [
            (int(a), float(s))
            for a, s in zip(group["artist_id"], group["score"])
        ]
    print(f"[cache] loaded {len(df):,} candidate rows from {path}")
    return out


def cache_exists(path: Path = DEFAULT_CACHE) -> bool:
    return path.exists()
