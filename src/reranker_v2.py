"""
reranker_v2.py — Run the re-ranker with BOTH target modes ("user" input
matching and "global" 30% female floor) in a single sweep, using cached
ALS candidates to skip re-fitting.

Imports the greedy logic from reranker.py — no duplication. Only the
target distribution is mode-dependent; the greedy MMR construction is
identical regardless of where the target comes from.

Run after `als.py` has saved candidates via candidate_cache, OR run this
script directly — it will fit ALS once and cache candidates if none exist.

Output:
  results/reranker-sweep-user.csv     (input-matching, redundant with run 1)
  results/reranker-sweep-global.csv   (30% female floor, the new variant)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import cast

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import DEFAULT_K, FAIRNESS_GENDERS, evaluate_all  # noqa: E402
from reranker import (  # noqa: E402  -- reuse what works
    rerank_user,
    user_target_distribution,
    _to_user_artist_sets,
    _load_processed,
    _flatten_for_csv,
    _print_sweep_table,
)
from als import train_als, als_candidate_scores, DEFAULT_PROCESSED_DIR  # noqa: E402
from candidate_cache import save_candidates, load_candidates, cache_exists  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAMBDA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Global target: 30% female floor (proposal direction).
# Remaining 70% split between male and mixed roughly in proportion to
# personalized ALS exposure: ALS gave 77.7% male and 12.1% mixed, so within
# the remaining 70%, male:mixed is ~87:13. That gives male=0.609, mixed=0.091.
# Rounded for readability:
GLOBAL_TARGET = {
    "male": 0.60,
    "female": 0.30,
    "mixed": 0.10,
}

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Target dispatch
# ---------------------------------------------------------------------------

def get_target(
    mode: str,
    train_artists: set[int],
    gender: dict[int, str],
) -> dict[str, float] | None:
    """Return the target gender distribution for the given mode.

    mode="user":   user's input distribution (None if no known-gender history)
    mode="global": the same GLOBAL_TARGET for every user
    """
    if mode == "user":
        return user_target_distribution(train_artists, gender)
    elif mode == "global":
        return dict(GLOBAL_TARGET)
    else:
        raise ValueError(f"unknown target mode: {mode!r}")


def rerank_all(
    candidates: dict[int, list[tuple[int, float]]],
    train_sets: dict[int, set[int]],
    gender: dict[int, str],
    lam: float,
    mode: str,
    k: int = DEFAULT_K,
) -> dict[int, list[int]]:
    """Re-rank every user's candidate pool at a single (λ, mode)."""
    out: dict[int, list[int]] = {}
    for user, cands in candidates.items():
        target = get_target(mode, train_sets.get(user, set()), gender)
        out[user] = rerank_user(cands, target, gender, lam, k)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _get_candidates(train_df, eligible) -> dict[int, list[tuple[int, float]]]:
    """Load from cache if present; otherwise fit ALS and save."""
    if cache_exists():
        return load_candidates()
    print("[cache] no cache found; fitting ALS to populate it")
    bundle = train_als(train_df)
    candidates = als_candidate_scores(bundle, eligible, n=100)
    save_candidates(candidates)
    return candidates


def run_sweep(
    candidates: dict[int, list[tuple[int, float]]],
    train_sets: dict[int, set[int]],
    val_sets: dict[int, set[int]],
    gender_lookup: dict[int, str],
    mode: str,
) -> list[dict]:
    """Sweep λ for a single target mode."""
    print(f"\n{'#' * 60}")
    print(f"# TARGET MODE: {mode}")
    if mode == "global":
        print(f"#   target = {GLOBAL_TARGET}")
    else:
        print("#   target = each user's own train distribution")
    print(f"{'#' * 60}")

    rows: list[dict] = []
    for lam in LAMBDA_SWEEP:
        print(f"\n[sweep] mode={mode}  λ={lam:.1f}")
        t0 = time.perf_counter()
        top_k = rerank_all(
            candidates, train_sets, gender_lookup, lam=lam, mode=mode, k=DEFAULT_K
        )
        rerank_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        results = evaluate_all(
            top_k=top_k,
            train=train_sets,
            test=val_sets,
            gender=gender_lookup,
            k=DEFAULT_K,
        )
        eval_time = time.perf_counter() - t0
        print(
            f"        rerank: {rerank_time:.1f}s  eval: {eval_time:.1f}s  "
            f"Recall@10: {results['recall@k']:.4f}  "
            f"F-exp: {results['exposure_share']['female']:.3f}  "
            f"F-disp: {results['bias_disparity']['female']:+.4f}"
        )
        rows.append(_flatten_for_csv(lam, results))
    return rows


def main(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> dict[str, list[dict]]:
    print(f"[load] reading parquet from {processed_dir.resolve()}")
    t0 = time.perf_counter()
    train_df, val_df, _test_df, gender_df, _heard_df = _load_processed(processed_dir)
    print(
        f"[load] train={len(train_df):,}  val={len(val_df):,}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    eligible = val_df["user_id"].unique().tolist()
    print(f"[setup] {len(eligible):,} users to evaluate against val")

    candidates = _get_candidates(train_df, eligible)

    train_sets = _to_user_artist_sets(train_df)
    val_sets = _to_user_artist_sets(val_df)
    gender_lookup = cast(
        dict[int, str],
        dict(zip(gender_df["artist_id"], gender_df["gender"])),
    )

    # Sweep both modes
    all_rows: dict[str, list[dict]] = {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for mode in ("user", "global"):
        rows = run_sweep(candidates, train_sets, val_sets, gender_lookup, mode)
        all_rows[mode] = rows
        _print_sweep_table(rows)

        out_csv = RESULTS_DIR / f"reranker-sweep-{mode}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] wrote {len(rows)} rows to {out_csv}")

    return all_rows


if __name__ == "__main__":
    main()
