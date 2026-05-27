"""
reranker.py — Fairness-aware re-ranker for the Fairness-Aware Music
Recommendation project (DSCI 641).

This is the project's actual contribution. Takes ALS's top-N scored candidates
per user and re-ranks them list-level to balance relevance against gender
fairness, controlled by a single tunable λ ∈ [0, 1].

Method (greedy MMR-style, after Carbonell & Goldstein 1998 / Steck 2018):
  At each position 1..K, select the candidate that maximizes:

    score(c) = (1 - λ) * relevance(c) + λ * fairness_gain(c | list_so_far)

  - relevance(c): ALS score for candidate c, min-max normalized within the
    user's candidate pool to [0, 1].
  - fairness_gain(c | list): how much adding c moves the list's gender
    composition TOWARD the user's input (train) gender distribution.
    Formally: -|target - dist(list ∪ {c})|_1 + |target - dist(list)|_1.
    Positive means c improves alignment; negative means it worsens.

Design decisions:
  - Target = user's own training-set gender distribution among known-gender
    artists. The re-ranker improves the same quantity bias_disparity measures.
  - Unknown-gender candidates are treated as gender-neutral: they contribute
    to no group's tally and incur no fairness penalty/bonus. This avoids
    forcing the re-ranker to avoid or prefer them based on missing data.
  - Users with no known-gender artists in train: fall back to pure relevance
    (no fairness signal available).
  - λ = 0 should produce ALS's ranking exactly (sanity check at runtime).
  - λ = 1 ignores relevance entirely; ranks purely by fairness gain.

Run `python src/reranker.py` to fit ALS once, sweep λ across 11 values,
print the trade-off table. Output is written to results/reranker-sweep.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import cast

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from als import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    als_candidate_scores,
    train_als,
)
from metrics import DEFAULT_K, FAIRNESS_GENDERS, evaluate_all  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAMBDA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CANDIDATE_POOL = 100
RESULTS_CSV = Path("results") / "reranker-sweep.csv"


# ---------------------------------------------------------------------------
# Per-user target distribution
# ---------------------------------------------------------------------------


def user_target_distribution(
    train_artists: set[int],
    gender: dict[int, str],
) -> dict[str, float] | None:
    """User's input gender distribution over KNOWN-gender artists.

    Returns None if the user has no known-gender artists in train — in that
    case the re-ranker falls back to pure relevance for that user.
    """
    counts = {g: 0 for g in FAIRNESS_GENDERS}
    for a in train_artists:
        g = gender.get(a, "unknown")
        if g in counts:
            counts[g] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    return {g: c / total for g, c in counts.items()}


def _list_distribution(
    artist_list: list[int],
    gender: dict[int, str],
) -> tuple[dict[str, int], int]:
    """Counts of each known gender in the current partial list + labeled total.

    Returns ({gender: count}, total_labeled). Unknown-gender items aren't
    counted in either side — they're gender-neutral.
    """
    counts = {g: 0 for g in FAIRNESS_GENDERS}
    for a in artist_list:
        g = gender.get(a, "unknown")
        if g in counts:
            counts[g] += 1
    return counts, sum(counts.values())


def _l1_distance(dist: dict[str, float], target: dict[str, float]) -> float:
    return sum(abs(dist[g] - target[g]) for g in target)


# ---------------------------------------------------------------------------
# Greedy re-ranking
# ---------------------------------------------------------------------------


def _normalize_scores(candidates: list[tuple[int, float]]) -> dict[int, float]:
    """Min-max normalize ALS scores to [0, 1] within the candidate pool.

    Required because ALS scores aren't on any standard scale, and we're
    about to add them to a [-1, 1] fairness term.
    """
    if not candidates:
        return {}
    scores = [s for _, s in candidates]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {a: 1.0 for a, _ in candidates}
    return {a: (s - lo) / (hi - lo) for a, s in candidates}


def _fairness_gain(
    candidate: int,
    current_counts: dict[str, int],
    current_total: int,
    target: dict[str, float],
    gender: dict[int, str],
) -> float:
    """How much does adding `candidate` improve alignment with target?

    Returns the (negative) L1 distance change. Positive value means adding
    this candidate moves the list closer to target; negative means away.

    Unknown-gender candidate: zero gain (gender-neutral).
    """
    g = gender.get(candidate, "unknown")
    if g not in current_counts:
        return 0.0

    # Distance before adding (current list).
    if current_total == 0:
        # Empty list dist is degenerate; define target distance as L1 of target itself.
        dist_before = sum(target.values())  # = 1.0 when target is a valid distribution
    else:
        dist_before_map = {g_: current_counts[g_] / current_total for g_ in FAIRNESS_GENDERS}
        dist_before = _l1_distance(dist_before_map, target)

    # Distance after adding.
    new_total = current_total + 1
    new_counts = dict(current_counts)
    new_counts[g] += 1
    dist_after_map = {g_: new_counts[g_] / new_total for g_ in FAIRNESS_GENDERS}
    dist_after = _l1_distance(dist_after_map, target)

    return dist_before - dist_after  # positive = improvement


def rerank_user(
    candidates: list[tuple[int, float]],
    target: dict[str, float] | None,
    gender: dict[int, str],
    lam: float,
    k: int,
) -> list[int]:
    """Greedy MMR-style re-ranking of one user's candidates.

    candidates: list of (artist_id, als_score), sorted desc by als_score.
    target:     user's input gender distribution; None -> pure relevance.
    """
    # Edge cases short-circuit to the original ranking.
    if not candidates or k == 0:
        return [a for a, _ in candidates[:k]]
    if lam == 0.0 or target is None:
        return [a for a, _ in candidates[:k]]

    norm = _normalize_scores(candidates)
    remaining = {a for a, _ in candidates}
    selected: list[int] = []
    counts = {g: 0 for g in FAIRNESS_GENDERS}
    total = 0

    while len(selected) < k and remaining:
        best_artist = None
        best_score = -float("inf")

        for a in remaining:
            rel = norm[a]
            fair = _fairness_gain(a, counts, total, target, gender)
            score = (1 - lam) * rel + lam * fair
            if score > best_score:
                best_score = score
                best_artist = a

        # best_artist is guaranteed non-None when remaining is non-empty.
        assert best_artist is not None
        selected.append(best_artist)
        remaining.discard(best_artist)
        g = gender.get(best_artist, "unknown")
        if g in counts:
            counts[g] += 1
            total += 1

    return selected


def rerank_all(
    candidates: dict[int, list[tuple[int, float]]],
    train_sets: dict[int, set[int]],
    gender: dict[int, str],
    lam: float,
    k: int = DEFAULT_K,
) -> dict[int, list[int]]:
    """Re-rank every user's candidate pool at a single λ."""
    out: dict[int, list[int]] = {}
    for user, cands in candidates.items():
        target = user_target_distribution(train_sets.get(user, set()), gender)
        out[user] = rerank_user(cands, target, gender, lam, k)
    return out


# ---------------------------------------------------------------------------
# Sweep driver
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


def _flatten_for_csv(lam: float, r: dict) -> dict:
    """Flatten the metrics dict for one row of the sweep CSV."""
    row = {
        "lambda": lam,
        "k": r["k"],
        "recall@k": r["recall@k"],
        "ndcg@k": r["ndcg@k"],
    }
    for g, v in r["exposure_share"].items():
        row[f"exposure_{g}"] = v
    for g, v in r["bias_disparity"].items():
        row[f"disparity_{g}"] = v
    return row


def _print_sweep_table(rows: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("RE-RANKER λ SWEEP @ K=10 (eval: val)")
    print("=" * 80)
    header = (
        f"{'λ':>4}  {'Recall':>7}  {'NDCG':>7}  "
        f"{'F-exp':>6}  {'M-exp':>6}  {'X-exp':>6}  "
        f"{'F-disp':>7}  {'M-disp':>7}  {'X-disp':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['lambda']:>4.1f}  "
            f"{row['recall@k']:>7.4f}  {row['ndcg@k']:>7.4f}  "
            f"{row['exposure_female']:>6.3f}  {row['exposure_male']:>6.3f}  {row['exposure_mixed']:>6.3f}  "
            f"{row['disparity_female']:>+7.4f}  {row['disparity_male']:>+7.4f}  {row['disparity_mixed']:>+7.4f}"
        )
    print("=" * 80)


def main(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> list[dict]:
    print(f"[load] reading parquet from {processed_dir.resolve()}")
    t0 = time.perf_counter()
    train_df, val_df, _test_df, gender_df, _heard_df = _load_processed(processed_dir)
    print(f"[load] train={len(train_df):,}  val={len(val_df):,}  ({time.perf_counter() - t0:.1f}s)")

    eligible = val_df["user_id"].unique().tolist()
    print(f"[setup] {len(eligible):,} users to evaluate against val")

    # Train ALS once, score candidates once. Then sweep λ over the cached candidates.
    bundle = train_als(train_df)
    candidates = als_candidate_scores(bundle, eligible, n=CANDIDATE_POOL)

    # Convert dataframes to dict form once.
    train_sets = _to_user_artist_sets(train_df)
    val_sets = _to_user_artist_sets(val_df)
    gender_lookup = cast(
        dict[int, str],
        dict(zip(gender_df["artist_id"], gender_df["gender"])),
    )

    rows: list[dict] = []
    for lam in LAMBDA_SWEEP:
        print(f"\n[sweep] λ = {lam:.1f}")
        t0 = time.perf_counter()
        top_k = rerank_all(candidates, train_sets, gender_lookup, lam=lam, k=DEFAULT_K)
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
            f"F-disp: {results['bias_disparity']['female']:+.4f}"
        )
        rows.append(_flatten_for_csv(lam, results))

    _print_sweep_table(rows)

    # Write CSV. ensure results/ exists.
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[done] wrote {len(rows)} rows to {RESULTS_CSV}")

    return rows


if __name__ == "__main__":
    main()
