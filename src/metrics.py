"""
metrics.py
Evaluation metrics for the Fairness-Aware Music Recommendation project (DSCI 641).

Two types of metrics:
    RANKING QUALITY
    - recall_at_k(top_k, test)
    - ndcg_at_k(top_k, test)

    FAIRNESS
    - bias_disparity(top_k, train, gender)
    - exposure_share(top_k, gender)

Conventions used throughout:
  top_k    dict[user_id, list[artist_id]]    K=10 ranked recommendations per user
  test     dict[user_id, set[artist_id]]     held-out artists per user (relevance)
  train    dict[user_id, set[artist_id]]     listened artists per user (for bias-disparity input prefs)
  gender   dict[artist_id, str]              one of {"male", "female", "mixed", "unknown"}

Design decisions:
  - K defaults to 10 (proposal).
  - NDCG uses BINARY relevance (1 if in test, else 0). Play-count-weighted would
    leak popularity into the fairness story.
  - Bias disparity formula (from proposal): (rec_pref - input_pref) / input_pref,
    per user, per gender, then averaged across users with non-zero input pref
    for that gender. Skipping zeros is necessary (no division by zero) and
    correct (a user with 0% female listening has no defined "amplification").
  - Exposure share is pooled across all users' top-K lists.
  - Unknown-gender artists are FILTERED OUT of fairness metrics but kept in
    ranking metrics. We measure ranking on the list users would see; we measure
    bias only on the labelled subset where bias is defined.
  - Already-heard sanity check: warns if any top-K artist appears in that user's
    train set. The model is responsible for excluding them; we just verify.

Run `python metrics.py` to execute the self-tests.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Iterable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_K = 10
FAIRNESS_GENDERS = ("male", "female", "mixed")  # "unknown" excluded from fairness math


# ---------------------------------------------------------------------------
# Ranking quality
# ---------------------------------------------------------------------------


def recall_at_k(
    top_k: dict[int, list[int]],
    test: dict[int, set[int]],
    k: int = DEFAULT_K,
) -> float:
    """Mean Recall@K across users.

    For each user with at least one held-out test item:
        recall_u = |top_k[:k] ∩ test_u| / |test_u|
    Returned value is the mean over those users.
    """
    recalls = []
    for user, recs in top_k.items():
        truth = test.get(user, set())
        if not truth:
            continue  # user has no held-out items; not evaluable
        hits = sum(1 for a in recs[:k] if a in truth)
        recalls.append(hits / len(truth))
    return sum(recalls) / len(recalls) if recalls else 0.0


def ndcg_at_k(
    top_k: dict[int, list[int]],
    test: dict[int, set[int]],
    k: int = DEFAULT_K,
) -> float:
    """Mean NDCG@K with binary relevance.

    DCG = sum_{i=1..k} rel_i / log2(i + 1), where rel_i ∈ {0, 1}.
    IDCG = best-possible DCG given how many relevant items the user has
           (capped at k).
    NDCG = DCG / IDCG (defined for users with at least one relevant item).
    """
    ndcgs = []
    for user, recs in top_k.items():
        truth = test.get(user, set())
        if not truth:
            continue

        dcg = 0.0
        for i, artist in enumerate(recs[:k], start=1):
            if artist in truth:
                dcg += 1.0 / math.log2(i + 1)

        # IDCG: best case is all relevant items packed in the top slots.
        n_relevant_in_topk = min(len(truth), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant_in_topk + 1))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------


def _gender_distribution(
    artists: Iterable[int],
    gender: dict[int, str],
) -> dict[str, float]:
    """Fraction of `artists` that are male / female / mixed.

    Unknown-gender artists are excluded from the denominator — we only measure
    fairness on the labelled subset. If no artist in the input has a known
    gender, every group returns 0.0 (the safe value; the caller decides what
    to do with an undefined user).
    """
    counts = Counter(gender.get(a, "unknown") for a in artists)
    labelled = sum(counts[g] for g in FAIRNESS_GENDERS)
    if labelled == 0:
        return {g: 0.0 for g in FAIRNESS_GENDERS}
    return {g: counts[g] / labelled for g in FAIRNESS_GENDERS}


def bias_disparity(
    top_k: dict[int, list[int]],
    train: dict[int, set[int]],
    gender: dict[int, str],
    k: int = DEFAULT_K,
) -> dict[str, float]:
    """Mean per-user bias disparity, per gender group.

    Per the proposal:  disparity = (rec_pref - input_pref) / input_pref

    Computed per user, per gender, then averaged across users with non-zero
    input preference for that gender (users without that gender in their
    history have no defined amplification — we exclude rather than impute).

    Returns a dict like {"male": +0.12, "female": -0.30, "mixed": +0.05}.
    Positive = recommender amplified the input bias; negative = damped it.
    """
    # Sanity check: the model should have already excluded already-heard
    # artists from recommendations. Warn (don't error) if it didn't —
    # diagnostic for the modeling pipeline, not a hard failure of metrics.
    _warn_if_already_heard(top_k, train)

    per_user_disparities: dict[str, list[float]] = {g: [] for g in FAIRNESS_GENDERS}

    for user, recs in top_k.items():
        user_train = train.get(user, set())
        if not user_train:
            continue  # no input profile; no amplification to measure

        input_pref = _gender_distribution(user_train, gender)
        rec_pref = _gender_distribution(recs[:k], gender)

        for g in FAIRNESS_GENDERS:
            if input_pref[g] > 0:
                disparity = (rec_pref[g] - input_pref[g]) / input_pref[g]
                per_user_disparities[g].append(disparity)
            # else: skip — undefined for this user

    return {g: (sum(v) / len(v) if v else 0.0) for g, v in per_user_disparities.items()}


def exposure_share(
    top_k: dict[int, list[int]],
    gender: dict[int, str],
    k: int = DEFAULT_K,
) -> dict[str, float]:
    """Aggregate exposure share by gender across all top-K lists pooled.

    What fraction of recommendation slots (across the whole user base) go to
    each gender group? Slots filled by unknown-gender artists are excluded
    from the denominator — we report shares on the labelled subset.
    """
    all_slots: list[int] = []
    for recs in top_k.values():
        all_slots.extend(recs[:k])
    return _gender_distribution(all_slots, gender)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def _warn_if_already_heard(
    top_k: dict[int, list[int]],
    train: dict[int, set[int]],
) -> None:
    """Warn (once) if any recommendation appears in the user's train set.

    Discovery scenario means train artists should be filtered out before
    ranking. If the model didn't do that, this catches it.
    """
    offenders = 0
    for user, recs in top_k.items():
        user_train = train.get(user, set())
        if any(a in user_train for a in recs):
            offenders += 1
    if offenders:
        warnings.warn(
            f"{offenders} user(s) have already-heard artists in their top-K. "
            "The model should exclude train artists from recommendations "
            "before metrics are computed.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Convenience: run all four at once
# ---------------------------------------------------------------------------


def evaluate_all(
    top_k: dict[int, list[int]],
    train: dict[int, set[int]],
    test: dict[int, set[int]],
    gender: dict[int, str],
    k: int = DEFAULT_K,
) -> dict:
    """Run every metric and return a single results dict.

    Convenient for the experiment loop: one call per configuration.
    """
    return {
        "k": k,
        "recall@k": recall_at_k(top_k, test, k=k),
        "ndcg@k": ndcg_at_k(top_k, test, k=k),
        "bias_disparity": bias_disparity(top_k, train, gender, k=k),
        "exposure_share": exposure_share(top_k, gender, k=k),
    }


# ---------------------------------------------------------------------------
# Self-tests (hand-verified worked examples)
# ---------------------------------------------------------------------------


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _run_self_tests() -> None:
    """Tiny worked example where every metric can be checked by hand.

    Setup:
      - 3 users, K=3, 4 artists with known genders.
      - User 1: recs hit 1 of 2 truth items at position 1.
      - User 2: recs hit 1 of 1 truth item at position 2.
      - User 3: no truth items — skipped by ranking metrics, still counts
        for exposure.

    Artists:    A1=male, A2=female, A3=male, A4=female
    """
    gender = {1: "male", 2: "female", 3: "male", 4: "female"}

    top_k = {
        1: [1, 2, 3],  # truth = {1, 4}: hit A1 at pos 1, miss A4
        2: [3, 4, 1],  # truth = {4}:    hit A4 at pos 2
        3: [2, 4, 1],  # truth = set():  not evaluable for ranking
    }
    test = {1: {1, 4}, 2: {4}, 3: set()}
    train = {
        1: {3},  # 1/1 male input -> 100% male input pref
        2: {4},  # 1/1 female input -> 100% female input pref
        3: {1, 2, 3, 4},  # balanced: 50% male, 50% female
    }

    # --- Recall@3 ---
    # u1: 1/2 = 0.5
    # u2: 1/1 = 1.0
    # u3: skipped
    # mean = 0.75
    r = recall_at_k(top_k, test, k=3)
    assert _approx(r, 0.75), f"recall: expected 0.75, got {r}"

    # --- NDCG@3 (binary) ---
    # u1: DCG = 1/log2(2) = 1.0;  IDCG (2 relevant in top-3) = 1/log2(2) + 1/log2(3)
    # u2: DCG = 1/log2(3);         IDCG (1 relevant in top-3) = 1/log2(2) = 1.0
    # u3: skipped
    idcg_u1 = 1.0 + 1.0 / math.log2(3)
    expected_ndcg = ((1.0 / idcg_u1) + (1.0 / math.log2(3))) / 2
    n = ndcg_at_k(top_k, test, k=3)
    assert _approx(n, expected_ndcg), f"ndcg: expected {expected_ndcg}, got {n}"

    # --- Exposure share ---
    # All 9 slots, gender breakdown:
    #   A1 male x3, A2 female x2, A3 male x2, A4 female x2
    #   -> male=5/9, female=4/9
    ex = exposure_share(top_k, gender, k=3)
    assert _approx(ex["male"], 5 / 9), f"exposure male: {ex['male']}"
    assert _approx(ex["female"], 4 / 9), f"exposure female: {ex['female']}"
    assert ex["mixed"] == 0.0

    # --- Bias disparity ---
    # u1: input male=1.0, female=0.0;  rec male=2/3, female=1/3
    #     -> male disparity = (2/3 - 1)/1 = -1/3; female undefined (input=0, skip)
    # u2: input male=0.0, female=1.0;  rec male=2/3, female=1/3
    #     -> female disparity = (1/3 - 1)/1 = -2/3; male undefined
    # u3: input male=0.5, female=0.5;  rec (A2,A4,A1) -> male=1/3, female=2/3
    #     -> male disparity = (1/3 - 0.5)/0.5 = -1/3
    #     -> female disparity = (2/3 - 0.5)/0.5 = +1/3
    # mean male = avg(-1/3, -1/3) = -1/3
    # mean female = avg(-2/3, +1/3) = -1/6
    bd = bias_disparity(top_k, train, gender, k=3)
    assert _approx(bd["male"], -1 / 3), f"bias male: {bd['male']}"
    assert _approx(bd["female"], -1 / 6), f"bias female: {bd['female']}"

    # --- Already-heard warning ---
    # u1's recs include A3, which is in u1's train. Should fire a warning.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bias_disparity(top_k, train, gender, k=3)
        msgs = [str(x.message) for x in w]
        assert any("already-heard" in m for m in msgs), "expected already-heard warning to fire"

    # --- evaluate_all sanity ---
    out = evaluate_all(top_k, train, test, gender, k=3)
    assert set(out) == {"k", "recall@k", "ndcg@k", "bias_disparity", "exposure_share"}
    assert out["k"] == 3

    print("All self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
