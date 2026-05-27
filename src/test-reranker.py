"""Hand-verified tests for reranker.rerank_user / rerank_all."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reranker import _fairness_gain, rerank_user, user_target_distribution


def test_lambda_zero_returns_als_ranking():
    """λ=0 must return ALS's top-k exactly, regardless of fairness."""
    candidates = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]
    gender = {1: "male", 2: "male", 3: "male", 4: "female"}
    target = {"male": 0.5, "female": 0.5, "mixed": 0.0}
    result = rerank_user(candidates, target, gender, lam=0.0, k=3)
    assert result == [1, 2, 3], f"λ=0 should preserve ALS ranking, got {result}"


def test_no_target_falls_back_to_relevance():
    """User with no known-gender train artists -> pure relevance ranking."""
    candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
    gender = {1: "female", 2: "male", 3: "mixed"}
    result = rerank_user(candidates, target=None, gender=gender, lam=0.8, k=3)
    assert result == [1, 2, 3], f"None target should preserve ranking, got {result}"


def test_lambda_one_picks_fairness():
    """λ=1 ignores relevance, picks toward target distribution.

    Target = 100% female. Candidates: 3 male (high relevance), 1 female (low).
    With λ=1, the single female must be chosen first.
    """
    candidates = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.1)]
    gender = {1: "male", 2: "male", 3: "male", 4: "female"}
    target = {"male": 0.0, "female": 1.0, "mixed": 0.0}
    result = rerank_user(candidates, target, gender, lam=1.0, k=1)
    assert result == [4], f"λ=1 with female target should pick female, got {result}"


def test_target_matching_balances():
    """Realistic case: target 50/50, candidates are 3 male + 3 female with
    male having higher relevance. At λ=0 you get all males. At λ=1, after
    the first selection (any candidate is fine — start from empty list),
    subsequent picks should alternate to match target."""
    candidates = [
        (1, 0.95),
        (2, 0.90),
        (3, 0.85),  # male
        (4, 0.50),
        (5, 0.45),
        (6, 0.40),  # female
    ]
    gender = {1: "male", 2: "male", 3: "male", 4: "female", 5: "female", 6: "female"}
    target = {"male": 0.5, "female": 0.5, "mixed": 0.0}

    # λ=0: pure relevance, all males
    out_relevance = rerank_user(candidates, target, gender, lam=0.0, k=4)
    assert out_relevance == [1, 2, 3, 4], f"λ=0: got {out_relevance}"

    # λ=1: pure fairness. List grows step by step:
    #   - 1st pick: from empty list, all candidates have equal fairness gain
    #     of ... well, let's compute. dist_before = sum(target) = 1.0.
    #     After adding a male: counts={m:1,f:0,x:0}, total=1, dist_after_map={m:1,f:0,x:0}
    #       L1 distance to target = |1-0.5| + |0-0.5| + 0 = 1.0. Gain = 1.0 - 1.0 = 0.
    #     After adding a female: counts={m:0,f:1,x:0}, total=1, dist_after_map={m:0,f:1,x:0}
    #       L1 distance to target = |0-0.5| + |1-0.5| + 0 = 1.0. Gain = 0.
    #     -> tie. The function picks whichever candidate comes first in iteration
    #     order; the set iteration is nondeterministic, so we can't assert which
    #     one wins. But we CAN assert that the FINAL list at k=4 is 50/50.
    out_fair = rerank_user(candidates, target, gender, lam=1.0, k=4)
    male_count = sum(1 for a in out_fair if gender[a] == "male")
    female_count = sum(1 for a in out_fair if gender[a] == "female")
    assert male_count == 2 and female_count == 2, (
        f"λ=1 with 50/50 target should yield 50/50 list at k=4; "
        f"got males={male_count} females={female_count}"
    )


def test_fairness_gain_zero_for_unknown():
    """Unknown-gender candidate contributes zero fairness gain."""
    gender = {99: "unknown"}
    target = {"male": 0.5, "female": 0.5, "mixed": 0.0}
    gain = _fairness_gain(99, {"male": 1, "female": 0, "mixed": 0}, 1, target, gender)
    assert gain == 0.0, f"unknown should give 0 gain, got {gain}"


def test_user_target_distribution():
    """Target dist over known-gender train artists, unknowns dropped."""
    gender = {1: "male", 2: "female", 3: "unknown", 4: "mixed"}
    target = user_target_distribution({1, 2, 3, 4}, gender)
    # 3 known-gender artists: 1 male, 1 female, 1 mixed
    assert target == {"male": 1 / 3, "female": 1 / 3, "mixed": 1 / 3}, f"got {target}"

    # Edge case: only unknowns
    target_none = user_target_distribution({3}, gender)
    assert target_none is None, f"all-unknown should return None, got {target_none}"

    # Edge case: empty
    target_empty = user_target_distribution(set(), gender)
    assert target_empty is None


def test_empty_candidates():
    """Empty pool -> empty list, no crash."""
    result = rerank_user([], {"male": 0.5, "female": 0.5, "mixed": 0.0}, {}, lam=0.5, k=10)
    assert result == []


def test_fewer_candidates_than_k():
    """If pool has < k items, return all of them in re-ranked order."""
    candidates = [(1, 0.9), (2, 0.5)]
    gender = {1: "male", 2: "female"}
    target = {"male": 0.5, "female": 0.5, "mixed": 0.0}
    result = rerank_user(candidates, target, gender, lam=0.5, k=10)
    assert set(result) == {1, 2}, f"got {result}"
    assert len(result) == 2


if __name__ == "__main__":
    test_lambda_zero_returns_als_ranking()
    test_no_target_falls_back_to_relevance()
    test_lambda_one_picks_fairness()
    test_target_matching_balances()
    test_fairness_gain_zero_for_unknown()
    test_user_target_distribution()
    test_empty_candidates()
    test_fewer_candidates_than_k()
    print("All re-ranker tests passed.")
