"""Hand-verified tests for the global-target variant in reranker_v2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reranker_v2 import get_target, rerank_all, GLOBAL_TARGET


def test_global_target_ignores_user_history():
    """Global mode returns the same target regardless of user's train set."""
    gender = {1: "male", 2: "female"}
    # User 1 has only male history
    t1 = get_target("global", {1}, gender)
    # User 2 has only female history
    t2 = get_target("global", {2}, gender)
    # User 3 has empty history (would be None under "user" mode)
    t3 = get_target("global", set(), gender)
    assert t1 == GLOBAL_TARGET, f"user 1: got {t1}"
    assert t2 == GLOBAL_TARGET, f"user 2: got {t2}"
    assert t3 == GLOBAL_TARGET, f"user 3: got {t3}"


def test_user_target_still_works():
    """User mode preserves the original behavior."""
    gender = {1: "male", 2: "female", 3: "mixed"}
    t = get_target("user", {1, 2, 3}, gender)
    assert t == {"male": 1/3, "female": 1/3, "mixed": 1/3}, f"got {t}"
    # Empty history -> None
    assert get_target("user", set(), gender) is None


def test_global_target_pushes_toward_30pct_female():
    """At high λ, global mode should produce ~30% female across the top-K.

    Setup: 1 user, 10 candidates (7 male, 3 female). Pure relevance picks all
    males first (males have higher scores). With λ=1 and global target,
    the top-K should reflect GLOBAL_TARGET roughly.
    """
    candidates = {
        100: [
            (1, 0.99), (2, 0.95), (3, 0.92), (4, 0.88), (5, 0.85),
            (6, 0.82), (7, 0.78),  # 7 male
            (8, 0.60), (9, 0.55), (10, 0.50),  # 3 female
        ]
    }
    gender = {
        1: "male", 2: "male", 3: "male", 4: "male",
        5: "male", 6: "male", 7: "male",
        8: "female", 9: "female", 10: "female",
    }
    train_sets = {100: {1}}  # user listens to one male artist

    # λ=0: pure relevance -> top 5 are all male
    top_k_rel = rerank_all(candidates, train_sets, gender, lam=0.0, mode="global", k=5)
    males = sum(1 for a in top_k_rel[100] if gender[a] == "male")
    females = sum(1 for a in top_k_rel[100] if gender[a] == "female")
    assert males == 5 and females == 0, f"λ=0: got {males}M/{females}F"

    # λ=1: pure global target. 30% female of 5 = 1.5, so list should have
    # at least 1 female. Greedy will pick toward the 60/30/10 target.
    top_k_fair = rerank_all(candidates, train_sets, gender, lam=1.0, mode="global", k=5)
    females_fair = sum(1 for a in top_k_fair[100] if gender[a] == "female")
    assert females_fair >= 1, (
        f"λ=1 with global 30% female target should include at least one female; "
        f"got {females_fair} females in top-5"
    )


def test_global_target_used_even_for_no_history_user():
    """Global mode rescues users with no known-gender history (where user mode
    would fall back to pure relevance)."""
    candidates = {200: [(1, 0.9), (2, 0.5)]}
    gender = {1: "male", 2: "female"}
    train_sets = {200: set()}  # empty history

    # Under user mode at λ=1, target is None -> falls back to relevance
    # (already tested in test_reranker.py). Under global mode, target is
    # GLOBAL_TARGET, so re-ranker should consider fairness even though
    # the user has no history.

    # With λ=1 (pure fairness) and an empty list, the greedy step picks
    # the candidate with highest fairness gain. From empty:
    #   male candidate would move list to {m:1,f:0,x:0}; distance to target
    #     (0.6, 0.3, 0.1) = |1-0.6| + |0-0.3| + |0-0.1| = 0.8
    #     gain = 1.0 (empty dist) - 0.8 = 0.2
    #   female candidate would move list to {m:0,f:1,x:0}; distance to target
    #     = |0-0.6| + |1-0.3| + |0-0.1| = 1.4
    #     gain = 1.0 - 1.4 = -0.4
    # So male candidate has higher fairness gain from empty -> male picked first.
    top_k = rerank_all(candidates, train_sets, gender, lam=1.0, mode="global", k=2)
    assert top_k[200][0] == 1, (
        f"empty-history user, λ=1, global: should pick male first "
        f"(closer to 60/30/10 from empty); got {top_k[200]}"
    )


if __name__ == "__main__":
    test_global_target_ignores_user_history()
    test_user_target_still_works()
    test_global_target_pushes_toward_30pct_female()
    test_global_target_used_even_for_no_history_user()
    print("All v2 tests passed.")
