"""
prepare_data.py — Reproducible data pipeline for the Fairness-Aware Music
Recommendation project (DSCI 641).

Turns raw Last.fm + MusicBrainz files into clean, model-ready artifacts:
  - train / val / test interaction splits (per-user random holdout)
  - artist gender lookup table (classify_gender logic from EDA)
  - an "already-heard" mask so discovery eval excludes familiar artists
  - sanity-check printouts that should match the EDA numbers

Derived from EDA_recsys.ipynb. The gender join is a clean artist_id merge —
the gender_breakdown column is already in the raw artist file, so there is
NO fuzzy MusicBrainz matching to do here.

------------------------------------------------------------------------------
DATASET DECISION  (decided: LFM-1b — proposal to be amended)
------------------------------------------------------------------------------
The original proposal cited Last.fm 360K, but the EDA's entire fairness
analysis (gender coverage, per-user female share, top-N splits, artist-name
spot checks) was conducted on LFM-1b. We default to LFM-1b so the EDA serves
as pipeline validation rather than re-running EDA under time pressure. The
proposal will be amended with a one-paragraph justification; Shakespeare 2020
methodology (classify_gender rule, bias-disparity metric) is preserved either
way. To switch back to 360K, change DATASET below — nothing else changes.

Tradeoff accepted: LFM-1b is ~48M interactions vs 360K's ~6.5M, so ALS will be
slower / more memory-hungry. That cost is known and tunable (downsample users,
fewer factors/iterations); re-validating gender coverage on 360K is an unknown
cost, which is the bigger schedule risk.
------------------------------------------------------------------------------

Usage:
    python prepare_data.py --data-dir ./data --out-dir ./processed
    python prepare_data.py --data-dir ./data --out-dir ./processed --dataset lfm1b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default dataset. "lfm1b" matches what the EDA fairness analysis actually ran
# on (decided — see DATASET DECISION above). "lastfm360k" matches the original
# proposal text before amendment.
DATASET = "lfm1b"

# Per-user holdout sizes (from proposal: "reserving 5 items per user each for
# validation and test").
N_VAL = 5
N_TEST = 5

# A user needs enough interactions to give up 10 to eval and still train on
# something. 5 val + 5 test + at least 5 train.
MIN_INTERACTIONS_PER_USER = N_VAL + N_TEST + 5

RANDOM_SEED = 641  # reproducibility

DATASET_FILES = {
    "lastfm360k": {
        "listening": "LastFM360k-Le75.txt",
        "artists": "LastFM360k-MB-artists.txt",
        # 360K artist file has a 4th mb_id column the LFM-1b file lacks.
        "artist_cols": ["artist_id", "artist_name", "gender_breakdown", "mb_id"],
        "listening_kwargs": {},  # header row present; we rename below
    },
    "lfm1b": {
        "listening": "LFM-1b-Le75.csv",
        "artists": "LFM1b-MB-artists.txt",
        "artist_cols": ["artist_id", "artist_name", "gender_breakdown"],
        "listening_kwargs": {"header": None},
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw(data_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load listening + artist tables for the chosen dataset.

    Mirrors the load logic in the EDA notebook's first cell.
    """
    cfg = DATASET_FILES[dataset]

    listening_path = data_dir / cfg["listening"]
    artists_path = data_dir / cfg["artists"]

    for p in (listening_path, artists_path):
        if not p.exists():
            sys.exit(
                f"ERROR: expected file not found: {p}\n"
                f"Pass the right --data-dir (it should contain {cfg['listening']} "
                f"and {cfg['artists']})."
            )

    listening = pd.read_csv(listening_path, **cfg["listening_kwargs"])
    listening.columns = ["user_id", "artist_id", "play_count"]

    artists = pd.read_csv(
        artists_path,
        sep="\t",
        header=None,
        names=cfg["artist_cols"],
    )

    print(f"[load] listening: {listening.shape}  artists: {artists.shape}")
    return listening, artists


# Gender classification  (logic from EDA_recsys.ipynb)
def classify_gender(row) -> str:
    """female-only -> female, male-only -> male, both -> mixed, neither -> unknown.

    This is the exact rule from the EDA notebook. Do not "improve" it without
    the team agreeing — consistency with the EDA numbers is the point.
    """
    if row["female"] > 0 and row["male"] == 0:
        return "female"
    elif row["male"] > 0 and row["female"] == 0:
        return "male"
    elif row["female"] > 0 and row["male"] > 0:
        return "mixed"
    else:
        return "unknown"


def build_gender_table(artists: pd.DataFrame) -> pd.DataFrame:
    """Split gender_breakdown 'u/m/f/o/n' and apply classify_gender.

    Returns a tidy [artist_id, gender] lookup.
    """
    artists = artists.copy()
    split = artists["gender_breakdown"].str.split("/", expand=True).astype(int)
    split.columns = ["unknown", "male", "female", "other", "na"]
    artists[["unknown", "male", "female", "other", "na"]] = split
    artists["gender"] = artists.apply(classify_gender, axis=1)

    dist = artists["gender"].value_counts(normalize=True)
    print("[gender] artist-level distribution:")
    for g, frac in dist.items():
        print(f"         {g:<8} {frac:.4f}")

    return artists[["artist_id", "gender"]]


# Split  (per-user random holdout)
def per_user_holdout(
    listening: pd.DataFrame,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    min_interactions: int = MIN_INTERACTIONS_PER_USER,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve n_val + n_test random items per user; rest is train.

    The Last.fm 360K data has aggregated playcounts with no timestamps, so a
    temporal split is impossible at the interaction level (proposal, Splitting
    Strategy). A single random holdout is sufficient given ~360K users.

    Users with fewer than `min_interactions` records are dropped from eval
    (they can't give up 10 items and still train on anything meaningful).
    Those users' interactions still go into train so the model sees them.
    """
    rng = np.random.default_rng(seed)

    counts = listening.groupby("user_id").size()
    eligible = set(counts[counts >= min_interactions].index)
    n_dropped = len(counts) - len(eligible)
    print(
        f"[split] {len(eligible)} users eligible for eval; "
        f"{n_dropped} users below {min_interactions} interactions "
        f"(kept in train only)"
    )

    # Shuffle once, then take the first n_test / next n_val per eligible user.
    shuffled = listening.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled["_rank"] = shuffled.groupby("user_id").cumcount()

    is_eligible = shuffled["user_id"].isin(eligible)
    test_mask = is_eligible & (shuffled["_rank"] < n_test)
    val_mask = is_eligible & (shuffled["_rank"] >= n_test) & (shuffled["_rank"] < n_test + n_val)

    test = shuffled[test_mask].drop(columns="_rank")
    val = shuffled[val_mask].drop(columns="_rank")
    train = shuffled[~(test_mask | val_mask)].drop(columns="_rank")

    print(f"[split] train: {len(train)}  val: {len(val)}  test: {len(test)}")
    return train, val, test


# Already-heard mask  (discovery = unfamiliar artists only)
def build_already_heard(train: pd.DataFrame) -> pd.DataFrame:
    """Per-user set of artists seen in TRAIN, as a long [user_id, artist_id] table.

    The discovery scenario means we exclude already-heard artists before
    evaluation (proposal, User Interaction Model). Evaluation code should
    remove these (user, artist) pairs from each user's candidate list.
    """
    heard = train[["user_id", "artist_id"]].drop_duplicates()
    print(f"[heard] {len(heard)} (user, artist) pairs to exclude at eval time")
    return heard


# Sanity checks which should line up with the EDA notebook
def sanity_checks(listening: pd.DataFrame, gender: pd.DataFrame) -> None:
    """Print the headline EDA numbers so we can confirm nothing drifted."""
    merged = listening.merge(gender, on="artist_id", how="left")
    merged["gender"] = merged["gender"].fillna("unknown")

    print("\n=== SANITY CHECKS (compare against EDA_recsys.ipynb) ===")

    plays = merged.groupby("gender")["play_count"].sum()
    print("Share of plays by gender (EDA: f .077 / m .584 / mix .098 / unk .242):")
    for g, frac in (plays / plays.sum()).items():
        print(f"  {g:<8} {frac:.4f}")

    top = (
        merged.groupby(["artist_id", "gender"])["play_count"]
        .sum()
        .reset_index()
        .sort_values("play_count", ascending=False)
    )
    for n in (100, 500, 1000):
        d = top.head(n)["gender"].value_counts(normalize=True).to_dict()
        print(
            f"Top {n}: male={d.get('male', 0):.3f} "
            f"female={d.get('female', 0):.3f} "
            f"mixed={d.get('mixed', 0):.3f} "
            f"unknown={d.get('unknown', 0):.3f}"
        )
    print("(Proposal cites top-1000 ~ 67/7/15/12 male/female/mixed/unknown)")
    print("=== end sanity checks ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed"))
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_FILES),
        default=DATASET,
        help="lfm1b (EDA-validated, default) or lastfm360k (original proposal)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] dataset={args.dataset} seed={RANDOM_SEED}")

    listening, artists = load_raw(args.data_dir, args.dataset)
    gender = build_gender_table(artists)
    train, val, test = per_user_holdout(listening)
    heard = build_already_heard(train)

    sanity_checks(listening, gender)

    # Parquet keeps dtypes and is fast for the model scripts to read back.
    out = args.out_dir
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    gender.to_parquet(out / "artist_gender.parquet", index=False)
    heard.to_parquet(out / "already_heard.parquet", index=False)

    print(f"[done] wrote 5 parquet files to {out.resolve()}")
    print("       train / val / test / artist_gender / already_heard")


if __name__ == "__main__":
    main()
