# Popularity baseline — results
**Dataset:** LFM-1b (full)
**Evaluation set:** val (5 held-out interactions per eligible user)
**K:** 10
**Candidate pool:** 500
**Eligible users evaluated:** 79,185

## Ranking quality

| Metric    | Value  |
|-----------|--------|
| Recall@10 | 0.0119 |
| NDCG@10   | 0.0098 |

## Fairness — exposure share

| Gender  | Share of recommendation slots |
|---------|-------------------------------|
| male    | 0.9175                        |
| female  | 0.0223                        |
| mixed   | 0.0603                        |

## Fairness — bias disparity

| Gender  | Disparity | Interpretation                        |
|---------|-----------|---------------------------------------|
| male    | +0.2239   | Amplified (recs over-serve male)      |
| female  | −0.6241   | Suppressed (recs under-serve female)  |
| mixed   | −0.3743   | Suppressed                            |

## Notes

- Numbers are from `python src/popularity.py` against `processed/val.parquet`.
- Floor for ranking quality: ALS should beat both metrics by 10× or more.
- Headline fairness finding: users' listening histories contain substantially
  more female and mixed-gender artists than the popularity recommender ever
  surfaces back to them. There is real room for the re-ranker to push female
  disparity from −0.62 toward zero.
- The exposure share / bias disparity disconnect: 92% male exposure with only
  +0.22 male disparity means users' input profiles are themselves heavily
  male-skewed; 92% is ~22% more than the average user's ~75% male input share.
