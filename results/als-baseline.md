# ALS baseline — results

**Dataset:** LFM-1b (full, ≥5 listener filter)
**Evaluation set:** val (5 held-out interactions per eligible user)
**K:** 10

## Configuration

- `implicit.als.AlternatingLeastSquares`
- factors=100, regularization=0.01, iterations=15, alpha=10.0
- Play counts log-scaled (`log1p`) before passing to ALS to compress the
  dynamic range; alpha=10 then scales the compressed confidence signal.
- Same ≥5 listener catalog filter as item-KNN (541,474 of 584,930 artists).

## Runtime

- Fit: ~12 min
- Candidate scoring (top-100 per user, 79K users): ~4 min
- Metrics: 37s

## Ranking quality

| Metric    | Value  | vs. popularity | vs. KNN |
|-----------|--------|----------------|---------|
| Recall@10 | 0.0745 | 6.3× lift      | +7%     |
| NDCG@10   | 0.0682 | 7.0× lift      | +8%     |

## Fairness — exposure share

| Gender  | ALS share | KNN share | Pop share |
|---------|-----------|-----------|-----------|
| male    | 0.7773    | 0.7710    | 0.9175    |
| female  | 0.1013    | 0.0708    | 0.0223    |
| mixed   | 0.1214    | 0.1582    | 0.0603    |

## Fairness — bias disparity

| Gender  | ALS     | KNN     | Pop     |
|---------|---------|---------|---------|
| male    | +0.0078 | +0.0125 | +0.2239 |
| female  | −0.3603 | −0.4714 | −0.6241 |
| mixed   | +0.1729 | +0.5360 | −0.3743 |

## Tuning notes

- Initial run with alpha=40 (Hu/Koren default) gave Recall=0.0262 — much
  worse than KNN. Diagnosis: alpha=40 makes high-play-count interactions
  dominate confidence (~200× ratio between 5 plays and 1000 plays), and ALS
  overfits to those few high-confidence pairs.
- Reducing to alpha=1 raised Recall to 0.0601 (still below KNN).
- Final config: log1p(play_count) + alpha=10 + factors=100. Beats KNN as
  expected; ranking ordering is now popularity < KNN < ALS.

## Notes

- Personalization monotonically improves female exposure (2.2% → 7.1% → 10.1%)
  but does not close the gap — female disparity remains −0.36 under ALS.
  This is the gap the re-ranker is designed to close.
- Male disparity is essentially solved by either personalized algorithm
  (+0.01 for both). The "personalization amplifies bias" story is more
  nuanced than Shakespeare 2020's headline: amplification is the *popularity*
  pattern; personalization corrects on male, partially corrects on female.
- Matrix factorization's mixed-gender amplification (+0.17) is much more
  moderate than item-KNN's (+0.54) — the hard nearest-neighbor co-occurrence
  signal in KNN amplifies mixed-gender bands more aggressively than ALS's
  smoothed latent factors.
