# ALS baseline — results

**Run date:** 2026-05-26
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
| NDCG@10   | 0.0682 | 7.0× lift      | +8%
