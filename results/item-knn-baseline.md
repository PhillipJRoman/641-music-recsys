# Item-KNN baseline — results

**Run date:** 2026-05-26
**Dataset:** LFM-1b (full, ≥5 listener filter)
**Evaluation set:** val (5 held-out interactions per eligible user)
**K:** 10
**K_neighbors:** 50
**Min listeners:** 5
**Eligible users evaluated:** 79,185

## Configuration

- `implicit.nearest_neighbours.CosineRecommender` on binarized user-item matrix
- Catalog filter: kept 541,474 of 584,930 artists (≥5 listeners)
- 47,379,478 of 47,531,330 train interactions retained (99.7%)

## Runtime

- Fit: 374.7s (~6 min)
- Recommend (79K users): 35.7s
- Metrics: 43.9s

## Ranking quality

| Metric    | Value  | vs. popularity |
|-----------|--------|----------------|
| Recall@10 | 0.0695 | 5.8× lift      |
| NDCG@10   | 0.0634 | 6.5× lift      |

## Fairness — exposure share

| Gender  | KNN share | Popularity share |
|---------|-----------|------------------|
| male    | 0.7710    | 0.9175           |
| female  | 0.0708    | 0.0223           |
| mixed   | 0.1582    | 0.0603           |

## Fairness — bias disparity

| Gender  | KNN     | Popularity | Interpretation              |
|---------|---------|------------|-----------------------------|
| male    | +0.0125 | +0.2239    | Near-perfectly calibrated   |
| female  | −0.4714 | −0.6241    | Still suppressed, less so   |
| mixed   | +0.5360 | −0.3743    | Now over-amplified          |

## Notes

- Personalization mostly absorbs the male amplification that popularity produces:
  male disparity drops from +0.22 to ~0.
- Female artists remain under-served by ~47% relative to users' input profiles
  — this is the gap the re-ranker is designed to close.
- **Surprising:** KNN amplifies mixed-gender groups (+0.54), reversing popularity's
  −0.37 suppression. Hypothesis: mixed-gender bands (Fleetwood Mac, Paramore,
  The xx) have strong co-occurrence patterns in user histories, and KNN
  aggressively surfaces co-occurring artists, many of which are also mixed.
- Reframes the project's headline: not "personalization amplifies bias" uniformly
  but "personalization corrects male over-exposure while still under-serving
  female artists and over-serving mixed groups."
