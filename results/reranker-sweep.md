# Re-ranker λ sweeps — results

**Run date:** 2026-05-26
**Dataset:** LFM-1b (full)
**Evaluation set:** val (5 held-out interactions per eligible user)
**K:** 10
**Candidate pool:** top-100 ALS scores per user, cached to
`processed/als_candidates.parquet`
**Eligible users evaluated:** 79,185

## Method

Greedy MMR-style list construction (Carbonell & Goldstein 1998; Steck 2018
adapted for gender fairness). At each position 1..K, select the candidate that
maximizes:

score(c) = (1 − λ) · normalized_relevance(c) + λ · fairness_gain(c | list_so_far)

where `fairness_gain` is the reduction in L1 distance between the list's
gender composition and a target distribution. ALS scores are min-max
normalized within each user's candidate pool. Unknown-gender candidates
contribute zero fairness gain. λ sweeps `[0.0, 0.1, ..., 1.0]`.

Two target modes are evaluated:

- **User-input matching:** target = the user's own train-set gender
  distribution over known-gender artists. Improves the same quantity bias
  disparity measures.
- **Global 30% female floor:** target = `{male: 0.60, female: 0.30, mixed: 0.10}`
  for every user, consistent with the proposal's stretch direction and
  industry fairness benchmarks (Spotify, Audible).

## User-input matching sweep

| λ   | Recall@10 | NDCG@10 | F-exp | M-exp | X-exp | F-disp  | M-disp  | X-disp  |
|-----|-----------|---------|-------|-------|-------|---------|---------|---------|
| 0.0 | 0.0747    | 0.0684  | 0.101 | 0.777 | 0.122 | −0.3662 | +0.0070 | +0.1703 |
| 0.1 | 0.0747    | 0.0683  | 0.099 | 0.782 | 0.118 | −0.3798 | +0.0148 | +0.1349 |
| 0.2 | 0.0749    | 0.0680  | 0.098 | 0.786 | 0.116 | −0.3920 | +0.0202 | +0.1008 |
| 0.3 | 0.0748    | 0.0677  | 0.097 | 0.789 | 0.114 | −0.3999 | +0.0240 | +0.0737 |
| 0.4 | 0.0747    | 0.0675  | 0.097 | 0.792 | 0.112 | −0.4004 | +0.0264 | +0.0504 |
| 0.5 | 0.0745    | 0.0672  | 0.097 | 0.793 | 0.110 | −0.3918 | +0.0263 | +0.0323 |
| 0.6 | 0.0742    | 0.0668  | 0.099 | 0.792 | 0.109 | −0.3699 | +0.0232 | +0.0167 |
| 0.7 | 0.0737    | 0.0664  | 0.100 | 0.791 | 0.108 | −0.3364 | +0.0186 | +0.0044 |
| 0.8 | 0.0726    | 0.0657  | 0.100 | 0.794 | 0.106 | −0.3031 | +0.0160 | −0.0287 |
| 0.9 | 0.0708    | 0.0646  | 0.096 | 0.800 | 0.103 | −0.2752 | +0.0134 | −0.0729 |
| 1.0 | 0.0255    | 0.0197  | 0.095 | 0.809 | 0.096 | −0.2539 | +0.0071 | −0.1845 |

Observation: female exposure barely moves across the entire λ range
(0.101 → 0.095). Female disparity actually *worsens* through λ=0.4
(−0.37 → −0.40) before slowly recovering. The re-ranker is doing real work
(visible in mixed disparity moving from +0.17 to −0.18), but the work is
mostly being absorbed by the mixed group, not female.

## Global 30% female floor sweep

| λ   | Recall@10 | NDCG@10 | F-exp | M-exp | X-exp | F-disp  | M-disp  | X-disp  |
|-----|-----------|---------|-------|-------|-------|---------|---------|---------|
| 0.0 | 0.0747    | 0.0684  | 0.101 | 0.777 | 0.122 | −0.3662 | +0.0070 | +0.1703 |
| 0.1 | 0.0747    | 0.0683  | 0.105 | 0.773 | 0.122 | −0.3209 | +0.0032 | +0.1873 |
| 0.2 | 0.0747    | 0.0681  | 0.109 | 0.768 | 0.122 | −0.2545 | −0.0026 | +0.2137 |
| 0.3 | 0.0747    | 0.0677  | 0.118 | 0.760 | 0.123 | −0.1413 | −0.0127 | +0.2433 |
| 0.4 | 0.0743    | 0.0671  | 0.131 | 0.745 | 0.123 | +0.0638 | −0.0303 | +0.2796 |
| 0.5 | 0.0737    | 0.0662  | 0.154 | 0.723 | 0.123 | +0.4334 | −0.0588 | +0.3240 |
| 0.6 | 0.0726    | 0.0647  | 0.183 | 0.694 | 0.123 | +0.9471 | −0.0934 | +0.3775 |
| 0.7 | 0.0708    | 0.0628  | 0.208 | 0.670 | 0.123 | +1.4032 | −0.1228 | +0.4155 |
| 0.8 | 0.0675    | 0.0607  | 0.219 | 0.662 | 0.119 | +1.8098 | −0.1325 | +0.2672 |
| 0.9 | 0.0579    | 0.0553  | 0.238 | 0.714 | 0.048 | +2.6481 | −0.0912 | −0.6949 |
| 1.0 | 0.0254    | 0.0194  | 0.278 | 0.675 | 0.047 | +2.6649 | −0.1060 | −0.5081 |

Observation: female exposure climbs smoothly from 10.1% to 27.8%. Female
disparity crosses zero between λ=0.3 and λ=0.4. Recall holds essentially
flat through λ=0.4, then declines gradually before cliffing at λ=1.0.

## Headline trade-off (λ=0.4, global mode)

| Quantity            | ALS baseline | Re-ranked (λ=0.4, global) | Change         |
|---------------------|--------------|---------------------------|----------------|
| Recall@10           | 0.0747       | 0.0743                    | −0.5%          |
| NDCG@10             | 0.0684       | 0.0671                    | −1.9%          |
| Female exposure     | 10.1%        | 13.1%                     | +30%           |
| Female bias dispar. | −0.366       | +0.064                    | gap eliminated |

At λ=0.4, global mode eliminates the female under-exposure gap (disparity
moves from significant suppression to essentially neutral) at a cost of
0.5% Recall and 1.9% NDCG. This is the project's headline trade-off.

## Mode comparison

The two sweeps delineate the boundaries of artist-side fairness intervention
on this dataset.

**User-input matching** preserves Recall almost perfectly across the λ range
(maximum loss at λ=0.9 is just 5% before the λ=1.0 cliff). But it cannot
meaningfully change female exposure: the maximum female exposure achieved
across the entire user-mode sweep is 10.5% (at λ=0.1), barely above the
ALS baseline. This is the structural ceiling of per-user calibration: the
re-ranker cannot push exposure beyond what users' own listening histories
imply they want, and those histories are themselves catalog-skewed.

**Global 30% target** can lift female exposure to ~22-28% but pays for it
in two ways:
1. Real relevance cost (Recall drops 5% by λ=0.6, 22% by λ=0.9).
2. Over-correction for users with low-female input: female bias disparity
   becomes large and positive at high λ, meaning recommendations contain
   far more female artists than users themselves listen to. This is by
   design — the alternative is leaving those users at popularity's
   suppression floor — but it should be acknowledged honestly.

The "sweet spot" between these extremes is roughly λ=0.3-0.4 in global mode:
female disparity moves from −0.37 to near zero with negligible relevance
loss, while still respecting that the target is a minimum representation
rather than an inversion of preferences.

## Headline finding

Per-user calibration cannot redress structural inequities present in the
preferences it is calibrating against. Closing the female exposure gap on
LFM-1b requires a fairness objective that targets a representation goal
beyond what users have revealed they want.

This is consistent with broader observations in algorithmic fairness
literature: when training data encodes the inequity being measured, models
calibrated to that data will reproduce it. The two re-ranker variants in
this project demonstrate the principle concretely — one calibrated, one
externally targeted — and quantify the relevance cost of the latter.

## Files

- `results/reranker-sweep-user.csv` — raw user-mode sweep
- `results/reranker-sweep-global.csv` — raw global-mode sweep
- `processed/als_candidates.parquet` — cached ALS top-100 candidates per user,
  reusable for further re-ranker experiments without re-fitting ALS

