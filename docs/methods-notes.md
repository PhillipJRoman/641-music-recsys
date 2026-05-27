# Methods Notes

We will be tracking any decisions made in the methods or metrics for this project.

**Dataset**: We use the LFM-1b dataset (Schedl, 2016), a Last.fm interaction corpus of approximately 79,000 users and 48 million user-artist play-count records, joined to gender metadata derived from MusicBrainz following the methodology of Shakespeare et al. (2020). Among publicly available music-listening datasets, LFM-1b is well-suited to our research question because it pairs large-scale implicit feedback with the artist-level gender metadata required to measure exposure bias. Artist gender is classified from a five-way breakdown (unknown/male/female/other/na) into four exclusive categories: female-only artists, male-only artists, mixed-gender groups (containing both male and female members), and unknown (no gender signal). Although gender metadata is missing for 86% of artists in the catalog, those artists account for only 24% of total play volume, leaving sufficient coverage to measure fairness on the recommendations that drive user experience. We train and evaluate on the full interaction set and compute fairness metrics over the known-gender subset of recommendations, as proposed.

## Baselines

### Popularity Baseline Recommender
The popularity baseline ranks artists by total play count across all training
interactions and serves the top K to every user, with each user's already-heard
artists filtered out. It is the no-personalization floor: any model that
cannot beat it on ranking quality is not doing useful work. It also serves
as the fairness reference point for the re-ranker, since popularity bias is
a known channel through which gender bias propagates, the popularity
baseline's bias-disparity numbers are the "do nothing" benchmark the
re-ranker is designed to improve on.

We use plain total play count as the popularity signal rather than unique
listener count or log-transformed plays. This is the simplest defensible
choice and is what the proposal implies. The per-user already-heard filter
is the only personalization signal in this baseline — every user sees the
same global ranking, differing only by which artists are removed.
