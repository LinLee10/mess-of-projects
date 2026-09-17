# H003 recovered neighborhood smoothing experiment

Verified on September 17, 2026 from run 35273655571, source commit f698bbbff3059d7ba7c539de106ebb9573b4088f. This result uses only generation seed 17 and remains exploratory.

## Scope and mechanism

All 348563 Covertype training records and 116314 real development records were used. The reserved test was not accessed. The adaptive beta kernel width is fitted separately in 232 wilderness, soil and cover contexts using training leave one out nearest neighbor spacing divided by a training only beta pilot spacing, squared and clipped between one and the context size. Exact context marginals are assigned after generation. This is a project hypothesis built on established copula and smoothing methods, not a scientific priority claim.

## Recovered results

Neighbor raw: tree balanced accuracy 87.004135 percent; linear balanced accuracy 51.372871 percent; 105159 correct development predictions; unusually close fraction 53.740931 percent; four exact copies.

Neighbor rank: tree balanced accuracy 86.914134 percent; linear balanced accuracy 51.126611 percent; 105193 correct development predictions; unusually close fraction 46.543380 percent; seven exact copies.

Adaptive beta: tree balanced accuracy 85.769574 percent; linear balanced accuracy 50.760706 percent; 104270 correct development predictions; unusually close fraction 5.706859 percent; two exact copies and 348563 unique records.

The unchanged H001 seed 17 real tree reference has balanced accuracy 86.333882 percent and 104420 correct predictions. Adaptive beta is 0.564308 percentage points below that reference, with 150 fewer correct predictions. It is not superior to real data.

## Important limitations of the favorable pooled proximity result

The threshold is the real development fifth percentile nearest training distance, 0.1082745141, measured within identical contexts after training standard deviation scaling. Adaptive beta has 19892 of 348563 records below that threshold. Neighbor rank has 162233. This is a descriptive proximity comparison, not a privacy guarantee or an equivalence test.

Matching this one percentile does not establish matching the complete distance distribution. The adaptive beta median distance is 0.2099961557, while the real development median is 0.2558710745. Context specific distance distributions and coverage still require separate analysis.

Rarest class recall is 76.437848 percent for adaptive beta, 79.591837 percent for neighbor rank, and 81.632653 percent for neighbor raw. The matched real reference is 77.551020 percent. This class has 539 development observations. Do not hide this tradeoff behind pooled balanced accuracy or an overall quality ratio.

## Verification and recovery

Artifact 10518774406, named h003-seed17-35273655571-1, is 147932551 bytes. Its SHA256 a79f3361089b439664eea525d5ef19d0736b1cb0ac1d9676be37de7cdfbac432 matches GitHub metadata. All 64 internal file digests match. Its hosted audit passed independent generation reconstruction, full table quality, full nearest distance recomputation, model reloads and metric checks.

A fresh local checker with eight arithmetic and corruption controls recomputed all six student metric summaries from saved predictions, verified actual development labels, reloaded all six saved models, checked complete context frequencies and exact marginals for both calibrated variants, and recomputed the saved proximity counts. All passed. Same assistant authorship applies; this is independent computation, not independent agent or human endorsement.

The H001 input archive SHA256 e9ea0efca9c240695fafcc201d9cf7d091ab24e2ca9dd20eb8d7f9211c2a3ad6 and its 58 internal digests were also verified. The B002 input archive SHA256 b5a351d1c0b4008ff2daa37ec7deea7bfd690d2fdd97041f4e4d9e327b1470ee and its 29 internal digests were verified. Original evidence remains unchanged. Hosted artifacts expire December 16, 2026.

## Decision and successor

Retain as a verified single seed development observation, not a promoted generator. Replicate the unchanged implementation and both controls with seeds 29 and 41 before bandwidth tuning. Report per class recall, both learners, the full distance distribution and context specific proximity as well as the pooled fifth percentile statistic. At least five seeds, fresh retraining, mechanism ablations and transfer remain necessary for stronger claims.
