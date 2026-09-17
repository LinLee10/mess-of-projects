# H002 recovered dependence comparison

Verified on September 17, 2026 from retained run 35271388310, source commit f40b8fc0fbca70eae5f2aaed21adb033566725a6. All three completed generation seeds were recovered. No new generation or training was performed to recover these results.

## Scope

Every method used all 348563 training rows and was evaluated on all 116314 real development rows. Every numerical marginal within each wilderness, soil and cover context is exactly identical to the training marginal. The reserved test was not accessed. The seven class recalls have equal weight in balanced accuracy.

## Results

Values below are percentages. The uncertainty column is sample standard deviation across generation seeds 17, 29 and 41, not uncertainty over independent datasets.

Gaussian rank: tree balanced accuracy 76.360340, standard deviation 0.385148 percentage points; linear balanced accuracy 50.669692.

Beta rank: tree balanced accuracy 87.176385, standard deviation 0.195918 percentage points; linear balanced accuracy 50.972896.

Wide beta rank: tree balanced accuracy 85.255319, standard deviation 0.413882 percentage points; linear balanced accuracy 50.739256.

Independent rank: tree balanced accuracy 73.337892, standard deviation 0.146594 percentage points; linear balanced accuracy 52.533611.

The beta versus independent rank difference is 13.838492 percentage points for the tree. The linear learner instead prefers independent ranks. This is a controlled dependence intervention for this dataset, split and learner recipe, not a universal ranking or a scientific priority claim.

## Recovery receipts

Seed 17: artifact 10518567231, 236566449 bytes, SHA256 82eb52121c1c9781859903544413dd4632e8bc93bceca0f96d99205d3c287816.

Seed 29: artifact 10519641731, 236567716 bytes, SHA256 cb3acad1948c56131ece1cde32afc74f6f991b64449554e546a67ace9c7e4084.

Seed 41: artifact 10519717588, 236562240 bytes, SHA256 5b911c8ce1e332c47fc93357257629d7d94cf08a633967fba71f87683177a9e9.

All three archive SHA256 values matched the GitHub artifact metadata. All 189 internal manifest digests matched. Hosted generation reconstruction, complete table quality and inference audit reports have status passed.

A fresh local checker, supported by eight corruption and arithmetic controls, independently recomputed all 24 student confusion matrices and metric summaries from the saved predictions, verified development labels against recovered H001 data, reloaded all 24 saved models, and checked every contextual marginal against the training table. All passed. No production scoring or generator code was imported by this checker. Two combined local verification commands hit the 45 second tool limit; bounded individual seed checks subsequently passed with unchanged code and assertions. No training job was repeated.

All original evidence remains immutable. Hosted artifacts expire December 16, 2026. Same assistant authorship applies to implementation and checking; separate processes do not constitute independent agent or human review.

## Decision

Retain H002 as a verified development finding. High beta utility does not establish realistic geometric novelty or privacy. The proximity concern motivated H003, whose recovered results and limitations are recorded separately. No final confirmation or generalization claim is accepted.
