# H004A: measured bandwidth response after marginal assignment

Registered before any new response measurements. This is a training only mechanism experiment, not a frontier generator claim or a final holdout evaluation. Parent source is 73f5e0c4717235ecd18c413b94686994e58ecd14. Preserve the unmerged recovery dependency, original generators, historical tests and evidence.

## Question

Does measuring the actual response of generated nearest neighbor distances to beta bandwidth repair the central geometry undershoot of the H003 squared pilot ratio rule? Can a median objective conceal a worse distribution shape? Paired inverse empirical CDF and exact marginal assignment on the same latent draws will separate the postprocessing effect from the overall response. Neither nearest distance matching nor lower copying establishes privacy or overall synthetic fidelity.

## Frozen inputs

Use all 348563 H001 training observations and all 232 B002 contexts. Use original H003 scale, training leave one out distances, pilot distances and bandwidths after verifying the source arrays and identities. H001 archive SHA256 is e9ea0efca9c240695fafcc201d9cf7d091ab24e2ca9dd20eb8d7f9211c2a3ad6; B002 is b5a351d1c0b4008ff2daa37ec7deea7bfd690d2fdd97041f4e4d9e327b1470ee; H003 seed17 is a79f3361089b439664eea525d5ef19d0736b1cb0ac1d9676be37de7cdfbac432. Their internal manifests must pass. Source identities of rank_synthesis.py, neighborhood.py and copula.py are separately frozen in the execution manifest. No development values, reserved observations, new data or student fits enter H004A.

## Fixed design

For context size n and frozen heuristic h, the grid is the sorted unique clipping to [1,n] of [1, h/2, h, 2h, 4h, 8h]. Record clipping, duplicate settings and boundary selections. Calibration seeds are 41017 and 41029. Audit generation seeds are 43017, 43029 and 43041. For a context g and seed s, initialize numpy default_rng(SeedSequence([s,g])). Reinitialize for each bandwidth. This is a matched seed convention, not a claim of monotonic common random numbers under beta rejection sampling.

Keep the existing identity keyed ranks and exact marginal assignment. Each original row supplies one anchor, without replacement, so this is a stratified adaptation of published empirical beta resampling, not its independent anchor algorithm. The paired inverse empirical CDF control maps each latent u to sorted_reference[min(floor(n*u),n-1)] and is not required to preserve exact marginal counts. Record both nearest distances and original anchor displacement before and after assignment.

Distance quantiles are 0.05, 0.25, 0.50, 0.75 and 0.95. Targets are the quantiles of the original training leave one out distances within each context. Candidate E selects the width minimizing the mean across calibration seeds of absolute median error divided by target median. Candidate Q selects the width minimizing mean absolute error across the five quantiles and both seeds, divided by target median. Ties choose the smallest width. A context with zero target median selects width 1. The selection uses only calibration measurements, is persisted and hashed before audit generation, and is not changed after observing audit outputs.

Audit four methods: unit bandwidth beta, unchanged H003 heuristic, E and Q. Use every original training row and context. Keep singleton, constant and zero target contexts in all accounting. Primary reporting population is contexts with n at least 100 and positive target median; report its exact row coverage and also all remaining contexts. Weight by original context counts, not equal context weights. No population confidence interval is inferred from generation seeds.

## Measurements and decision

Report weighted absolute relative median error, weighted normalized five quantile error, context KS distance against training leave one out distances, fractions below training fifth percentile and median, zero distances, exact duplicates, and contextual Spearman correlation error. Report class strata, sparse context behavior and bandwidth boundary selections. A larger width is not automatically a better generator. Preserve all settings, including failed and nonmonotonic responses.

The prespecified mechanism screen requires E or Q to reduce weighted median error by at least 30 percent versus the heuristic in every audit seed, with exact conditional marginals and counts, and no increase of more than 0.1 percentage point in the zero distance fraction. This is only permission to consider a separately registered utility challenge, not synthesis promotion. Any tail, dependence or rare context regressions must remain visible. A failed screen is a valid result; no replacement seeds or grid extension is allowed in this unit.

## Controls and evidence

A separate reviewer writes contract tests before implementation results are available. Test finite scalar controls, exact shape validation without broadcasting, nonfinite curves, tie breaking, zero targets, constant columns, duplicate rows, singleton contexts, and nonmonotonic curves. Compare distance computation to an independent brute force oracle on fixtures. Original tests are immutable. The reviewer owns its tests and any scientifically necessary revisions.

Before full execution, require the recovery gate, original research controls, new contracts, a complete small fixture, and corruption rejection. Retain command arrays, exit codes, stdout, stderr, runtime versions, input hashes, protocol and source identity, all per context calibration tables and distance arrays, frozen selections and complete audit tables. Check postassignment distances with a different nearest neighbor implementation and independently reconstruct selection from retained measurements. Use atomic writes and retain partial contexts on interruption; a successor cannot treat a partial result as complete.

CPU only, at most two computational threads, one scientific worker, 30 minutes wall time for the complete measurement pass. A separate verification pass has a 30 minute ceiling. Stop and retain partial evidence on resource failure rather than cap rows or rerun failed seeds. No paid service, settings change, merge, force push, original artifact overwrite or reserved test access. The repository Actions policy remains unchanged.

## Prior work and claim boundary

Segers, Sibuya and Tsukahara, The Empirical Beta Copula, arXiv:1607.04430; Kiriliouk, Segers and Tsukahara, Resampling Procedures with Empirical Beta Copulas, arXiv:1905.12466, Algorithm 3.2; Kojadinovic and Yi, A class of smooth, possibly data adaptive nonparametric copula estimators containing the empirical beta copula, arXiv:2106.10726. Beta smoothing, adaptive smoothing and rank assignment are established. The possible contribution here is an independently checked mechanism diagnosis and its empirical scope, not the invention of those components. Detailed reading scope and implementation departures will be recorded separately.
