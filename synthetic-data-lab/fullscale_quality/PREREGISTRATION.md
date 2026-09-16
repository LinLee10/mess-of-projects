# Full scale quality benchmark protocol

This is a new experiment, separate from the original small tabular pilot. The complete official UCI archives for Beijing, Covertype and Individual Household Electric Power Consumption were acquired in Actions run 35154311702. The total original source count is 3,077,039 records. The source archive SHA256 values are b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4 (Beijing), 89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d (Covertype), and 9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff (electricity). Dataset attribution and CC BY 4.0 terms are retained.

## Scope and data boundaries

All original rows are audited. Training and testing use every eligible record, with no random row caps. Missing modeled measurements and invalid single field schemas are excluded with an explicit complete source ledger. Cross measurement inconsistencies remain in the sources and are reported. Shared dates remain together for time series sources; Covertype uses blocks of 1024 consecutive original records. These are snapshot experiments, not forecasts or geographic independence claims. Missingness and full sequence synthesis are outside scope.

The generators model numerical values conditional on empirical categorical tuples. Context tuples are resampled from the real training distribution, rather than learned by a neural categorical generator. This boundary applies to every method. Thus the experiment does not certify learning of the full mixed table distribution or novel categorical combinations.

## Fixed comparisons

Methods: Gaussian conditional copula, context matched nearest neighbor interpolation, conditional Wasserstein GAN, conditional VAE, and numerical conditional diffusion. GAN and VAE are local adaptations derived from those mechanism families, not exact CTGAN or TVAE reproductions. Diffusion is not the complete mixed feature TabDDPM implementation.

Generator seeds: 17, 29 and 41. Neural candidates: ordinary coordinates at six complete epochs, constraint coordinates at six complete epochs, then the same constraint coordinate model at twelve complete epochs. Statistical methods compare ordinary and constraint coordinates. Every neural epoch visits every eligible real training row exactly once, verified by exposure arrays. Minibatches have at most 2048 records; they do not reduce dataset coverage.

Constraint coordinates represent Beijing coarse particle concentration as fine particles plus a nonnegative gap and dew point as temperature minus a nonnegative spread. Electricity uses nonnegative energy fractions whose sum cannot exceed total energy. Covertype uses circular aspect and bounded slope and hillshade coordinates. Encoding changes to real inconsistent measurements are counted explicitly. Every method uses a complete empirical numerical CDF with randomized tie intervals and observed marginal support inversion. This can enforce single field limits by construction; it does not prove joint fidelity.

## Benchmark and selection

Report separately: row rule compliance, marginal KS distances, rank correlation differences, frequencies of training defined marginal and conditional queries, joint events, rare tails, rounded copies of real training records, and downstream utility.

Engineering distribution gates: at least 99 percent row rule compliance, at least 90 percent query agreement, at least 80 percent tail query agreement, maximum marginal KS no greater than 0.10, maximum rank correlation gap no greater than 0.10, and no more than 1 percent rounded training copies. Query agreement requires absolute probability error no greater than max(0.001, 0.10 times the real reference frequency). These are declared engineering tolerances, not a standard, a statistical confidence interval, or the fraction of records proven true.

Use 100,000 generated records for each validation candidate. Select lexicographically by number of distribution gates passed, total query agreement, tail agreement, smaller correlation gap, then smaller KS. Do not use final tests for selection. Preserve all candidates and all validation outcomes. Freeze candidates, code hashes, configurations and selections before final evaluation.

Final synthetic training tables have exactly as many records as eligible real training tables. A separate synthetic quality table has exactly as many records as the full eligible real test table. Downstream learners are a linear model and histogram gradient boosting, each trained on all records. Regression tasks predict PM2.5 and temperature for Beijing, active power and voltage for electricity. Covertype predicts cover type. Labels and regression targets are excluded from the respective predictors.

Utility gates: each regression learner and target must have MAE no more than 10 percent above the real training baseline. Each classification learner must lose no more than two percentage points of balanced accuracy and no more than five percentage points of recall in any class. A niche consistency claim requires every distribution and utility gate to pass for every one of the three generator seeds. Results of all methods, including failures, are reported.

## Acceptance limits

There is no automatic percentage of universally high quality rows. Passing this benchmark establishes only the declared snapshot constraints and utility. No formal privacy, causal truth, temporal fidelity, zero shot transfer between domains, or frontier superiority is implied. Fixed epochs do not equal fixed FLOPs. Public real datasets can themselves violate consistency screens; those defects are not hidden. The work and separately implemented checking are authored by the same assistant, not an independent human review.
