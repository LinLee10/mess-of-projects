# H003R completed replication and geometric challenge

## Decision

The H003 adaptive beta result survived generation seeds 29 and 41 using unchanged implementation, inputs, learners and checker tests. Together with seed 17, it consistently reduces excessive proximity while retaining near real tree utility. It is a reproducible development tradeoff, not a promoted universal generator or established original contribution.

## Experiment and verification

Run 35283101267 completed both jobs successfully. Job IDs are 105409299781 for seed 29 and 105409299994 for seed 41. Each used all 348563 training records, generated 348563 records per method and scored all 116314 real development records. No reserved test was read or scored. Original H003 source f698bbbff3059d7ba7c539de106ebb9573b4088f was frozen; orchestration source aa6858f7fbbcdf9196af9dcdcd7b6debf6229b69 and preregistration 58771a5a15123c7582ae4d9796a0774f54378e2c are distinct.

All 31 unchanged implementation and checker controls passed in each job. Hosted full generation reconstruction, quality, distance, inference and arithmetic checks passed. Both archives were retrieved; archive hashes and 206 internal file digests matched. A fresh readonly checker reloaded all 12 additional students, verified development labels, recomputed their metrics, checked contextual marginals and recomputed saved proximity counts. All passed. Twenty local checker, diagnostic and archive corruption controls passed. The pilot geometry arrays matched seed 17 exactly in both new fits. Source and original tests matched byte for byte; generated Python bytecode caches did not, which is recorded separately rather than misreported as a source mismatch.

## Three seed results

Percentages below are means across generation seeds 17, 29 and 41. The stated uncertainty is sample standard deviation across these three seeds, not population uncertainty.

Neighbor raw: tree balanced accuracy 86.793288 percent, standard deviation 0.422841 percentage points; linear balanced accuracy 51.396777 percent; unusually close fraction 53.778609 percent.

Neighbor rank: tree balanced accuracy 86.877423 percent, standard deviation 0.320404 percentage points; linear balanced accuracy 51.100011 percent; unusually close fraction 46.645131 percent.

Adaptive beta: tree balanced accuracy 86.187283 percent, standard deviation 0.364173 percentage points; linear balanced accuracy 50.746727 percent; unusually close fraction 5.706286 percent.

The matched real tree reference is 86.333882 percent balanced accuracy. Adaptive beta is 0.146600 percentage points lower on average. It produces an average 104154 correct predictions, versus 104420 for real data. The mean ordinary accuracy and balanced accuracy must not be conflated. Both new seeds passed the prospectively registered directional proximity and two percentage point utility screen. Passing that screen is not a promotion decision.

Adaptive beta has exactly two training copies and 348563 unique rows in every seed, retains exact contextual numerical marginals, and passes all limited hard rules. Its linear utility remains below both neighbor controls. Rarest class recall averages 79.282622 percent, with 2.525736 percentage point standard deviation, versus 80.766852 for neighbor raw, 79.468151 for neighbor rank and 77.551020 for the matched real reference. The initial seed 17 rare class regression is therefore not a consistent regression against real data. There are only 539 development cases in that class.

## New exploratory finding: one good tail statistic is not distance distribution matching

The unusually close diagnostic counts synthetic distances below the real development fifth percentile, using nearest training records within identical cover, wilderness and soil contexts after training standard deviation scaling. Adaptive beta achieves 5.706286 percent versus 46.645131 for neighbor rank and 53.778609 for neighbor raw. This is a large, replicated improvement and is not a formal privacy guarantee.

However, within the 110 contexts having at least 100 development records, 70.880978 percent of adaptive synthetic records fall below the corresponding real median distance. These contexts cover 337216 generated records, or 96.744634 percent of each table. The same statistic is 50.181129 percent for a deterministic random split of real development distances. The real control uses smaller reference samples and is descriptive, not a matched hypothesis test.

The mean context weighted KS distance is 0.220116 for adaptive beta, improved from 0.582477 for neighbor rank, but above 0.044338 for the real versus real control. Its median pooled distance is about 82.1 percent of the real median. The conditional fifth percentile fraction is also near six percent, so the evidence does not support claiming that pooled tail matching hides a severe conditional tail failure. The stronger supported conclusion is that tail improvement coexists with a persistent mismatch in the central distance distribution.

## Mechanism challenge and next unit

For seed 17, actual synthetic median distance is about 82.45 percent of the training leave one out target across supported contexts, while the real development target ratio is about 100.21 percent. This points to undershooting of the fitted training objective, not merely development distribution shift. A nearest anchor switching explanation is insufficient: generated anchor distance has median 0.214519 versus nearest distance 0.209996, both below the real median 0.255871. Only 15.29 percent have a strictly closer alternative anchor. Nonlinearity from beta perturbation, marginal reassignment and discrete features remains a hypothesis, not a demonstrated cause.

The next bounded mechanism study should measure the actual postassignment distance response to bandwidth using training data only, then test direct calibration against the existing one step squared ratio rule. Preserve the same marginals, populations, students and unchanged controls. Do not tune on the reserved test. Five generation seeds, pilot seed robustness, repeated student training, relationship prediction, rare context sensitivity and Beijing transfer remain open.

## Boundaries and evidence

Same assistant authorship applies to investigator, implementation and checking. Independent computation is not external agent or human review. Historical final test exposure is unknown. Copulas, adaptive smoothing and multidimensional synthetic data evaluation have prior art; scientific priority is not established. The original LLM research program remains unexecuted by this tabular cycle.

See ARTIFACT_INDEX.json, METRICS.csv and the source under units/H003R/review. All original artifacts remain unchanged. Hosted evidence expires December 16, 2026. No paid runner, main update, force push or new dataset was used.
