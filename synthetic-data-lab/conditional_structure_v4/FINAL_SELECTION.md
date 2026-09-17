# Final selection and evaluation gate

The following choices are recorded before opening the final real test results in this continuation. They supplement PROTOCOL.md. The three generation seeds are 17, 29 and 41. All eligible training rows were used. The selection rule is mean development balanced accuracy for Covertype or mean development absolute error for Beijing, subject to hard validity matching the real reference and exact copy fraction below one percent. This is a utility oriented rule, not an overall synthetic quality certificate.

## Selected recipes

Covertype: nearest neighbor interpolation with k=15 inside identical categorical bundles, context specific rank calibration, then reconstruction of the three illumination columns from slope and aspect with sampled joint training residuals. The mean development balanced accuracy was 0.8110415812. The same recipe without illumination reconstruction scored 0.8107161660, so there is no meaningful development accuracy advantage from that final step alone. Its purpose is to restore a separately measured geometric relationship.

Beijing: the retained within category random endpoint interpolation followed by pooled rank calibration. The mean development PM2.5 absolute error was 13.9316436692. Context calibrated nearest neighbor interpolation scored 13.9593817710. This small difference is not described as a demonstrated population advantage.

## Disclosed development amendments

A permutation repair prototype swaps values only inside identical categorical bundles to preserve numerical multisets while changing relationships. Greedy and stochastic versions were both tested; their failures are retained. They were not selected as winners.

A documented hillshade trigonometric basis was fitted only to real training data. It is public prior art, not an original physical law. Repairs using the conditional mean or sampled joint residuals were evaluated. A soft residual envelope is not treated as a universal hard validity rule.

A partition Gaussian density control was also added. It stores context leaf means, covariance factors and categorical counts rather than complete numeric row anchors. Fixed leaf sizes 128 and 512 were screened; the 128 variant and its context calibrated repair were repeated across all three generation seeds. These are combinations of established density estimation and structural modeling ideas, not claimed scientific firsts.

These were adaptive development expansions of the initial search plan. They are explicitly disclosed, rather than retroactively represented as initial preregistration. No further generator architecture changes are permitted after this selection record.

## Final comparisons

Evaluate the real reference, random endpoint interpolation with raw, pooled and contextual calibration, nearest neighbor raw and contextual variants, conditional Gaussian, whitened diffusion and partition density variants. Covertype additionally includes the selected geometric repair. Retain three seeds where completed and identify single seed controls as such. The permutation variants remain failure controls.

Use frozen LightGBM students, then a fixed ExtraTrees sensitivity learner. Add the predefined real plus synthetic augmentation tests. For Beijing include a PM10 excluded sensitivity test: the primary task is contemporaneous PM2.5 regression, not forecasting. Feature relationship predictors cover every numerical column for the selected comparison tables.

Volume checks use 0.25, 2 and 5 times the training count for the selected recipes and fixed seed 17. They are generated and their students fitted before final scoring. No claim about 10 times generation is authorized without execution evidence.

All candidate source files, parameter choices, trained students, generated tables and partition hashes must be sealed before final scoring. No final test outcome may guide further fitting or selection. A separate process checks saved metric arithmetic, selected checkpoint inference and source integrity. The same assistant authors the implementation and verifier; no independent human or LLM agent endorsement is implied.

Results require their own retained evidence. This document is a frozen plan and development selection record, not proof that final evaluation has completed.
