# Second review of the synthetic data research extension

## Acceptance decision

The software is published, real tabular training has executed, and the saved outcome arithmetic passes an independently written checker. The broader research program is not complete. In particular, the evidence does not establish a stable winning generator, portable exact training reproduction, a privacy guarantee, or execution of the original six language model methods.

The review found six categories of implementation weaknesses and added twelve regression tests before correcting them. The original 34 tests were preserved. All 46 tests subsequently passed locally and on GitHub. More importantly, a complete fresh training replay changed results. That observation supersedes any unqualified interpretation of the earlier reproducibility claim.

## Publication and retained evidence

The project is in `LinLee10/mess-of-projects`, on branch `research/synthetic-data-reviewed-20260911`, inside `synthetic-data-lab`. It is an additive standalone extension, not a clone or fork of Arhaan's repository. The original repository is credited in `NOTICE.md`. Existing project files and the main branch were not modified.

Source publication commit: `41f4036bcb987a83cef0f14d65c6f1093b062874`.

GitHub execution evidence commit: `89eb2a73b64b3b829b84f1011fd0ce700f2aa6a9`.

[GitHub run 34658103152](https://github.com/LinLee10/mess-of-projects/actions/runs/34658103152) completed successfully. Its [execution manifest](ci_runs/34658103152-1/CI_EXECUTION.json) records successful tests, generation and training, final evaluation, analysis, separate artifact verification, checkpoint export, and schema screening. Its [complete evidence directory](ci_runs/34658103152-1/) contains the newly generated datasets, checkpoints, predictions, and logs.

These are new GitHub execution artifacts. They are not a silent replacement or an exact upload of the earlier session archive. The original delivery archive remains identified by SHA256 `c346790386c064dc016273e561ca27e3c8fc5cc97354c27f2c4d36be7b1f67ee`. The historical file receipt is preserved separately.

## What the review verified

The original archive's six delivery checksums matched. Its stored candidate and evaluation manifests were checked against actual files. A separate verifier recomputed neural network inference directly from saved tensors and logistic inference directly from saved coefficients. Its metric calculations do not call the implementation's scoring function or scikit learn's metrics.

The [GitHub artifact audit](ci_runs/34658103152-1/results/independent_audit.json) verified 3,695 candidate files and 1,045 evaluation files. All 1,044 stored prediction arrays reproduced exactly within that GitHub runtime. All 4,176 scalar metric checks passed, with a maximum log loss arithmetic discrepancy of approximately 5.6e-17. Accuracy, balanced accuracy, and macro F1 arithmetic agreed exactly. Partition identifiers were disjoint within each declared split.

Observed fit counts were 84 neural generators, 696 primary neural students, 96 probe neural students, and 348 logistic students. These are repeated fits of small tabular models, not language models. The counts describe executed conditions, not 1,044 independent statistical experiments.

Tests cover actual weight updates and checkpoint reloads, but tests alone cannot establish the scientific validity of the sampling distribution or the correctness of a general synthesis method. The [full CI log](ci_runs/34658103152-1/ci_execution.log) records all 46 passing tests and the executed training stages.

## Defects found and corrected

| Area | Observed weakness | Correction and evidence |
| :--- | :--- | :--- |
| Evaluation labels | Fractional labels could be truncated to integers; malformed label dimensions and unknown classes were insufficiently rejected. | Validate numeric type, dimension, finiteness, integrality, and class bounds before conversion. Regression cases cover each boundary. |
| Numerical features | Complex valued features could lose their imaginary component when converted to real arrays. | Reject complex feature and label arrays rather than silently changing their meaning. |
| Portfolio controls | Nonfinite temperature, prior strength, and support thresholds could pass scalar range checks. | Require finite controls before weighting or screening. |
| Prediction CSV | Duplicate feature names could replace a supplied measurement through dictionary parsing. | Require unique named columns, a valid model feature schema, and complete rows before writing predictions. |
| Schema screening | Ragged CSV input could fail after creating partial output artifacts. | Reject malformed rows before creating the screening destination. |
| Evaluation provenance | An evaluation manifest could be internally intact while naming a different candidate manifest. | Require the evaluation's candidate hash to match the verified training manifest. Replace critical analysis assertions with explicit exceptions. |

The twelve added tests initially produced nine failures, one error, and two passes against the submitted implementation. They were not changed to encode preferred method scores. The corrected tests are in [test_review_contracts.py](tests/test_review_contracts.py). Six restored local source files were also checked against their published Git blob identities.

## Reproducibility has three different meanings here

### Recomputing the original saved scores

The original stored labels and probabilities produce the originally reported metrics. This supports internal arithmetic consistency. It does not prove that the generating algorithm is scientifically faithful to a paper, that the data are appropriate for deployment, or that retraining will recover the same results.

### Reloading a saved checkpoint in another runtime

During local review, 348 of the original 1,044 prediction arrays matched exactly. The remaining neural probability arrays differed by at most approximately 3.58e-7, without changing any predicted class. That is strong numerical agreement, but it is not bitwise equality.

The newly published GitHub run reproduced all 1,044 of its own prediction arrays exactly. These two observations are compatible: same runtime checkpoint consistency and portability across runtimes are different requirements.

### Retraining from the original protocol

A complete local training and evaluation replay finished with successful exit codes. Its training split arrays and initial model weights agreed with the original, and its recorded Python and library versions matched. Nevertheless, none of the 876 neural final state dictionaries matched exactly.

Balanced accuracy changed in 515 of 1,044 evaluated conditions. Six of twelve development selected conditions changed. The largest individual balanced accuracy change was about 16.02 percentage points, and the largest change in an aggregated method group mean was about 4.69 points. There were 2,413 changed class predictions across the repeated conditions.

These differences are too consequential to dismiss as harmless formatting or a rounding change in a final report. The cause was not established. Small early numerical differences were observed, but the review did not isolate whether CPU dispatch, numerical libraries, runtime settings, or another factor caused their propagation. The current environment manifest is insufficient for that diagnosis.

The temporary full local replay directory was lost when the scratch environment reset. Its comparison summary was retained from the observed execution outputs and is explicitly labeled as such in the review record. It is not presented as a substitute for a complete retained replay. In contrast, the GitHub execution has committed full fresh artifacts and logs at an immutable commit.

## What the published GitHub run says about the proposed method

The table below is a descriptive extract from the [published mean results](ci_runs/34658103152-1/results/analysis_v1/mean_results.csv). It uses the neural student. All generator rows train on synthetic data alone; the real data row is the control. Values are mean balanced accuracy percentages across the declared repeated conditions.

| Training source | Wine | Diagnostic benchmark |
| :--- | ---: | ---: |
| Real data only | 98.77 | 96.31 |
| Within class interpolation | 98.61 | 96.04 |
| Numerical TabDDPM | 97.62 | 94.98 |
| Uniform generator mixture | 94.61 | 93.26 |
| Mixture with support screening | 94.89 | 95.00 |
| Mixture with utility weights and support screening | 95.32 | 94.68 |

The combined portfolio is not the best method in either domain. Support screening improved the mixture relative to the uniform mixture in this run, but the amount of improvement differed from the historical run. These observations do not establish a stable advantage from a particular component combination.

Utility weights use class specific development log loss. They do not directly optimize development balanced accuracy. The final model selection rule separately uses development balanced accuracy, then log loss. Confusing these criteria would misdescribe the experiment.

These comparisons use overlapping repeated splits and very small data. They do not supply a population confidence interval or a universal ranking. The real data baseline is repeated under generator seed directories even though it does not depend on a generated dataset; those copies must not be treated as independent evidence.

## Remaining engineering and scientific limitations

The numerical TabDDPM implementation is a numerical adaptation, not its complete mixed numerical and categorical benchmark. CTGAN and TVAE use local architectures and training budgets. No result in this repository reproduces a published headline score. The unconditional diffusion variant intentionally removes class information and then attaches the requested class label; it is a negative control, not a credible production synthesis competitor.

The training budget matches iteration counts, not actual floating point operations. CTGAN critic updates, rejection sampling, portfolio component training, and probe training have different costs. A claim of equal compute efficiency requires fuller accounting. Source selection also uses a small development set, so selection noise is a plausible explanation to test, not a demonstrated causal explanation.

A file hash establishes byte identity, not an independently timestamped preregistration. The general evaluation command checks stored artifacts but does not automatically execute the sealed code snapshot or compare every loaded module to it. The immutable source checkout recorded by CI and the separate verifier provide stronger evidence for this particular run, but do not eliminate that general boundary.

The Wine screen enforces finite nonnegative values, declared columns, and class identifiers. It does not verify chemical consistency or class correctness. Models and the portfolio reference arrays are not differentially private. Neither generated diagnostic data nor classifiers are clinical tools.

The original six language model methods remain outside this extension's executed track. Self Instruct, Evol Instruct, Magpie, STaR, instruction backtranslation, and Simula were not run with pretrained teachers and students by this work. The [research ledger](research/LITERATURE.md) is a selected set of relevant papers with unequal reading depth, not an exhaustive full text survey of the field. The portfolio uses a fixed candidate pool and one development weighting stage; it is not a demonstrated iterative process that invents new tasks from successive student failures.

## Review independence and next acceptance gate

The implementation review and independent arithmetic verifier were produced by the same assistant in separately executed checks. This is stronger than calling the production scorer twice, but it is not independent human authorship. A separate browser agent was assigned a read only source review, but the run ended with a tool block after 28 steps and returned no verdict. That attempt is recorded in `review/evidence/browser_review_attempt.json` and is not counted as a completed independent review. No independent human reviewer or independently authored training implementation approved this release.

The next acceptance gate is reproducibility diagnosis and stronger evaluation, not another claim of originality. Preserve the current source and results; record CPU and numerical backend details; repeat a small fixed subset in the same runtime and across runtimes; quantify source and selection stability; and preregister a genuinely independent, harder test before changing the proposed algorithm. Do not tune against the already inspected final tests.

The present release is acceptable as an auditable, executed research pilot. It is not accepted as a stable scientific winner, a production data generator, or completion of the original language model research program.
