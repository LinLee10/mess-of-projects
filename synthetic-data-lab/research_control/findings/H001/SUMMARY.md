# Completed calibration research cycle

Recovery, a full Covertype calibration experiment, and a second learner challenge completed. The tested correction improved the boosted tree learner but harmed the linear learner on class balanced accuracy. It is not promoted as a universal quality improvement or a novel generator.

## Data and experiment

All 348,563 eligible training rows were used. Every candidate generated a same sized synthetic table, and students were evaluated on all 116,314 real development rows. The 116,135 reserved rows were not scored. Source release size was 581,012. Natural class and category counts were preserved; balanced accuracy weights class recalls equally and does not mean the training population was rebalanced.

Generation seeds were 17, 29 and 41. Each seed used one fixed interpolation pool with four treatments: no correction, pooled stable rank correction, pooled ID keyed correction, and ID keyed correction within identical cover, wilderness and soil groups. All correction targets came from real training data. The label could condition generation but was never a student input.

There were 15 boosted tree fits and 15 logistic fits, including repeated real reference controls. This is one dataset and one development split, not 30 independent population experiments. No new neural generator fit occurred in this cycle.

## Measured outcomes

| Training source | Tree correct out of 116,314 | Tree balanced accuracy | Linear balanced accuracy |
| :--- | ---: | ---: | ---: |
| Real reference | 104,420 | 86.33% | 51.21% |
| Raw interpolation | 97,708 | 80.47% | 53.65% |
| Pooled stable rank correction | 95,911 | 71.78% | 52.49% |
| Pooled ID keyed correction | 95,936 | 72.04% | 52.49% |
| Contextual ID keyed correction | 99,059 | 82.18% | 50.80% |

Counts are rounded means across three generation seeds. The contextual correction gained 1.71 percentage points of tree balanced accuracy versus raw interpolation, approximately 1,351 more correct real cases by ordinary accuracy. It lost 2.85 balanced accuracy points for the linear learner. Both directions held in every seed. Tree recall of the rarest class, 539 development records, fell from 82.56% to 81.20%.

Contextual correction improved class conditional KS distance from 0.07686 to 0.01623 and average tail event frequency error from 0.00640 to 0.00158. Pooled and contextual marginal KS distances were nearly identical, about 0.00272 and 0.00274, despite very different tree utility. These are distribution errors, not good row percentages.

All candidate rows passed the limited hard range and category rules. Contextual outputs contained two exact training copies per 348,563 records in each seed. Neither statistic establishes complete realism or privacy. Near copies, full feature relationship prediction, stronger baselines and final confirmation remain pending.

## Source level finding

The surviving original calibrate function used stable sorting of numerical ties. Reordering a full interpolation table, applying that exact correction, and restoring record order changed 293,498 of 348,563 records, or 84.20%. Most changes were small: mean absolute differences were below one native unit per feature. ID keyed ties changed zero restored records in that test. This is order sensitivity, not an 84.20% data error rate or leakage of heldout answers.

On a deliberately constant numerical generator, legacy correction imposed almost perfect shared rank dependence: average absolute Spearman correlation across 45 feature pairs was 0.999673, versus 0.178664 in real data. Keyed ties yielded 0.001968, which also does not recover real dependencies. Midrank correction left constant output. A calibrator can manufacture attractive individual column distributions without recovering the relationships the generator failed to learn.

An initial source equivalence check also exposed integer truncation in the old output allocation. Floating input behavior matched; integer behavior differed in 11,538 records by at most one unit. The missing historical input dtype is unknown. No missing historic ranking was reconstructed from chat.

## Verification and storage

Run 35263683516 verified both raw archives, passed 46 unchanged project tests and tested exact mid epoch checkpoint continuation on a 47 row software fixture. Omitting randomness, optimizer or data order changed final weights; corruption and configuration mismatches were rejected. This is a scoped CPU recovery contract, not universal training reproducibility.

Run 35264679818 passed 18 new controls and completed all three full data tree experiments. Separate inference and metric checks passed. A further checker, supported by nine controls, independently recomputed complete table hard rules, marginals, conditional distributions, 140 tail events, 666 joint queries, copying, diversity and lineage for all three seeds.

Run 35265966995 passed four secondary controls and completed all three linear comparisons and separate inference checks. Its real reference fits differed by six correct decisions between hosts despite converging. The cause was not isolated; full cross machine training equality is not claimed.

All seven uploaded artifacts were retrieved and verified: 441,618,087 archive bytes and 475 internal file digests. The artifacts expire December 16, 2026; see ARTIFACT_INDEX.json. One local combined checking command timed out and passed when the remaining seed was run separately without changes. The dtype assertion failure is also retained.

Same assistant authorship applies throughout. Separate verifier code and processes are not independent human or agent endorsement. Historical final test exposure is unknown. These are development findings only.

## Originality and next unit

Conditional calibration, quantile mapping, tie handling and structural evaluation have substantial prior art. Cannon's MBC work already separates marginal and dependence correction and advises context partitioning. Li and colleagues show that common tie treatments can bias copula dependence. TabStruct and Measuring the Dependency Gap already question marginal or single target utility as adequate fidelity measures. No scientific priority or superiority over these methods is established here.

The project now has reproducible evidence of a specific code defect and a repair with a learner dependent benefit. The next bounded unit is a full data contextual Gaussian copula baseline using the same frozen training/development records and both fixed learners. This is needed before promoting a new mechanism. No new dataset is needed, and no job is left running.

Primary sources inspected, with scope recorded in the full report:

1. https://alexcannon.r-universe.dev/MBC/doc/manual.html
2. https://arxiv.org/abs/1612.06968
3. https://arxiv.org/abs/2509.11950
4. https://arxiv.org/abs/2607.21636

Execution runs:

https://github.com/LinLee10/mess-of-projects/actions/runs/35263683516

https://github.com/LinLee10/mess-of-projects/actions/runs/35264679818

https://github.com/LinLee10/mess-of-projects/actions/runs/35265966995
