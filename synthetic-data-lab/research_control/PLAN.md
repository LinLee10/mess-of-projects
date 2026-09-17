# Resumable Synthetic Data Research Plan

## Objective and current evidence

Find a defensible new mechanism that improves useful synthetic data in at least one niche, then test what survives adaptation to another niche. Keep the hypothesis search open. Bound each execution unit, preserve failures, and require evidence before claiming originality, accuracy gains, or transfer.

This document is an execution plan, not a completed experiment or a promise of uninterrupted background operation. No new model training was performed while preparing this plan.

The inspected research branch is `research/synthetic-data-reviewed-20260911` in `LinLee10/mess-of-projects`. Its starting commit is `b08b4c1d4838ddc7f6e04ae82375dcd62ceb3007`. Its recovery protocol records 807 missing artifacts referenced by an interrupted v4 experiment. That statement is evidence of a recovery problem, not permission to reconstruct results from chat tables. Earlier numerical and novelty claims must be linked to retained artifacts and independently checked before use as baselines. [1]

Both original archives are present in the current workspace, and their bytes match the supplied SHA256 values:

| Source | Expected rows in the official release | SHA256 of the retained archive |
| :--- | ---: | :--- |
| Beijing | 420,768 | `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4` |
| Covertype | 581,012 | `89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d` |

Byte verification does not establish scientific data quality. Original train membership arrays are not currently present with the recovered schemas. Matching reported row counts will not be described as reproducing the original partitions. [1]

## Operating model

Use one canonical `STATE.json`, this plan, and an append only hypothesis and experiment ledger. At session start read the state, then only the source, papers, and evidence needed for the next unit. The conversation and uploaded reports supply context; artifacts supply measurement evidence. Do not restart literature review or invent another versioned pipeline after each interruption.

One work unit has one question and one deliverable: validate an archive, write checker controls, inspect a paper mechanism, fit one configuration and seed, score one artifact, or reproduce one comparison. An orchestration turn should select, execute or inspect one such unit, persist it, and report briefly. A compute job can take longer than a turn only on an explicitly launched runner with a recorded job ID and recoverable checkpoints. Do not imply that a chat response itself schedules continuing work.

Before every job, record its immutable ID, command, input hashes, code commit, dependencies, expected artifacts, compute ceiling and retry policy. Record the run as started before execution. Afterward record the exit status, partial work, output hashes, storage location and next action. A successful process exit is not equivalent to a verified result.

Checkpoint weights, optimizer and scheduler state, random number generator states, data order or sampler position, epoch and batch position, and fitted preprocessing. Preserve unfinished checkpoints separately from completed fits. Implement saves atomically. Before expensive execution, demonstrate an interrupted training unit can resume and match an uninterrupted control within a declared reproducibility contract.

Copy completed evidence to durable storage and verify retrieval before advancing. Git stores small source and decision files; model weights and generated tables need actual recoverable artifacts, not just a manifest digest. Record retention and expiry for hosted artifacts. Recover remotely after a local reset rather than relying on temporary sandbox paths. Never mark a unit verified until the checker and storage readback pass.

No unlimited jobs, purchases, hidden paid compute, force pushes, edits to historical results, or changes to unrelated repository content. No TinyFish. Check actual tool, runner, write and agent capabilities before assigning work. A permission or resource failure must save a blocked state and the exact smallest action needed, not trigger repeated blind attempts.

## Execution phases and gates

| Phase | Work units | Evidence required before advancing |
| :--- | :--- | :--- |
| 0. Recover truth | Inventory source, checkpoints, tables, final test access, logs, run IDs and claims. Recheck archives. Locate project operating instructions. | A claim ledger classifying evidence as verified, retained but unchecked, missing, or contradicted. No historical score is silently promoted. |
| 1. Prove recovery | Add the minimal state and checkpoint handling to the usable runner. Exercise interrupted save, resume, artifact upload, retrieval and corruption rejection. | A recoverable training control and checked storage receipt. No new large fitting matrix before this passes. |
| 2. Validate measurement | Freeze tasks, splits, metrics, hard rules, corruption controls, real reference comparisons, and the benchmark version. | Checker controls pass; benchmark detects known defects without rejecting ordinary real variation indiscriminately. |
| 3. Reestablish baselines | Start on Covertype. Reproduce full data real student, interpolation and contextual statistical controls; add a well supported neural reference. | Retained generated tables, checkpoints and independently recomputed development results from matched recipes. |
| 4. Explore mechanisms | Select one falsifiable hypothesis from a live backlog. Read closest primary work, implement one intervention, and compare against its ablation. | A completed experiment card with effect sizes, regressions, cost and a keep, revise or reject decision. Repeat this phase as needed. |
| 5. Confirm promising findings | Expand seeds, use a second student family, replay from a clean environment, and challenge the proposed explanation. | Benefit larger than registered practical and uncertainty thresholds, with no unacceptable loss on required quality dimensions. |
| 6. Test reuse and volume | Transfer the recipe through a documented domain adapter to Beijing. Test larger synthetic volumes for finalists. | Separate within domain, transfer, volume, and temporal conclusions. All attempted conditions remain visible. |
| 7. Confirm and report | Freeze the candidate and claims, complete eligible final evaluation, and compare with closest prior art. | Retained final results, reproducibility receipt, comparison to relevant existing methods and a narrowly worded contribution. |

Each phase consists of multiple small units. Do not execute an entire phase as one uncheckpointed monolithic script.

## Data and evaluation contract

Use every eligible training record and count every exclusion. Neural mini batches are allowed; silent source caps and conveniently small evaluation subsets are not. Begin with one synthetic training sized table per configuration. If resources are limited, reduce model families or concurrent jobs before discarding source data. Legitimate miniature software tests are not scientific benchmarks.

The recovered schemas report Covertype counts of 348,563 training, 116,314 development and 116,135 test rows. Beijing counts are 210,240, 105,408 and 105,120 respectively. Recover or regenerate a versioned split specification and explicit membership artifacts; distinguish a reconstruction from a verified historical replay. Keep identical predictor groups together. Covertype feature identity separation does not by itself establish spatial independence. [1]

Beijing is initially contemporaneous PM2.5 reconstruction, not forecasting. Report a sensitivity comparison excluding PM10 and require feature availability appropriate to the stated task. A separate sequence experiment must preserve site identity, ordering, missingness dynamics, episodes and temporal dependence. Chronological partitions alone do not make a row generator a sequence generator. [1]

Audit prior final test access. If its independence cannot be established, label evaluation on it exploratory. A random reshuffle of already inspected records does not produce a genuinely fresh confirmation dataset. Additional data is not required to begin recovery and exploration. If fresh external data becomes necessary, stop and ask for the exact dataset before proceeding to that phase, as requested.

Adaptive model search uses training and development only. Freeze one candidate family decision and a limited confirmation comparison before opening a final holdout. Never reopen that holdout for another tuning cycle. Preserve all tried hypotheses and seed configurations. Repeatedly searching until a final test happens to look good is not a discovery procedure. [5]

## What quality must mean

A 100% rule passing rate is legitimate when a representation enforces the rules. Do not manufacture extra constraints to make the rate lower. Separate genuine domain laws, empirical plausibility screens and statistical population fidelity. A broad benchmark and sample screening have substantial prior art; their combination alone is not a demonstrated scientific invention. [2,3,4]

| Dimension | Interpretable output |
| :--- | :--- |
| Hard validity | Valid rows divided by all generated rows, with every rule, missing value policy and rejection reason shown. |
| Distribution and frequency | Actual marginal and conditional errors; event counts per 10,000 rows; differences from comparably sized real reference splits. |
| Relationships | Error when predicting each feasible feature from others, plus nonlinear and conditional checks. No claim of recovering causal truth from prediction alone. |
| Rare cases | Actual counts, class recall and tail event errors for regions defined using training data only. Preserve natural population proportions. |
| Diversity and copying | Exact unique and copied row counts, near copy diagnostics relative to independent real data, and coverage. No formal privacy claim. |
| Covertype utility | Correct real predictions out of the actual denominator, balanced accuracy, each class recall and uncertainty. Balanced accuracy changes weighting of the metric, not the source population. |
| Beijing utility | Mean absolute PM2.5 error in micrograms per cubic meter, alongside R squared as a variance statistic, not a row correctness percentage. |
| Temporal fidelity | Correlation as a coefficient, episode length and transition errors, across multiple lags and stations, not one appealing lag statistic. |
| Reproducibility and cost | Saved arithmetic, reload behavior, repeat generation, full retraining and selection stability separately; runtime, memory, all draws and repair cost. |

Do not average these into a universal data quality percentage. Use raw outcomes and real reference variability to set tolerances. Set minimum meaningful gains and allowed regressions before scoring the intervention. A finalist should improve the target use case while meeting independently specified validity, copying, structural and rare event limits. Better accuracy alone is insufficient.

Before trusting comparisons, test copies, almost copies, prototype collapse, shuffled labels, independent columns, distorted class frequencies, erased tails, invalid values and broken temporal ordering. Each defect should fail its relevant dimension, not necessarily every metric. Add graded corruption levels, genuine real data controls, and unseen checker fixtures. Select controls without using final test records. Use appropriate grouping or temporal blocks for uncertainty; repeated seeds are not independent populations. [2,3,4,5]

## Initial hypothesis queue, not asserted inventions

The first exploration target is conditional fidelity in Covertype. It avoids simultaneously debugging sequence generation, two schemas and multiple neural architectures. Existing chat scores motivate questions but are not accepted baselines.

| Candidate question | Minimal controlled comparison | Result that would weaken or reject it |
| :--- | :--- | :--- |
| Does pooled calibration destroy important context relationships? | Apply no calibration, pooled calibration and contextual calibration to identical source pools. Introduce shrinkage toward pooled estimates only as a separately tested sparse context change. | No reproducible conditional or minority class benefit, or unacceptable loss of diversity or tails. |
| Can a residual generator recover tails without losing the useful local structure of interpolation? | Compare interpolation alone, a Gaussian residual control, and a learned conditional residual under recorded equal data and comparable compute budgets. | Gains disappear against the simple residual or result from near copies, relabeling or extra compute. |
| Can disagreement between verifiers identify the exact structure to repair? | Compare a fixed repair rule with a development guided selector using validity, conditional frequency and independent student signals. Test an evaluator not used to select repairs. | Only the selecting verifier improves, errors are displaced to rare groups, or the improvement disappears with another evaluator. |

These proposals overlap known ideas. Search nearest prior art before implementation and again before making a novelty claim. Read relevant equations, supplementary details and reference code; record what is reproduced, adapted, or omitted. The investigator may replace this entire queue when evidence supports a stronger direction.

Start screening with two complete matched seeds, not tiny datasets. Promote only plausible candidates to at least five matched seeds and a second student family when feasible. Small intervals across seeds do not substitute for uncertainty over dependent source observations. Log all comparisons and use paired effects where the design supports them.

After three informative cycles with no meaningful improvement, reconsider the hypothesis family and closest literature. Do not keep turning hyperparameters merely to find a lucky score. A failed hypothesis supplies evidence for the next one; it does not stop the overall search.

## Agent roles and separation

Use at most three genuinely separate agents when available. The investigator owns hypotheses, primary literature comparisons and priorities. The implementer owns a dedicated source worktree and execution artifacts. The critic owns separate acceptance tests, artifact checks and counterexamples, and must not rewrite production outputs or loosen gates to make results pass. Only the coordinator integrates reviewed changes.

Parallelize independent work, such as prior art review while a fit runs, not conflicting edits or competing heavy jobs. The implementer cannot silently edit critic tests. A changed test or threshold requires a recorded scientific reason and a new benchmark version. Run original tests unchanged unless their contract is demonstrably wrong and the change is separately reviewed.

If isolated agents are unavailable, execute separate roles sequentially and disclose same assistant authorship. A fresh process or second script is independent computation, not an independent research endorsement.

## What will count as success

For the requested model contribution, require a new defensible mechanism relative to the closest prior methods, an improvement on a substantial full data task, preserved required quality dimensions, independently checked artifacts, informative ablations, and repeated training evidence. A niche specific success counts as such; successful retraining on another schema is an additional claim that needs its own evidence.

A reproducible counterexample, boundary condition or failure mechanism may be an original scientific finding even without higher accuracy. Record that contribution honestly, but do not use it to claim the requested improved generator is finished. Likewise, a performance gain without established novelty is a useful engineering result, not a new research contribution.

Search at least the closest few methods and their followup work, not merely identical phrases. Record the exact proposed difference, the strongest known equivalent and the experiment that discriminates them. Use the wording "not found in the documented search" rather than "not public anywhere else." Discovery has no guaranteed date. Continue exploration across bounded cycles; pause only for explicit data, permission, compute or confirmation evidence needs.

## Scale, transfer and reporting

For a confirmed development finalist, test 1x and 2x synthetic volume before the optional 0.25x, 5x and 10x extensions. Count full raw outputs, accepted outputs, repairs and rejection costs. Test utility at matched student compute and, separately, matched epochs where feasible. More generated observations are not more independent real information.

Move to Beijing only after a useful Covertype recipe is retained and repeatable, or after evidence makes Beijing the better primary niche. Transfer the algorithmic recipe with a declared schema and domain rule adapter; do not describe transferring forest weights into pollution features as established generalization. Beijing was already explored in this project and is not a pristine external domain. Evaluate sequence behavior separately and ask for new data before any new dataset phase.

Every conversational update should report only the completed unit, measured change, reliability status, blocker if any and next unit. Keep full paper notes, tables and failure logs in artifacts. Every reported number must identify its run, split, variant, seed aggregation, denominator and metric definition. Generate score tables from retained records instead of retyping them from conversation memory.

## Immediate next unit

Complete recovery inventory and a verified artifact retrieval test. Do not start another generation matrix yet. Produce a concise claim ledger and exact missing evidence list, then implement and validate checkpoint resume. `STATE.json` records this as the next action; no experimental job is currently authorized by this plan file alone or reported as running.

## Sources and reading boundary

This plan uses the available project conversation, uploaded research instructions, retained v3 schemas and protocol, the small data v2 report, the locally verified raw input archive, and the current repository recovery protocol. It does not certify that every historical claim or paper was independently reproduced.

1. Project recovery protocol at inspected commit: https://github.com/LinLee10/mess-of-projects/blob/b08b4c1d4838ddc7f6e04ae82375dcd62ceb3007/synthetic-data-lab/conditional_recovery_v5/PROTOCOL.md
2. Alaa et al. How Faithful is your Synthetic Data? ICML 2022. Publisher abstract and evaluation framing checked for this plan: https://proceedings.mlr.press/v162/alaa22a.html
3. Meehan et al. A Three Sample Hypothesis Test for Evaluating Generative Models. AISTATS 2020. Publisher abstract and copying test framing checked: https://proceedings.mlr.press/v108/meehan20a.html
4. Jiang et al. TabStruct: Measuring Structural Fidelity of Tabular Data. Primary paper abstract checked: https://arxiv.org/abs/2509.11950
5. Nakkiran and Blasiok. The Generic Holdout: Preventing False Discoveries in Adaptive Data Science. Primary abstract checked for adaptive evaluation risks: https://arxiv.org/abs/1809.05596
