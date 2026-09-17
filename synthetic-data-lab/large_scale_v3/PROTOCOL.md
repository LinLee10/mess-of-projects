# Resumed large dataset experiment

This is an additive continuation from commit cc6cdfe7b31b560732729b5efe4da5101f9b42d8. The interrupted attempt saved archive verification and dependency workflows, not completed large experiment results. The current execution recovered their artifacts and verified the raw SHA256 values.

## Sources and partitions

Beijing archive SHA256: b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4.

Covertype archive SHA256: 89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d.

All 420768 Beijing station hours and all 581012 Covertype records are assigned to partitions. No arbitrary training cap is used. Beijing uses 210240 training rows before March 2015, 105408 development rows in the following year, and 105120 final test rows in the last year. Missing measurements remain represented through explicit masks; supervised tasks exclude only rows lacking the required target and report those counts. Covertype partitions by a hash of feature identity into 348563 training, 116314 development and 116135 final test rows. This avoids identical feature records crossing partitions but is not spatial validation.

## Methods and controls

The executed matrix is intended to contain global and conditioned Gaussian copulas, interpolation within identical categorical bundles, pinned TVAE components, a CTGAN adaptation with explicit full epoch coverage, local mixed numerical and categorical diffusion, and a local neural flow. Seeds are 17 and 29. Neural checkpoints are at four and twelve complete epochs. Full epoch traversal replaces the earlier fixed step assumption. Local architectures and changed training policies must not be presented as exact paper reproductions.

Each candidate generates as many rows as its real training partition. Raw outputs and a separate numerical marginal calibration using only real training values are compared on development data. No rows are silently repaired, dropped or replaced. Copying, shuffled labels, independent columns, collapsed prototypes, invalid values and erased upper tails are benchmark controls. Every record contributes to inexpensive rule, distribution, frequency, copy and coverage calculations.

## Selection and final evaluation

Per method, select the checkpoint and calibration variant across both fixed seeds using minimum maximum marginal KS, then mean KS, subject to a hard rule pass rate no worse than the real reference. If no candidate satisfies the rule gate, report that failure and prioritize validity before distance. This narrow selection objective is not an overall quality claim. Preserve all development outcomes and utility tradeoffs. Do not select a favorable seed.

Only after candidate files, source, configuration and these choices are frozen may final test metrics be computed. No further generator tuning may use those test outcomes. Apply the mechanically chosen recipe from each source domain to the other domain as a separate recipe portability comparison, not as transfer of trained weights or evidence of cross-domain causal learning.

Ridge and LightGBM students use all eligible rows. Final selected candidates receive synthetic only and augmentation tests against a real only reference. Additional structural checks predict numerical columns from other columns. Rare events use thresholds fitted on real training data. Missingness, category frequencies, copying and dataset collapse are distinct dimensions. Do not invent a universal quality percentage.

## Temporal and volume diagnostics

Ordinary independent row generators are not temporal models. A separate seasonal latent Gaussian VAR experiment and an IID ablation test persistence and cross-station relationships on an explicit calendar. Calendar validity from supplied timestamps is not a learned generator achievement.

A feasible fitted conditional copula is intended for volume checks at 0.25, 1, 2, 5 and 10 times the real training count. All deterministic checks and streaming aggregate counts must include every generated row. Declare any expensive diagnostics or student fits omitted at larger scales. More generated rows are not more independent real information.

## Integrity and acceptance

Retain input and output hashes, complete epoch counters, source snapshots, environment, model checkpoints, predictions, failures and runtimes. Test metric code using deliberately corrupted data and separately implement verification of stored outputs. Distinguish same-runtime retraining, checkpoint reload, seed variation and cross-runtime portability. Same assistant authorship is not an independent researcher endorsement.

This protocol was recorded during development execution and before final test scoring. It is not claimed to precede all development work. Completed results require their own verified evidence, not this document. No paid compute is authorized by this protocol. Existing project files, historical experiment evidence and main are not to be overwritten.
