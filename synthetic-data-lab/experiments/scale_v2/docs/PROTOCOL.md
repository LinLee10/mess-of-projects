# Full data scale and transfer protocol

This protocol is committed before any new final test scores are inspected. It is a new tabular experiment, not a reproduction of a published headline result and not execution of the original language model program.

## Questions

Does the usefulness of synthetic training data persist when every available real training row is used? Do larger synthetic tables help? Do the original four generator portfolio variants help on larger problems? Does explicitly reconstructing known feature equations improve consistency without damaging learning? Does a generator trained on red wine transfer to white wine, and vice versa? Does repeatedly fitting to synthetic outputs improve transfer without new target observations?

## Full source data

UCI red Wine Quality has 1,599 rows, white Wine Quality 4,898, Dry Bean 13,611, and Sensorless Drive Diagnosis 58,509. These total 78,617 source rows. This study uses the complete downloadable tables, not a capped or stratified small extraction. Source archives, exact member names, hashes, row counts, feature counts and attribution are recorded. UCI sources are licensed CC BY 4.0.

All source rows are assigned to train, development or test by a fixed feature content hash. Identical input rows stay in the same partition even if their labels differ. Expected proportions are 60%, 20%, 20%; exact counts follow the hash, not a forced row sample. No unused partition exists. The shared wine schema uses the same hash salt in both domains to prevent duplicate input leakage across transfer. The original wine score is explicitly mapped to the binary target quality >= 6 for a common task; this is not the old cultivar classification task. Dry Bean and Sensorless retain original categories. This experiment cannot establish unseen motor or operating condition generalization because trial identifiers are unavailable in the chosen source file.

## Implementations and coverage

The prior generators.py Git blob is pinned to 9be7fd0245d378a0b6295edfb95b70e293ffb9b6. A fail closed transformer exports the exact derived source used for this experiment. It changes only real row scheduling and identical transformer reuse. CTGAN retains its log frequency class choices, but draws within each class from complete shuffled cycles. TVAE and numerical DDPM use complete shuffled row cycles. Every successful neural generator must report at least one gradient exposure for every real training row. All students enforce the same requirement. Reusing a fitted transformation is keyed by exact training feature and label hashes, transformation parameters and seed, never by test data.

The executed generator families are interpolation, conditional Gaussian copula, CTGAN, TVAE and numerical DDPM. Three component ablations remove CTGAN packing, reduce its mixture transform to one mode, and replace DDPM quantile preprocessing with standardization. Original model architectures, losses and generative equations remain unchanged. These local architectures and budgets are not full paper reproductions.

There are three independent synthesis seeds and two neural student seeds on one fixed new partition. Generators use at least 1,000 updates or sufficient updates for nominal five full passes, whichever is larger. Exposure counts, not nominal epochs, determine acceptance. The primary student uses full row cycles and the same update count across native conditions within each dataset. Its fixed budget covers every row in the largest synthetic condition. A histogram gradient boosting classifier supplies a separate inductive bias, fitted on every permitted row, without internal early stopping or a hidden validation subset. Fixed iteration budgets do not imply matched FLOPs, GPU hours, total runtime, or optimal hyperparameter tuning.

## Comparisons

Each generator is tested as a replacement for real training data and as augmentation alongside the full real training partition. Every method receives the same class allocation and synthetic row count within a condition. No bootstrap copying method is advertised as a new synthesis method. The real data control is fitted once per student seed, not copied under each generator seed.

The four portfolio variants are uniform mixing, development loss weighting, support screening, and both. Component pools contain four times the real training row count. All pool rows are retained in artifacts. The primary synthetic training size equals the full real training size. DDPM and the combined portfolio additionally produce tables three times that size. Selecting a defined synthetic condition from a generated pool does not discard real source rows.

Dry Bean contains known deterministic relationships. Six derived quantities are checked: aspect ratio, eccentricity, equivalent diameter, solidity, roundness and compactness. The new comparison separately screens impossible primitive geometry, then recomputes those six quantities on exactly the same accepted rows. It does not alter class labels or silently repair impossible primitive measurements. Validity of six equations is not proof of validity of all shape factors, category labels, privacy or biological realism.

Wine transfer evaluates source real data, source synthetic data and source synthetic data plus every real target training row. Three generation stages using the copula compare repeated synthetic replacement against retaining target real observations at each stage. These are two adjacent domains with compatible feature and label semantics. Training the same architecture on Dry Bean or Sensorless is an algorithm portability check, not transfer of wine knowledge to unrelated motor or crop labels.

## Evaluation and evidence

Candidate fits and development choices are frozen with source and artifact hashes before a separate command loads final test arrays. Every test row is scored. Report correct predictions, total predictions, ordinary accuracy, balanced accuracy, class recall, F1 and log loss. Binary AUC is supplementary. Full table marginal discrepancies, correlation differences, exact copies, duplicate rates and named geometry violations are separate from student utility. Heldout distribution metrics are computed only after freezing. No single overall quality percentage is invented.

Independent checker code uses no production scorer. It checks source reconstruction, row accounting, source partition overlap, candidate identity, stored development and test metrics, all recorded row exposure ledgers, checkpoint identity and numerical inference. Failed generators or insufficient class yields remain explicit failures and never receive substituted outputs. A complete same seed DDPM fit and generation replay is retained for every domain. That establishes only within runtime reproducibility; no cross hardware equality is presumed.

Acceptance is bounded: an executed benchmark may pass software and evidence checks without demonstrating superior synthetic data. Any conclusion about a component must identify the changed lines, the comparison, data domain, student, resource budget and observed seed variability. Already inspected final tests may not be used to revise generators and call the revised score unbiased.

## Selection interpretation

The stored development ranking groups candidates by generation seed, target domain and learner. Real controls have their own seed zero group; the final report must compare to those controls explicitly and must not claim that per synthesis seed winners beat real training. Source only wine generation sees no target data, but the transfer development scores are observed on the target. Reporting those scores is not a zero target information model selection protocol. Transfer anchor conditions use all target training observations, so they do not demonstrate transfer without target evidence.

## Primary sources

UCI Wine Quality: https://archive.ics.uci.edu/dataset/186/wine+quality

UCI Dry Bean: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

UCI Sensorless Drive Diagnosis: https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis

CTGAN and TVAE: https://papers.nips.cc/paper_files/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html

TabDDPM: https://proceedings.mlr.press/v202/kotelnikov23a.html

This initial protocol fixes the experiment, not its outcome. Presence of a workflow or this document is not evidence that it completed.
