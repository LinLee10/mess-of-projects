# H003R unchanged method replication

Registered September 17, 2026 after recovering all H002 and H003 evidence and before observing H003 generation seeds 29 and 41.

## Question

Does the H003 adaptive beta intervention retain its utility versus training proximity tradeoff across two additional generation seeds, without changing code, bandwidth targets, contexts, pilot randomness, students, partitions or metrics?

## Frozen execution

Use implementation and checker source at f698bbbff3059d7ba7c539de106ebb9573b4088f. Run all three methods, neighbor_raw, neighbor_rank and adaptive_beta, separately for generation seeds 29 and 41. The fitting pilot remains seed base 24017. This tests generation variability, not variability across independently fitted pilot seeds or independent populations. Tree learner randomness remains 991 as in H003.

Use the retained H001 seed 17 training and development tables and retained B002 generator context definition. Verify every input manifest before fitting. Full sizes are 348563 training and generated records and 116314 development records. No arbitrary row caps, final test reads or new datasets are permitted.

The dependency gate consists of the unchanged H002 rank synthesis tests, H003 neighborhood tests and H003 checker controls. Preserve these tests exactly. The original H003 checker must pass separately after each fit, including generation reconstruction, complete table fidelity, nearest distance recomputation, inference reloads and metrics.

Run on standard ubuntu 24.04 CPU runners for this public repository, at most two concurrent jobs, 20 minutes per job, 12 minutes for fitting and five for the audit. No larger or paid runner, purchase, force push, main edit or automatic retry is authorized. Preserve complete or partial outputs even on failure. Each job must upload retained source, parameters, generated tables, model files, predictions, distances, logs and SHA256 manifests for 90 days. Retrieve archives and verify hashes before treating results as accepted or launching a successor.

## Analysis and acceptance boundary

Report every method and both learners for both seeds, plus aggregate means and sample standard deviations across seeds 17, 29 and 41. Report matched differences and all class recalls, especially the 539 record rarest class. No significance claim based on treating these seeds as independent datasets is permitted.

The replication question is directional: in each new seed, does adaptive beta reduce the unusually close fraction versus both neighbor controls while remaining within two absolute tree balanced accuracy percentage points of the frozen real reference? This prospective screen is not a promotion standard. Count a failed seed explicitly rather than changing the screen or adding replacement seeds.

Separately challenge the adequacy of the existing pooled proximity metric. Report full distance quantiles and distance distributions stratified by class and available contexts. Treat newly designed contextual diagnostics as exploratory analyses of development evidence, not preregistered confirmation. Do not tune bandwidth against their outcomes during this replication.

No successful outcome of this two seed extension by itself establishes scientific novelty, privacy, full population fidelity, rare group noninferiority, transfer, or final confirmation. Five seeds, pilot robustness, clean retraining and ablation requirements remain open.

## Provenance and separation

The workflow orchestration commit and the frozen implementation commit must be recorded separately. Original artifacts and source are immutable. Analysis and checker scripts do not modify production tests. Independent agent execution is unavailable in this session; the investigator and checker roles are performed sequentially by the same assistant and explicitly labeled as such.
