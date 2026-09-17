# Continuation review and execution notes

The fresh review source is retained under review. It reads saved generated tables, predictions, models and distance arrays without importing production generation or scoring functions. All 20 local controls passed. Hosted H003R jobs separately passed all 31 preserved original controls and complete generation, fidelity, nearest distance and model reconstruction checks. These are separate computations by the same assistant, not independent agent or human authorship.

The original full H003 sources and tests matched both replication archives byte for byte. An initial comparison that included generated Python bytecode failed because cache bytes differed across hosts. Source comparisons were then scoped to actual source; cache differences remain disclosed and no cache equality is claimed. Every downloaded archive and listed internal digest was checked independently.

Two early combined local recovery audit calls exceeded the 45 second tool ceiling. Individual seed audits completed with unchanged assertions. No training run was repeated for these timeouts. The initial workflow static validator also matched final_test_scored metadata when looking for a broad test_ substring; it was replaced by explicit checks for the three original test program names. No production test or experiment acceptance threshold was weakened.

The contextual distance analysis and real versus real split are exploratory analyses of development evidence. They were not used to fit, select or change the replicated generator. The real control uses deterministic permutation seed base 60617 and contexts with at least 100 total development rows. Report its finite sample and sample size limitations. No KS significance probability is used.

## Reproduction

Use Python 3.13 with numpy 2.3.5, scipy 1.17.0, pandas 2.2.3, scikit learn 1.8.0 and LightGBM 4.6.0. Download the archives identified in findings/H001, findings/H002, findings/H003 and findings/H003R. Verify and safely extract with review/recover_artifact.py. Preserve their original bytes.

Set RECOVERED_ROOT to the parent of directories h001-seed17, h003-seed17, h003r-seed29 and h003r-seed41. Set AUDIT_OUTPUT_ROOT to an existing output directory. Run review/test_audit.py, review/test_context_proximity.py and review/test_recovery.py. Then run audit_recovery.py separately for each experiment directory name. context_proximity.py accepts a run directory, H001 evidence directory and output JSON path. Use output names context_proximity_seed17.json, context_proximity_seed29.json and context_proximity_seed41.json. Run aggregate_results.py last. real_reference_control.py accepts the H001 evidence directory, H003 seed 17 evidence directory and output path.

This supplemental code does not replace the original H003 acceptance checker. Retain and inspect the hosted AUDIT.json, tests and logs as well. Current evidence expires December 16, 2026.
