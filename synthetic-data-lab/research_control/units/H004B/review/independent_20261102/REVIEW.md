# H004B independent acceptance checker

Delivered independent checker code and 17 small unit/corruption controls. No H004B outcomes or real data were available. Full population execution, actual LightGBM model loading, and acceptance remain pending. This review provides no scientific endorsement or comparative performance result.

The implementation imports no production or historical scoring modules. NumPy is sufficient for the controls. LightGBM is imported only inside full verification and used only to load and infer tree models. Logistic inference is reconstructed directly from JSON using ten scaled numeric features, four wilderness indicators, forty soil indicators, and stable multiclass softmax.

## Supported checks

* Exact 19 condition matrix, comprising one real reference and six methods at each of three registered seeds. Exact 38 model file coverage, condition RESULT agreement with SUMMARY, and all seven development labels present.
* Every learner's saved labels, probabilities, confusion matrix, correct count, sample count, accuracy, balanced accuracy and seven class recalls in METRICS and condition/summary results. Confusion matrices and counts must be integers. Probability arrays must have exact shape, finite bounded entries, and row sums within absolute 1e-10. Metric arithmetic uses absolute 1e-14.
* Every loaded model's inference agrees at rtol 1e-9 and atol 1e-10 and preserves every predicted class, including nearly tied classes. Logistic classes must be exactly 0 through 6. Tree outputs must have seven columns and the model must have twelve features and 150 iterations. Linear convergence, iterations, warnings, configuration and scaling parameters are checked. Tree recipe metadata is checked.
* Complete H004B manifest coverage and hashes, COMPLETE seal, and tree DONE coverage/hashes. Every entry of each original input manifest is hashed. Original input manifests need not enumerate their own container or all extracted archive metadata. Files actually consumed are additionally bound through START and the frozen training file digest.
* Full H004A prerequisite counts/status, every prerequisite artifact hash, and all twelve consumed H004A table paths and hashes. Original training population, context membership, category ranges, source numeric marginals, and source-table binding are checked. No reserved table is loaded.
* Exact saved permutation from default_rng(88017). Every post-permutation int64 training table hash is checked in RESULT and learner metadata. Saved linear means and scales are recomputed from that same training table.
* All three iid reference seeds are reconstructed context by context. Uniform anchors, keyed original ranks, exact beta draws, original identities, context counts and shared latent lineage must agree. Both inverse and rank tables must exactly reconstruct from that lineage. This also verifies every contextual inverse marginal and the rank reference's exact original marginal multiset.
* Engineering screen is recomputed from per-seed metrics, with separate tree and linear balanced accuracy tolerances and all seven per-class mean recall vetoes. The geometry boolean is taken from the bound independent H004A prerequisite report. Every protected input, prerequisite and run file hashed at entry is rehashed before supported verification success.

## Explicit limitations

The checker returns `verified_supported_checks`, never an unqualified full acceptance status. The supplementary iid nearest distances, exact-copy counts, unique-row counts, contextual geometry and collapsed-column dependence diagnostic arithmetic are not independently recomputed. H004A geometry numerical rederivation is delegated to its bound prerequisite report. There is no independent audit of the historical checker's `checked` subrecords; those are not used as acceptance evidence. Authenticating the prerequisite author's identity or proving its prior execution cannot be done from an unsigned JSON report alone.

Full fit history, training recipe execution and resource use cannot be established solely from models and metadata. Runtime package identities are not independently authenticated against a pinned environment. The runner START records protocol commit identifiers but lacks protocol byte hashes. The delivered evidence identity includes current protocol byte hashes through the protected snapshot. Full verification has not run here and requires the coordinator's pinned Linux runtime with LightGBM. The coordinator must enforce the separate 30 minute verification ceiling externally. A complete scientific/diagnostic acceptance decision requires closure of these limitations rather than interpreting supported-check success as full protocol acceptance.

## Runner preexecution and deadline findings

1. `run_guarded.py` checks output and execution destinations only against input paths. It does not reject equal output/execution paths or execution nested beneath output. Its execution-directory creation then creates the output directory, and `run_utility.py` necessarily rejects the run because output already exists. Reject equal or overlapping destinations before making directories. This is a deterministic preexecution correctness defect.
2. The wall clock starts before admission, but `admit()` executes synchronously outside the bounded child. Slow admission is not interrupted at the 1800 second deadline. If admission consumes the budget, `run_bounded()` raises before EXECUTION.json is written. The worker process budget does subtract admission time, which is correct, but it does not bound the complete entry operation. Enforce a deadline around the entire entry and retain a failure record on prelaunch errors. TERM/KILL cleanup also permits up to ten seconds beyond the worker timeout; distinguish work deadline from cleanup deadline explicitly.
3. The H004A admission validates the twelve consumed audit table bytes and population counters, but not original protected inputs or student/source pins. Those checks occur later inside `run_utility.py` before fitting. That ordering does prevent fits on changed source/input bytes, but the guarded admission record alone is insufficient evidence that all prefit gates passed. START lacks hashes of the two H004B protocol files and full transitive dependencies. Record these before full fits as required by the protocol.

These findings are review observations. No production source, original tests or protocols were edited, and no runner was executed. One controlled writer was used; no subagents, installs, browsing, or Git mutations occurred. An initial directory search mistakenly enumerated AGENTS.md path names in sibling temporary directories. No sibling file contents were read, and subsequent work was confined to this checkout. This scope deviation is retained explicitly.

## Execution and evidence

The actual commands, stdout, stderr, UTC times and exit codes are in `logs/execution.json` and its referenced log files. The initial test run passed all 17 controls. A final run after stricter context, recipe and integer checks is retained separately. Failures, if any, remain in the log history. Tests use only fabricated small arrays and temporary files and exercise arithmetic corruption, absent classes, nonfinite probabilities, malformed shapes, invalid categories indirectly through encoding contracts, changed predicted classes, matrix omissions/duplicates, manifest tampering/traversal/coverage, JSON duplicates, convergence/scaling corruption, lineage corruption, and screen vetoes.

`logs/protected_before.json` and `logs/protected_after.json` enumerate SHA256 for every original non-Git file outside independent_h004b. `logs/protected_comparison.json` records their equality. Existing request/events/stderr artifacts in independent_h004b were preserved. `logs/deliverable_sha256.json` identifies the checker, controls and this review. There were no full-run inputs to snapshot; the checker itself supplies before/after file identity checking on full execution.

Next action: coordinator executes the exact delivered checker in the pinned Linux runtime against completed H004B artifacts with an external 1800 second verification limit, retains the result and logs, and evaluates the listed unsupported diagnostics separately.

```sh
python3 -B independent_h004b/verify_utility.py \
  --inputs ROOT --audit H004A_AUDIT --verification H004A_REPORT \
  --run H004B_OUTPUT --output NEW_JSON
```
