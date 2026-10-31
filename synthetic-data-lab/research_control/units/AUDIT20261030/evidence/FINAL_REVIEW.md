# PASS: bounded recovery repair

Candidate: `2a4129f699a45ecad4aeda2f7bf4902ba161c422`.
Comparison baseline: `a16346497fc791b5d7234c82d998cd245f1ba5dc`.
Checkout: `/private/tmp/scale-synthetic-final-review-20261030`.

The bounded recovery contract passes. No concrete remaining issue was found within the reviewed contract. This review used the isolated checkout and did not author or modify production code or tests.

## Focused source review

The production helper diff contains 19 additions and 3 deletions, 22 changed lines. The helper, original controls, frozen external controls, supplemental CLI controls and gate were read.

1. Noncanonical member spelling: `safe_name` now requires the raw name to equal its `PurePosixPath` representation. The external `./payload.txt` rejection control passes, with no destination created.
2. Distinct member aliases overwriting the same extracted file: the noncanonical alias is rejected before extraction. The additional normalized name set checks directory and file names after removing a directory entry's trailing slash. The external `payload.txt` plus `./payload.txt` overwrite reproducer now raises `ValueError` and creates no destination.
3. Duplicate manifest keys: `object_pairs_hook=unique_manifest_pairs` detects repeated keys before JSON object construction can discard the earlier value. Both conflicting and identical duplicate key controls pass.

Valid directory entries remain supported: `nested/` has its single directory marker removed for validation, and the explicit directory compatibility control verifies recovered `nested/payload.txt` bytes and a checked file count of one. The valid external fixture also verifies the receipt, manifest content, payload content, archive preservation and staging cleanup. Existing destination preservation and rejection cleanup controls pass. The CLI controls verify successful recovery and receipt, repeat destination rejection, and rejection of bad outer and internal hashes.

## Actual execution

* `python3 review_acceptance/run_recovery_gate.py --output final_review/gate`: exit 0, 27 passed in 0.17s, empty stderr. The gate reports `passed` at the exact candidate HEAD.
* `python3 -O review_acceptance/run_recovery_gate.py --output final_review/gate_optimized`: exit 1, empty stdout, stderr contains `RuntimeError: Optimized Python is not an accepted validation environment`. The fresh output path was absent afterwards. The explicit optimization guard executes before test execution and does not depend on an assertion.

Runtime: Python 3.12.8, macOS 15.6.1, arm64. `TMPDIR` was redirected under `final_review/tmp` and bytecode writing was disabled. The gate disables pytest plugin autoload and cache writing. No package installation or network access was used.

Full command arrays, stdout, stderr, exit codes, runtime, before and after hashes, and tracked preservation checks are in `execution.json`. The gate's own report, raw pytest logs and exact copied source/test files are under `gate/`. The gate's subprocess exit code is 0. Both protected test hashes match the gate's frozen constants.

## Exact identities

All SHA256 values below matched before and after execution.

| File | SHA256 before and after |
| --- | --- |
| `synthetic-data-lab/research_control/units/H003R/review/recover_artifact.py` | `69adb10b1894f1d6d8cce2af0e44dbd96917ca4d4c7d06045b43d510ca161392` |
| `synthetic-data-lab/research_control/units/H003R/review/test_recovery.py` | `63a3e6d16caeea82986877af2e3007f4d01fde9465199ef41754b68499441b94` |
| `review_acceptance/test_recovery_acceptance.py` | `8588320e2785ae4629e9f0f142047c2cdb50e6661197c52b2baed9728e888b42` |
| `review_acceptance/test_recovery_cli.py` | `6511484cf4cc6f84a9a2ddfc3b4ee5f3e2226bc89c115b3a8e9fa3b80c771d48` |
| `review_acceptance/run_recovery_gate.py` | `4695fbe63ffc38eda1c74a933d03e8d69ada23573e481ddb3d03d52153515368` |

Tracked files were clean before and after execution. `git diff --exit-code HEAD --` returned 0 with empty stdout and stderr, and HEAD remained the candidate. No tracked file or test was changed. All review writes are under `final_review`.

## Limitations and stopping point

This PASS is limited to the recovery API and CLI cases exercised by the frozen controls and the focused production diff. The comparison baseline's three failures were read from its retained baseline review; they were not rerun in this final review. The candidate's corresponding rejection tests were actually executed. This is local validation on the stated interpreter and platform, not a remote CI result or cross-platform qualification. No scientific review, training, holdout access, general race analysis or resource exhaustion investigation was performed. No deployment, publication, commit or push is authorized or implied.

Review complete. Next action: none within this bounded request.
