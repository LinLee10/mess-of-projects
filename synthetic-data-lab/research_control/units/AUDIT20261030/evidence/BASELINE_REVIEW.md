# Bounded recovery baseline review

Outcome: the unchanged recovery baseline fails three of the six new contract controls. The original six controls pass. No repair or scientific acceptance is claimed.

## Candidate and scope

Checkout: `/private/tmp/scale-synthetic-review-20261030`. Git HEAD: `a16346497fc791b5d7234c82d998cd245f1ba5dc`. This check covers the recovery function in `synthetic-data-lab/research_control/units/H003R/review/recover_artifact.py` only. Exactly six new pytest tests were written in `review_acceptance/test_recovery_acceptance.py`, using tiny synthetic ZIPs with correctly computed outer SHA256 pins. Fixtures use fixed ZIP timestamps. The internal hash is intentionally wrong only where necessary to test rejection or ambiguous conflicting records.

The supplied bounded contract and the existing contract derived acceptance plan supply the expectations. The source and original tests were read before the new tests were written, so this is not blind test authorship. No separate reviewer or subagent was used.

## Executed results

| Suite or control | Result |
| --- | --- |
| Original `test_recovery.py`, six tests | 6 passed, exit 0 |
| New acceptance suite, six tests | 3 failed, 3 passed, exit 1 |
| Valid pinned archive and exact recovered bytes | Passed |
| Noncanonical member path rejection | Failed: no ValueError raised |
| Distinct member alias overwrite rejection | Failed: no ValueError raised |
| Duplicate JSON manifest key rejection | Failed: no ValueError raised |
| Existing nonempty destination preservation | Passed |
| Cleanup after internal digest rejection | Passed |

Both suites ran immediately after test creation against the unchanged source with Python 3.12.8 and pytest 7.4.3. Exact command argument arrays, complete stdout and stderr, exit codes, environment overrides and elapsed times are in `baseline_test_execution.json`. Bytecode writing and pytest cache writing were disabled. The initial attempt to invoke `python` failed with exit 127 before executing any Python code; the installed `python3` then ran both suites successfully. That setup failure is also recorded.

## Exact reproduced failures

1. A ZIP containing only `./payload.txt` and a manifest mapping `payload.txt` to the correct payload digest is accepted. The retained output contains `payload.txt` with bytes `noncanonical`. The raw member name is noncanonical even though extraction normalizes it.
2. A ZIP containing `payload.txt` with bytes `original evidence`, followed by `./payload.txt` with bytes `replacement evidence`, is accepted when the manifest lists the final payload digest. Readback confirms the accepted destination contains `replacement evidence`. Distinct raw member names bypass the duplicate name check and resolve to the same extracted path. This demonstrates an overwrite within recovery staging, not a write outside staging.
3. A manifest with two `payload.txt` keys, first an all zero digest and then the correct digest, is accepted. The retained raw manifest still contains both conflicting records. Normal JSON object loading retains the last value, so the ambiguity disappears before internal digest verification.

All three failed at the unchanged rejection expectation. The accepted output directories and original fixture ZIPs remain under `review_acceptance/pytest_tmp/`; exact relative paths, sizes, SHA256 values and failure payload readbacks are retained in `baseline_test_execution.json`. No expectations were weakened and no production fixes were attempted.

## Preservation and evidence identity

Production source SHA256 before and after: `29ab91e02b6746476c32b4fc663de7a6aba8f73fbd6ee1a9aa5946a535134af7`.

Original test SHA256 before and after: `63a3e6d16caeea82986877af2e3007f4d01fde9465199ef41754b68499441b94`.

New acceptance test SHA256: `8588320e2785ae4629e9f0f142047c2cdb50e6661197c52b2baed9728e888b42`.

All eight Python files in the owning review directory have matching before and after hashes in the execution record. All files listed in the preexisting `protected_before.json` also still match that snapshot. Existing review evidence was preserved. Git status remains `?? review_acceptance/` with no tracked changes.

## Limits and next action

This is a bounded synthetic engineering check of the callable recovery API on macOS. The valid fixture checks one nested payload; the noncanonical control checks the leading `./` spelling; the existing destination control checks a nonempty directory; cleanup checks a digest failure after extraction. The original controls additionally exercise a wrong outer pin, corrupt internal digest, unlisted file and selected unsafe paths. No general completeness beyond those cases is claimed.

The CLI receipt write path, other path spellings, other destination types, concurrent races, resource exhaustion, historical scientific artifacts and scientific results were not tested. The original tests were not changed. No packages were installed, network requests made, other projects accessed or Git mutations performed.

Next action: review these three reproduced contract failures and authorize a separately scoped recovery repair if desired. Repair and subsequent acceptance are outside this check.
