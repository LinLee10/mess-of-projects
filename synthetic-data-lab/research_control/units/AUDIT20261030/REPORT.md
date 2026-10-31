# Recovery and research continuity audit

Audit date: October 30, 2026, Pacific time.

## Decision

The original research source and its required H003R evidence are recoverable. The earlier claim that commit a163464 was unavailable is superseded by an actual clone and artifact retrieval. H004A remains the next original Covertype mechanism experiment. The private INT interaction studies are a separate research line and must not silently replace this plan.

This change repairs three reproduced archive recovery contract failures. It does not change a generator, any historical scientific artifact, any original test, the main branch, or the existing research branch. It does not establish scientific novelty or certify a production synthesis product.

## Source identity and reading

Repository: LinLee10/mess-of-projects. Reviewed research head: a16346497fc791b5d7234c82d998cd245f1ba5dc. All 26 Markdown documents in synthetic-data-lab were read, together with the relevant recovery, generation, distance and checking code. Other portfolio projects were excluded. The full historical ChatGPT transcripts were not available; visible project context and recovered handoffs were used without claiming complete chat retrieval.

The existing main reference was 0f4e17cf6aff36fadaca9a3698e16a49e1486b10. The existing research reference was a16346497fc791b5d7234c82d998cd245f1ba5dc. Their prechange receipt is retained in evidence/remote_refs_before.txt.

## Reproduced defects and repair

1. A noncanonical ZIP path such as ./payload.txt was accepted after extraction normalized it. The helper now requires canonical relative paths before extraction.
2. Distinct raw member names could address the same extracted file and overwrite earlier evidence. The helper now rejects canonical target collisions and rejects normalization aliases rather than accepting their final contents.
3. Duplicate JSON manifest keys silently retained the last value. The helper now rejects duplicate keys while parsing the original manifest.

These are demonstrated integrity and ambiguity failures in correctly pinned synthetic fixtures. They are not evidence of an attack on the historical archives. Every retrieved historical archive passed its expected whole archive digest and complete internal manifest.

The production repair is limited to recover_artifact.py. Explicit directory members remain supported. Concurrent filesystem races, hostile resource exhaustion and arbitrary receipt path conflicts were not established as solved by this bounded repair.

## Independent engineering controls

An isolated external Codex session authored six new tests before the repair. The original six tests passed on the unchanged baseline; the new suite produced three failures and three passes. Actual commands, stdout, stderr, failing fixture hashes and protected file hashes are retained in evidence/baseline_test_execution.json. The reviewer narrative is preserved unchanged in evidence/BASELINE_REVIEW.md.

The new independent test file is retained byte for byte under the repository root review_acceptance directory so its original source resolution remains unchanged. Its SHA256 is 8588320e2785ae4629e9f0f142047c2cdb50e6661197c52b2baed9728e888b42. The original test SHA256 remains 63a3e6d16caeea82986877af2e3007f4d01fde9465199ef41754b68499441b94.

The repaired source passes both suites. Fifteen additional coordinator controls exercise canonical path variants, ordinary ZIP directory compatibility, identical duplicate manifest keys, the actual CLI, corrupt outer hashes, corrupt internal hashes, repeated recovery refusal and cleanup. The combined local gate passed 27 tests. The gate pins both protected test files, rejects optimized Python, records exact commands and hashes, preserves failure logs and returns failure on timeout or altered evidence.

The existing 46 core tests and 14 artifact checker tests also passed on the Mac compatibility runtime. These are software compatibility checks, not a claim of exact scientific retraining in that runtime. Their complete outputs are in evidence/legacy_compatibility_execution.json.

The first external reviewer attempt failed authentication. A broad Codex attempt timed out without completed tests. A focused Codex attempt then completed the six tests and baseline execution. Failed attempts are not counted as approvals. Independent engineering test authorship does not establish independent scientific replication or human approval.

## Fresh full data verification

Five original GitHub Actions archives were retrieved: H001 seed 17, B002 seed 17, H003 seed 17, and H003R seeds 29 and 41. All five pinned archive SHA256 values and all 357 internal file hashes passed. The original train and development arrays contain 348563 and 116314 rows respectively.

The recovered H001 membership arrays are disjoint for training and development. The declared predictor hash split was recomputed and matched. No feature identity hash overlaps were found. All three H003 input records bind to the recovered H001 and B002 manifests and the exact training bytes. Reserved observations were not loaded or scored. H001 itself documented its partition as a reconstruction; this recovery does not prove the identity of an older missing partition.

The original full H003 checker ran separately for all three generation seeds in the recorded Linux numerical environment. All three passed. It exactly reconstructed nine generated tables, recomputed complete table quality and nearest distances with its separate algorithm, checked exact contextual marginals where required, reloaded all 18 saved student models and recomputed their development scores. This is fresh execution of the original verifier, not fresh training of the entire scientific study and not independently authored scientific acceptance.

The unchanged H001, B002, H001C, H002 and H003 control suites passed 66 tests in that environment. The separate private INT checkpoint also restored exactly and passed its 197 historical tests. Those counts refer to distinct research lines and must not be presented as one scientific validation score.

A fresh engineering fixture additionally exercised all three production generation methods, six newly fitted student models, saved artifacts and the original separate checker. It used 560 constructed training rows and 210 constructed evaluation rows, not a scientific Covertype subset. All stages passed. Deliberately changing a saved correct prediction count was rejected. This fixture supplies execution evidence, not a performance result.

Full archive receipts, input binding results and complete replay logs are retained in the accompanying audit evidence package. This repository report does not replace those raw records.

## Remaining blockers and required decisions

The source location, required input archive availability and reproduced recovery parser defects are resolved within this audit scope. The remaining research gates are not hidden software fixes.

1. H004A has not run. The bandwidth response mechanism and a successful repair remain unproven. Follow NEXT_STEPS.md before new candidate fitting.
2. Historical final holdout exposure is unknown. Existing records cannot support a claim of pristine final confirmation. New external data requires the user's explicit approval.
3. Five seed promotion, variation in pilot and learner fitting, broader dependency fidelity, rare context sensitivity, strong modern references, transfer and volume remain unfinished.
4. Scientific novelty and independently authored scientific acceptance remain unestablished. Passing recovery or arithmetic tests does not close either gate.
5. Original Actions evidence expires December 16, 2026. Preserve complete archives in durable project storage and verify retrieval before relying on them beyond that date. A digest or this report alone is not a backup.

## Reproduce the repair gate

From the repository root, run:

```sh
python review_acceptance/run_recovery_gate.py --output /tmp/scale-recovery-acceptance-new
```

Use a new output directory. The scoped workflow runs the same gate on Python 3.12 and 3.13 using ordinary public CPU runners. Workflow presence is not proof of execution; its completed run and exact candidate commit must be checked separately. No merge or scientific promotion is authorized by a passing recovery gate.
