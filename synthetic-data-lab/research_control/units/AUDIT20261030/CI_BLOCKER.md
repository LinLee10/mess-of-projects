# Hosted CI is blocked by repository policy

The repository settings read returned enabled false and sha_pinning_required false. The exact read command, output and exit code are retained in evidence/actions_permissions_receipt.json. No repository setting was changed. The pushed commit had zero Actions runs and zero check runs when inspected.

The new workflow is committed, but it has not executed. Local validation and the isolated reviewer both passed all 27 recovery controls. The isolated reviewer also confirmed that optimized Python is explicitly rejected, and that protected source and test bytes remain unchanged. Read evidence/FINAL_REVIEW.md and evidence/final_review_execution.json.

Repository owner approval is required before changing this repository wide execution policy. Do not infer that a broad debugging request authorizes changing existing automation safeguards or unrestricted action permissions. Preserve the existing allowed action policy when considering reactivation. After an approved setting change, trigger only the scoped recovery workflow and retrieve both interpreter results at the exact candidate commit. Do not merge based on a workflow file, an empty check list or a locally passing test alone.

This block affects hosted enforcement, not the completed local recovery repair or the original full data verification. No claim that the hosted Python matrix passed is made.
