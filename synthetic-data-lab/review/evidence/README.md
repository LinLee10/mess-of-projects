# Evidence identity

`historical_delivery_receipt.json` describes the earlier local delivery. Its `remote_published: false` field is a historical fact, not the present publication state. Formatting was normalized for this copy; the artifact hashes inside it identify the original files.

`second_review_summary.json` describes the later review, including failed exact retraining and successful GitHub execution. The raw temporary local replay directory did not survive a scratch environment reset. Its summary was reconstructed from observed execution outputs, not recreated or claimed to be a retained full run.

The full newly executed GitHub evidence is in `../../ci_runs/34658103152-1/`. Those artifacts have a separate source commit, run identity, and candidate manifest. Do not merge their metrics with the historical run or present one as an exact copy of the other.
