# Synthetic Data Lab: reviewed execution

This is a runnable tabular research extension and a second review of its original outputs. It is published in an isolated research branch of LinLee10/mess-of-projects. Existing projects and main are unchanged. The upstream Arhaan2/synthetic-data-lab project is credited in NOTICE.md; this folder is not a fork or a complete implementation of its six language model papers.

## Evidence status

The original study contains 84 neural generator fits, 696 main neural student fits, 96 probe student fits and 348 logistic fits. Those counts and 1,044 stored test conditions were checked independently of the implementation's metric functions. The 34 original tests passed. Twelve additional review tests exposed input and provenance weaknesses before fixes; all 46 now pass.

A complete fresh execution of the original training protocol also completed. Exact replay failed: 515 balanced accuracy scores changed and only six of twelve development winners stayed the same. No claim of portable bitwise training reproducibility or a stable best method is made. The original saved metrics are internally consistent; matching their arithmetic is a different claim from reproducing training.

The source implements local CTGAN, TVAE, numerical TabDDPM, copula, interpolation, an explicit copying control, component ablations and four generator portfolios. It does not execute Self Instruct, Evol Instruct, Magpie, STaR, backtranslation or Simula with pretrained language models. No frontier research improvement or privacy guarantee is established.

## Run

Use Python 3.13.5 and requirements-executed.txt for the recorded dependency versions. CPU builds of Torch are sufficient. Exact floating point equality across systems is not guaranteed.

```bash
cd synthetic-data-lab
python -m pip install -r requirements-executed.txt
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m synthlab_next.cli train --config configs/pilot.json --output results/pilot_v1
PYTHONPATH=src python -m synthlab_next.cli evaluate --output results/pilot_v1
PYTHONPATH=src python -m synthlab_next.cli analyze --run results/pilot_v1 --output results/analysis_v1
python review/audit_artifacts.py --run results/pilot_v1 --output results/independent_audit.json
PYTHONPATH=src python scripts/export_wine.py
```

The analysis command is intentionally specific to the declared full pilot. A smaller custom configuration needs its own analysis contract. Generation and training reject existing output directories; do not overwrite historical evidence.

## Published execution artifacts

The repository workflow performs a fresh CPU execution behind all software tests, checks saved metrics using separate verifier code, exports models and screens the Wine CSV. On success it commits the new results, checkpoints and logs into a uniquely named ci_runs directory on this research branch only. Until its actual job and commit are verified, workflow presence is not evidence of completed execution.

These are new CI results, not an upload of the earlier session's binary archive. The original archive remains separately identified by its checksum in the review record. A changed winning condition in a fresh run must not be relabeled as the original result.

## Interpretation

The study covers two small numerical classification datasets with overlapping repeated splits. Each generator uses a fixed iteration budget, not matched FLOPs. Main students use matched update counts and class allocations. The portfolio's utility weights derive from class specific development log loss, while overall model selection uses development balanced accuracy then log loss.

A support screen rejects near copies and distant candidates using training data distances. It is not a domain correctness or privacy certificate. Hash manifests detect later artifact changes, but do not independently prove the time at which preregistration happened.

The legacy scripts/publish_to_github.py helper is retained and tested for optional upstream fork integration. It was not used for this publication. This branch was written through authenticated GitHub operations.
