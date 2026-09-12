# Synthetic Data Lab: reviewed execution

Real tabular generators, trained students, controlled comparisons, committed artifacts, and a second review of the earlier extension. Published in an isolated research branch of `LinLee10/mess-of-projects`; existing projects and main are unchanged. This folder is not a fork or a complete execution of Arhaan's six language model paper implementations. See [NOTICE.md](NOTICE.md).

## Start here

[Second review report](SECOND_REVIEW.md) records the implementation fixes, actual results, failed exact retraining and scientific limits. [Final hosted audit](review/FINAL_AUDIT.md) records the additional successful independent arithmetic implementation and its numerical contract. [Literature coverage](research/LITERATURE.md) accounts for thirty selected paper records without claiming equal reading depth or exhaustive field coverage.

[Training and evaluation run](https://github.com/LinLee10/mess-of-projects/actions/runs/34658103152) succeeded and committed [all fresh execution artifacts](ci_runs/34658103152-1/), including checkpoints, generated datasets, predictions, metrics and logs. [Separate artifact review](https://github.com/LinLee10/mess-of-projects/actions/runs/34659793166) also succeeded and committed its [full report](review/runs/34659793166-1/report.json). These are observed execution results, not merely runnable workflows.

## Evidence and limits

The executed study contains 84 neural generator fits, 696 main neural student fits, 96 probe fits, 348 logistic fits, and 1,044 evaluated conditions. All 46 software tests and 14 separate checker tests passed in the final audit workflow. The checker imports no production modules and independently recomputes metrics, model inference, source partitions, training provenance, and development selections.

A complete local retraining replay changed 515 balanced accuracy results and six of twelve development selections. Portable exact training reproducibility is not established. The original saved scores are internally consistent, but that is not proof of a stable algorithm ranking. The separate hosted checker also distinguishes stored arithmetic from tiny checkpoint reload differences; it does not silently call them exact.

Implemented methods are local CTGAN, TVAE, numerical TabDDPM, copula, within class interpolation, a copying control, component ablations and four static portfolios. The study covers two small numerical datasets with overlapping repeated splits. Generator iteration counts are not equal compute budgets. No novel superiority, chemical validity, clinical suitability or privacy guarantee is established.

Self Instruct, Evol Instruct, Magpie, STaR, instruction backtranslation and Simula were not executed with pretrained teachers and students in this extension. The portfolio is not a demonstrated iterative process that invents new tasks from student failures.

## Reproduce

Use Python 3.13.5 and `requirements-executed.txt`. CPU Torch is sufficient.

```bash
cd synthetic-data-lab
python -m pip install -r requirements-executed.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python -m unittest discover -s review -p test_audit_v2.py -v
PYTHONPATH=src python -m synthlab_next.cli train --config configs/pilot.json --output results/pilot_v1
PYTHONPATH=src python -m synthlab_next.cli evaluate --output results/pilot_v1
PYTHONPATH=src python -m synthlab_next.cli analyze --run results/pilot_v1 --output results/analysis_v1
python review/audit_v2.py --run results/pilot_v1 --output results/second_audit.json
PYTHONPATH=src python scripts/export_wine.py
```

The analysis contract is specific to the full declared pilot. Existing result directories are never overwritten. The checked in `ci_runs` data are immutable historical evidence and are not overwritten by these commands.

## Review independence

The second review and separately written checker were authored by the same assistant and executed in isolated checks. They are not independent human research endorsements. A separate browser review attempt returned no verdict and is not counted as completed approval. The failed checker run, its correction, and the failed exact training replay remain documented rather than erased.
