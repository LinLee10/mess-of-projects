# Final hosted artifact audit

The additional [GitHub review run 34659793166](https://github.com/LinLee10/mess-of-projects/actions/runs/34659793166) completed successfully. Its full report and log were committed in `1cd81a7c13ebd3ae899cc9d2c13d6e990cc52358`, under [review/runs/34659793166-1](runs/34659793166-1/). It audited the already committed experiment at `89eb2a73b64b3b829b84f1011fd0ce700f2aa6a9`; it did not change the experiment files or train new candidates.

This supplements the [second review report](../SECOND_REVIEW.md). All 46 existing software tests and all 14 checker tests passed. The original 34 tests were retained, twelve implementation review regressions were added, and the separate checker has its own fourteen controls.

## Verified scope

The checker imported no `synthlab_next` production modules. It independently checked the manifest contents, all six source dataset partitions, the expected full set of 1,044 evaluation conditions, saved test metric arithmetic, 696 stored neural development probability arrays, neural and logistic checkpoint inference, 84 neural generator weight and training ledgers, 96 probe checkpoints, and all twelve recorded development selections. The report lists the eight checked metric families, including per class metrics and confusion matrices.

## A failed check was retained and corrected

[The first additional audit, run 34659405623](https://github.com/LinLee10/mess-of-projects/actions/runs/34659405623), failed on reconstructed development log loss. All 46 software tests and the initial ten checker tests had passed. The checker had conflated a fresh floating point inference calculation with arithmetic on a stored probability array.

The corrected checker strictly recomputes stored test and neural development scores at absolute tolerance 1e-10. That acceptance threshold was not relaxed. It separately verifies reloaded probabilities with absolute tolerance 1e-6 and relative tolerance 1e-5, requiring every predicted class to remain identical. Four further controls prevent these inference tolerances from accepting altered stored metrics, material probability changes or changed decisions. The full [numerical contract](NUMERICAL_CONTRACT.md) records the distinction.

Logistic development arrays and probe development arrays were not saved separately in the experiment. Their reconstructed losses therefore provide weaker evidence than arithmetic on a preserved probability array. The checker records this limitation and all differences above the strict arithmetic threshold.

## Observed outcome

The final report records a maximum checkpoint probability difference of approximately 2.98e-7 in the second hosted runtime. Only four test probability arrays were bitwise identical there; every predicted class still matched. All 696 stored neural development score arrays passed strict arithmetic verification, and all twelve recorded development choices were recovered.

This does not conflict with the original hosted execution, where all 1,044 arrays reproduced exactly in its own runtime. It establishes numerical portability within the stated bounds, not universal bitwise equality. It also does not resolve the much larger differences observed when retraining complete models from the original protocol.

A separate implementation and a separate hosted runtime provide useful checks, but the same assistant authored the implementation review and checker. No external human reviewer approved the scientific claims. The additional browser agent review failed without a verdict and is not counted.

The final acceptance remains: published, executed, internally checked research pilot; not a stable method ranking, privacy certificate, production generator, demonstrated scientific novelty, or completion of the original language model program.
