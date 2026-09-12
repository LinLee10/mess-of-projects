# Numerical audit contract

The first hosted audit, run 34659405623, failed on a reconstructed development log loss after 46 existing tests and 10 checker tests passed. That failure remains in GitHub Actions history. Its checker incorrectly applied the strict saved arithmetic tolerance to a fresh inference calculation without consulting the already stored development probabilities.

The corrected checker separates these contracts. Every test score and every neural student development score is recomputed from the stored probability array and checked at absolute tolerance 1e-10. Those checks were not relaxed. Saved neural development probabilities also determine the recomputed development ranking.

Checkpoint inference is a separate numerical portability test. Recomputed probabilities must agree within absolute tolerance 1e-6 and relative tolerance 1e-5, and every predicted class must remain identical. The report records all nonexact arrays and their maximum absolute differences. These were already the first checker's test inference bounds; the correction applies the same distinction to development inference.

Logistic development predictions and probe development predictions were not stored separately in this experiment. Their reconstructed count and decision metrics must match exactly. Reconstructed loss metrics use the explicit inference bounds above and all differences greater than 1e-10 are reported. This is weaker evidence than recomputing from a saved probability array. Future experiments should store those arrays as well.

Four additional checker controls reject material probability changes, changes of predicted class even inside numerical tolerance, material reconstructed loss errors, and applying inference tolerance to a corrupted saved metric. Together with the ten original checker controls, there are fourteen checker tests. No production test or immutable experiment artifact is changed by this correction.

A successful checker result establishes its stated artifact and numerical contracts, not a scientific reproduction, external author review, privacy guarantee or performance advantage. The observed hosted job result remains authoritative; the presence of this document is not evidence that it passed.
