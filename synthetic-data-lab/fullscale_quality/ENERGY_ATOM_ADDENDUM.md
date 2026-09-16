# Electricity zero value intervention

During validation, before opening the electricity final test, the structured coordinate decoder was found to map exact zero submeter readings to tiny positive values. The encoder clips an energy fraction to 1e-8 before taking its logit; ordinary sigmoid inversion returns 1e-8 rather than the original zero. This preserves approximate numerical error while destroying the frequency of the physically meaningful off state.

In the complete real validation table, the three submeter zero fractions are about 91.97 percent, 69.99 percent and 42.35 percent. In the structured copula seed 17 validation table they are all zero percent. A hand constructed exact zero roundtrip and an exact unit fraction roundtrip fail the original decoder. Those failing tests are retained before repair.

The prospective intervention restores the lower clipped logit endpoint to exactly zero and the upper endpoint to exactly one before decoding each energy fraction. Ordinary nonzero fractions remain unchanged. Encoding, generator weights, training data, epoch coverage, context draws and random seeds remain unchanged. This is a numerical representation correction, not a new trained architecture.

Retain the original 39 electricity candidates and their original selections. Add repaired decoding variants of all 24 structured candidates, each generating 100,000 validation rows using its original validation generation seed. Compare paired original and repaired outputs, including off state frequencies, all original quality checks, and changed row counts. Use the unchanged validation selection rule over the extended 63 candidates. Freeze all extended candidates and their selections before final electricity test evaluation. Run the same complete synthetic training and test sizes and unchanged downstream utility criteria for the fifteen selected method and generator seed combinations.

This adds an explicitly validation driven iteration for electricity. Beijing and forest test outcomes must not be used to modify their generators or thresholds. The extended electricity experiment is version 1.1, not a silent revision of the original preregistration or a claim of equal tuning budgets across niches. Original artifacts are never overwritten.

Strict quality gates, source variation diagnostics and privacy limits remain unchanged. A fixed zero atom does not prove the full joint distribution is correct. The extended experiment must still earn its quality result through the frozen independent test.
