# Source variation diagnostic

This addendum is recorded during validation and before final model testing. It does not change the original selection rule, six distribution gates, utility gates, or candidate matrix.

The complete real training tables compared with their separate real validation tables pass only about 68 percent, 72 percent and 83 percent of the original fixed frequency queries for Beijing, electricity and Covertype respectively. A copying control is expected to fail the copying gate, but frequency discrepancies between two real partitions also reveal source group variation. It would be incorrect to attribute every fixed frequency failure solely to a generator.

Report a supplementary source variation comparison beside, not instead of, the original fixed gate results. For each query compute the probability and a grouped variability scale from every source date or original forest block. With group size n_g, number of matching records a_g, overall probability p, total records N and G groups, the squared scale is G/(G-1) times sum((a_g-p*n_g)^2)/N^2. This is a grouped approximation, not proof of independent dates or a formal confidence interval.

The supplementary frequency tolerance is the maximum of the original fixed tolerance and three times the square root of the sum of training group variance, reference group variance and p_reference*(1-p_reference)/N_synthetic. Preserve each tolerance, source frequency, source group count and synthetic frequency in the output. This diagnostic uses every eligible record and never resamples a smaller source table. It is not used to pick a different generator or retroactively make the original benchmark pass.

Adjacent dates may be correlated and forest blocks are not verified geographic groups. Thus the adjusted comparison is an engineering diagnostic, not a statistical guarantee. Structural rules, copying, downstream learning and the original strict success rate remain separate and unchanged. All final generator choices are still frozen using validation only.
