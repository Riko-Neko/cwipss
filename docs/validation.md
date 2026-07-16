# CWT Candidate Validation

Validation re-examines CWT candidates in the original time series. It does not
operate on the CWT response map.

Validation and multiple-testing correction are implemented in
`cwipss.analysis.validation` and `cwipss.analysis.statistics`.

For each selected candidate:

1. choose the candidate channel span;
2. extract a time window centered around `t_peak_rec`;
3. use `period_rec` as the period seed;
4. search nearby integer-record periods;
5. compute autocorrelation, FFT periodogram, folding metrics, and shuffle/null
   p-values.

The output is evidence for review only. A low p-value or q-value is not a signal
claim.

Validation can skip vetoed candidates by default or include them with
`--include-vetoed`.
