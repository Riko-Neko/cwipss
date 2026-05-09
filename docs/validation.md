# CWT Candidate Validation

Validation re-examines CWT candidates in the original time series. It does not
operate on the CWT response map.

For each selected candidate:

1. choose the candidate channel span;
2. extract a time window centered around `peak_record`;
3. use `peak_period_records` as the period seed;
4. search nearby integer-record periods;
5. compute autocorrelation, FFT periodogram, folding metrics, and shuffle/null
   p-values.

The output is evidence for review only. A low p-value or q-value is not a signal
claim.

Validation can skip vetoed candidates by default or include them with
`--include-vetoed`.
