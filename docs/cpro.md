# Calibrated Period-Ridge Observation

Calibrated Period-Ridge Observation (CPRO) compresses one physical frequency
channel's `period x time` CWT power map into one edge-preserving 1D observation
axis. Physical frequency channels remain independent.

For raw series `x(t)`, first-difference MAD estimates white-noise sigma:

```text
sigma = 1.4826 * MAD(diff(x)) / sqrt(2)
```

For period-dependent unit-noise CWT gain `g(p)` and power `P(p,t)`:

```text
A(p,t) = P(p,t) / (sigma^2 * g(p))
tau = max(threshold_snr, quantile(A, texture_quantile))
```

Local period-ridge contrast compares a 3-bin center with the surrounding part
of a 15-bin context:

```text
R(p,t) = center_mean(A) / sideband_mean(A)
Q(p,t) = tau * sP * softplus(log(A / tau) / sP)
W(p,t) = sigmoid(log(R / Rmin) / sR)
S(p,t) = period_mean_3(Q * W)
activity(t) = max_p S(p,t)
```

The CWT grid includes the smallest safe period context outside the requested
candidate domain. Those context rows contribute to center/sideband and support
calculations at the two boundaries, then are masked before the strongest-period
reduction. They therefore stabilize edge candidates without becoming candidate
periods themselves.

Defaults are `Rmin=1.5`, `sP=1.0`, `sR=0.10`, and period support `3` bins.
The PELT boundary reduction is fixed Top-1 and has no time smoothing. Absolute
calibrated power controls `Q`; local period contrast controls `W`; native PELT
owns all time-boundary placement.

After PELT proposes active segments, CPRO measures whether each response is
supported from both time directions on a consistent CWT period ridge. For each
period row, with decay `d=0.995`:

```text
F(p,t) = (1-d) S(p,t) + d F(p,t-1)
B(p,t) = (1-d) S(p,t) + d B(p,t+1)
H(p,t) = sqrt(F(p,t) B(p,t))
E(p,t) = S(p,t) * [min(H(p,t), S(p,t)) / S(p,t)]^2
```

For a PELT segment `I`:

```text
cont_mean = mean_t(max_p E(p,t)) / tau
ridge_lock = max_p sum_t E(p,t) / sum_t max_p E(p,t)
```

The segment is retained when `cont_mean >= 0.47` and `ridge_lock >= 0.94`.
The first condition suppresses isolated vertical energy; the second requires
the surviving energy to remain on one period row. These are continuous response
and shape criteria, not duration cutoffs. `pelt_min_size_records=64` is the only
time-size constraint. There is no separate 96-record segment gate or 640-record
window gate.

CPRF receives each continuity-retained PELT window's indices and evaluates the
original, unmasked CWT2D slice. CPRO selects time proposals; CPRF owns final
ridge strength, concentration, persistence, local contrast, and harmonic
diagnostics.

## CPU and CUDA

`cpro.py` and `cpro_cuda.py` implement the same formula. Explicit CUDA mode
fails when CUDA/CuPy is unavailable; it never invokes CPU CPRO as a scientific
fallback.

CUDA retains CWT2D and the CPRO continuity map on device. Only the 1D
`shape_activity` axis and calibration scalars cross to CPU for native PELT.
Returned window indices are transferred to GPU; continuity reduction and CPRF
operate on resident 2D arrays. Only per-window continuity/CPRF scalar packs
cross back to CPU. Native PELT can overlap the next queued GPU block.

## Manual Continuity Evaluation

The cutoff-free filter was selected on the fixed 1,993-case real single-channel
review set. It contains 313 cases with manually selected high-confidence Real
intervals, 816 FP intervals, and 605 pure-FP cases. Native PELT uses penalty
`16`, minimum size `64`, jump `8`, and standardized active mean `0.05`.

| Method | High-confidence Real recall | FP interval response | Pure-FP case retention | Median IoU | Median boundary error |
|---|---:|---:|---:|---:|---:|
| Bidirectional continuity + ridge lock | 304/313 = 97.12% | 68/816 = 8.33% | 61/605 = 10.08% | 0.912 | 42 records |
| Former 96/640 hard-duration gates | 95.53% | 2.70% | 2.64% | 0.901 | 40 records |

On deterministic even/odd review-rank halves, high-confidence recall is
`96.05%/98.14%`; FP-interval response is `6.83%/9.85%`; pure-FP retention is
`8.91%/11.26%`. A tested period-profile-coherence gate removed zero additional
segments at this working point and is intentionally absent from production.

Continuity calculation is `O(P*T)` for `P` period rows and `T` records. PELT
segment filtering is linear in the number of segments and never changes a PELT
boundary. The method remains channel-independent: `p` indexes CWT period rows
inside one physical frequency channel, never neighboring frequency channels.

The production implementation itself is replayed, rather than a copied
formula, by:

```bash
python tests/perf/cprf_manual_review/compare_activity.py \
  --production-continuity \
  --assert-production-target \
  --output-dir tests/perf/cprf_manual_review/artifacts/production_continuity_validation_v1
```

The command fails unless high-confidence recall remains at least `304/313`,
both deterministic halves remain at least 96%, FP interval hits remain at most
`68`, pure-FP retained cases remain at most `61`, and median IoU remains at
least `0.91`.

Authoritative reproducible output:

```text
tests/perf/cprf_manual_review/artifacts/production_continuity_validation_v1
```
