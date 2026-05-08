# SWT Candidate Veto Rules

The veto layer is an auditable pre-validation screen. It marks candidates that
look like common artifacts, but it does not delete raw candidates and it does
not validate periodicity.

## Status Values

| Status | Meaning |
| --- | --- |
| `needs_validation` | No configured veto rule fired. The candidate still requires original time-series validation. |
| `vetoed` | One or more configured veto rules fired. The candidate should not be promoted without manual review. |

## Current Rules

| Flag | Trigger | Default |
| --- | --- | --- |
| `broadband` | Candidate channel span divided by scanned channel span is at least `max_bandwidth_fraction`. | `0.75` |
| `time_edge` | Candidate touches the scanned record boundary, or lies within `edge_time_records`. | `0` records |
| `freq_edge` | Candidate touches the scanned frequency boundary, or lies within `edge_freq_mhz`. | `0.0` MHz |
| `fixed_channel` | Candidate is narrow in frequency but long in time. | bandwidth fraction `<= 0.01`, duration fraction `>= 0.25` |
| `burst_train` | Candidate is short in time and broad in frequency. | duration fraction `<= 0.02`, bandwidth fraction `>= 0.25` |

## Configuration

Veto settings live under the `veto` section:

```json
{
  "veto": {
    "enabled": true,
    "edge_time_records": 0,
    "edge_freq_mhz": 0.0,
    "max_bandwidth_fraction": 0.75,
    "max_fixed_channel_bandwidth_fraction": 0.01,
    "min_fixed_channel_duration_fraction": 0.25,
    "max_burst_duration_fraction": 0.02,
    "min_burst_bandwidth_fraction": 0.25
  }
}
```

Every fired rule writes both a short flag and a JSON details object containing
the measured metric and threshold. This makes later threshold tuning traceable.
