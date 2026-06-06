# Candidate Veto Rules

Veto rules are deterministic filters applied after candidate generation.

The rules and review context are defined in `cwipss.analysis.veto`.

Current flags:

- `broadband`: candidate spans too much of the scanned channel/frequency range.
- `time_edge`: optional; disabled by default because time-aggregated CWT
  candidates usually span the selected time range.
- `freq_edge`: candidate touches the scanned channel/frequency boundary.
- `fixed_channel`: optional; disabled by default because single-channel period
  candidates are a valid target in this project.
- `burst_train`: candidate is short in time and broad in frequency.

Vetoed candidates can still be validated when `include_vetoed` is enabled.
