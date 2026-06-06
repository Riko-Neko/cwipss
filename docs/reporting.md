# Reporting

`scripts/run_report.py` generates a Markdown review report for a single run,
batch run, or injection benchmark.

Report generation is implemented in `cwipss.reporting.report`; shared plots,
staged visualization, and candidate galleries remain in the same
`cwipss.reporting` package.

The report includes:

- run or batch summary;
- veto distribution;
- top CWT candidates by `peak_score`;
- top validation evidence rows;
- injection benchmark recovery tables when present;
- a link to `visualization/index.md` when staged figures were generated.

Reports are review artifacts. They do not claim a confirmed periodic signal.

Per-candidate galleries are generated separately with
`scripts/run_candidate_gallery.py`. Their entry point is
`candidate_gallery/index.md`; the report generator does not currently embed
all gallery images.
