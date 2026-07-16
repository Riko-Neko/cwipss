# Package Architecture

CWIPSS uses a small domain-oriented package layout. The structure separates
numerical signal processing from workflow orchestration and review products
without introducing framework-specific abstractions.

```text
src/cwipss/
  signal/
    cpro.py              CPU CPRO definition and absolute calibration
    cpro_cuda.py         CUDA CPRO definition; no host array transfer
    cprf.py              frozen CPU CPRF period-family filter
    cprf_cuda.py         device-resident CPRF with scalar-only host output
    cwt.py               CPU CWT and period-grid utilities
    cwt_cuda.py          CUDA CWT backend
    detection.py         CPU CPRO -> PELT -> CPRF orchestration
    detection_cuda.py    asynchronous GPU -> native PELT -> GPU bridge
    windows.py           required native C++ PELT bridge and window selection
  data/
    readers.py           reader protocol and instrument adapters
    schemas.py           CSV schemas and row normalization
  workflows/
    search.py            one-file end-to-end search
    batch.py             multi-file execution and merged outputs
  analysis/
    veto.py              deterministic candidate vetoes
    validation.py        original-series validation
    statistics.py        BH correction and evidence ranking
    simulation.py        synthetic signal definitions
    injection.py         background construction and signal injection
    injection_config.py  configuration-driven injection sampling
    benchmark.py         injection recovery benchmark
  reporting/
    report.py            Markdown reports
    plotting.py          shared plotting primitives
    visualization.py     staged run visualizations
    gallery.py           ranked per-candidate image galleries
  config.py              shared resolved configuration
  runtime.py             runtime metadata
```

The optional frequency-referenced detector is isolated under
`packages/frcr/`. Nothing in `src/cwipss` imports that package. The only
production scientific path is CPRO activity, native PELT windows, then CPRF.

## Dependency Direction

```text
scripts
  -> workflows / analysis / reporting

workflows
  -> signal / data / analysis
  -> reporting only for optional visualization

analysis
  -> signal / data
  -> reporting only for optional benchmark visualization

reporting
  -> signal / data

signal and data
  -> NumPy / SciPy / PyWavelets / CuPy / native extension
```

Lower-level packages do not import workflow modules. Instrument-specific input
logic belongs in `data/readers.py`; numerical candidate logic belongs in
`signal/`; scientific checks after candidate generation belong in `analysis/`.

The stable top-level API remains:

```python
from cwipss import CWTSearchConfig, run_cwt_search
```

Internal module paths are organized by responsibility and may be imported
directly by development scripts and tests.
