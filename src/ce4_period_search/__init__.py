"""SWT period-candidate search pipeline with a bundled CE-4 application adapter."""

from .config import SWTScanConfig
from .pipeline import run_swt_scan

__all__ = ["SWTScanConfig", "run_swt_scan"]
