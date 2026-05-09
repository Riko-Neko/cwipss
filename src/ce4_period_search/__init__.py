"""CWT period-candidate search pipeline with a bundled CE-4 application adapter."""

from .config import CWTSearchConfig
from .pipeline import run_cwt_search

__all__ = ["CWTSearchConfig", "run_cwt_search"]
