"""Public API for the standalone FRCR detector core."""

from .core import (
    FRCRChannelResult,
    FRCRParameters,
    frequency_halo_slice,
    frcr_channel,
    reference_channel_indices,
)

__all__ = [
    "FRCRChannelResult",
    "FRCRParameters",
    "frequency_halo_slice",
    "frcr_channel",
    "reference_channel_indices",
]
