"""
convert_to_quant - Quantization toolkit for neural network weights.

Supports FP8 and INT8 block-wise quantization formats
with learned rounding optimization for minimal accuracy loss.
"""

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("convert_to_quant")
except Exception:
    __version__ = "0.0.0.dev"  # Fallback for development/uninstalled mode

from .convert_to_quant import (  # pyrefly: ignore - intentional re-exports
    LearnedRoundingConverter,
    convert_to_fp8_scaled,
    main,
)

__all__ = [
    "__version__",
    "LearnedRoundingConverter",
    "convert_to_fp8_scaled",
    "main",
]
