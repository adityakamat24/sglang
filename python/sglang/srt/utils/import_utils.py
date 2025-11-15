"""
Import utilities for checking availability of optional dependencies.

This module provides cached functions to check if optional packages are installed,
allowing graceful degradation when packages are not available.
"""

from functools import cache


@cache
def has_arctic_inference() -> bool:
    """Check if Arctic Inference package is available.

    Arctic Inference is required for suffix decoding speculative method.
    Install via: pip install arctic-inference==0.1.1

    Returns:
        bool: True if arctic_inference can be imported, False otherwise.
    """
    try:
        import arctic_inference

        return True
    except ImportError:
        return False
