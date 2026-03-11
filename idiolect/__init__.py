"""
idiolect — Cross-lingual author attribution.

Find the same voice across different languages.
"""

from .scorer import compare, compare_with_ci, AttributionResult
from .features import extract_features, StyleFeatures

__version__ = "0.1.0"
__all__ = ["compare", "AttributionResult", "extract_features", "StyleFeatures"]
