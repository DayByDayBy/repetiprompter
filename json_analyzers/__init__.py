"""
JSON Analyzers package for LLM behavioral analysis.

Provides analyzers for:
- semantic_drift: Embedding-based drift detection
- structural_stats: Surface-level structural metrics
- language_detection: Language ID and entropy
- repetition_detection: Multi-layer repetition detection
- quality_signals: Response quality metrics
- convergence_tracker: Trajectory stability analysis
"""

from . import core

__version__ = "0.2.0"
