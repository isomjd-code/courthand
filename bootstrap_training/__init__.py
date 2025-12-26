"""
Bootstrap training workflow for Pylaia model using Gemini 3 as teacher.

This module implements a complete workflow where:
- Gemini 2.5 Flash detects image rotation
- Kraken + Pylaia segment and transcribe images
- Gemini 3 corrects transcriptions using database index data
- Pylaia model is retrained every 1,000 corrected lines
"""

from .workflow import BootstrapTrainingManager

__all__ = ["BootstrapTrainingManager"]

