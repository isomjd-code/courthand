"""
Workflow manager package for CP40 plea roll transcription and extraction.

This package provides the core workflow system for processing medieval legal documents
through a multi-stage pipeline:

Pipeline Sequence:
1. Kraken (line segmentation)
2. PyLaia (HTR recognition)
3. Post-correction and named entity extraction (Gemini 2.5 Flash + Bayesian)
4. Stitching (merge transcriptions from multiple images)
5. Expansion (expand abbreviations)
6. Translation (Latin to English)
7. Indexing (structured entity extraction)

Main components:
- workflow: WorkflowManager class orchestrating the entire pipeline
- image_grouper: ImageGrouper class for organizing images by case
- post_correction: LLM post-correction and Bayesian named entity correction
- prompt_builder: Functions for constructing AI prompts
- schemas: JSON schema definitions for structured extraction
- paleography: PaleographyMatcher for fuzzy matching medieval text
- utils: Helper functions for normalization and data cleaning
"""

from .workflow import WorkflowManager
from .image_grouper import ImageGrouper

__all__ = ["WorkflowManager", "ImageGrouper"]

