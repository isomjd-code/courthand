"""
Ground truth extraction utilities for CP40 plea roll data.

This package provides tools for extracting structured case data from the CP40
database to use as ground truth for validating transcription and extraction results.
It queries the database for cases matching specific roll and rotulus numbers,
extracting comprehensive case information including parties, events, locations,
and legal details.

Main components:
- extractor: Database querying and JSON export functionality
- query: SQL query definitions for extracting case data
- config: Configuration constants and paths
"""

from .extractor import extract_case_data, main

__all__ = ["extract_case_data", "main"]

