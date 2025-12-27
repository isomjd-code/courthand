"""Configuration for the report generator."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = "master_record.json" 
OUTPUT_LATEX_PATH = "comparison_report.tex"
DEFAULT_API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
# Note: We don't raise an error here at import time because the environment variable
# might be set later (e.g., via .env file loaded by python-dotenv).
# Validation happens at runtime in report.py main() function.
EMBED_MODEL = "gemini-embedding-001"

SIMILARITY_THRESHOLD = 0.78
ALIGNMENT_THRESHOLD = 0.65
PARTY_MATCH_THRESHOLD = 0.6

