"""Configuration for the report generator."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = "master_record.json" 
OUTPUT_LATEX_PATH = "comparison_report.tex"
DEFAULT_API_KEY = "AIzaSyBmFe4P5cV1L7L5EmjLVC32AQiTQHmgJ7A"
EMBED_MODEL = "gemini-embedding-001"

SIMILARITY_THRESHOLD = 0.78
ALIGNMENT_THRESHOLD = 0.65
PARTY_MATCH_THRESHOLD = 0.6

