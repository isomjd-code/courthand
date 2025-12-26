"""Configuration constants for ground truth extraction."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
"""Base directory of the project (parent of ground_truth package)."""

DB_PATH = os.getenv("CP40_DB_PATH", str(BASE_DIR / "cp40_database_new.sqlite"))
"""
Path to the CP40 SQLite database file.

Can be overridden via the CP40_DB_PATH environment variable.
Defaults to 'cp40_database_new.sqlite' in the project root.
"""

TARGET_ROLL = os.getenv("CP40_TARGET_ROLL", "562")
"""
Default roll number to search for when extracting ground truth data.

Can be overridden via the CP40_TARGET_ROLL environment variable.
Defaults to "562".
"""

TARGET_ROTULUS = os.getenv("CP40_TARGET_ROTULUS", "340")
"""
Default rotulus number to search for when extracting ground truth data.

Can be overridden via the CP40_TARGET_ROTULUS environment variable.
Defaults to "340".
"""

OUTPUT_DIR = os.getenv("CP40_GT_OUTPUT", "ground_truth_output")
"""
Directory where extracted ground truth JSON files will be saved.

Can be overridden via the CP40_GT_OUTPUT environment variable.
Defaults to "ground_truth_output" (relative to current working directory).
"""

