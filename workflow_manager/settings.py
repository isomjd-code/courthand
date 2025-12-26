"""Shared configuration and logging helpers for the workflow manager."""

from __future__ import annotations

import logging
import os
import sys
import random

# --- USER CONFIGURATION ---
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)
HOME_DIR = os.path.expanduser("~")
WORK_DIR = os.path.join(BASE_DIR, "cp40_processing")
IMAGE_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SURNAME_DB_PATH = os.path.join(BASE_DIR, "cp40_database_new.sqlite")
PLACE_DB_PATH = os.path.join(BASE_DIR, "places_data.db")

# Ensure normalized absolute paths
WORK_DIR = os.path.abspath(WORK_DIR)
IMAGE_DIR = os.path.abspath(IMAGE_DIR)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
LOG_DIR = os.path.abspath(LOG_DIR)

# VENV PATHS
PYLAIA_ENV = os.path.join(HOME_DIR, "projects/pylaia-env/bin/activate")
KRAKEN_ENV = os.path.join(HOME_DIR, "projects/kraken/kraken/bin/activate")

# MODEL PATHS
MODEL_DIR = os.path.join(BASE_DIR, "model_v10")
PYLAIA_MODEL = os.path.join(MODEL_DIR, "epoch=322-lowest_va_cer.ckpt") #CER of 25.9%
PYLAIA_SYMS = os.path.join(MODEL_DIR, "syms.txt")
PYLAIA_ARCH = os.path.join(MODEL_DIR, "model")

# LLM CONFIGURATION
# Gemini 3 Flash Preview API key - set via environment variable GEMINI_API_KEY (paid key required)
GEMINI_API_KEY = "AIzaSyBmFe4P5cV1L7L5EmjLVC32AQiTQHmgJ7A"
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable must be set with a paid API key")

MODEL_VISION = "gemini-3-flash-preview"  # Gemini 3 Flash Preview for vision tasks (post-correction)
MODEL_TEXT = "gemini-3-flash-preview"  # Gemini 3 Flash Preview for text tasks
THINKING_BUDGET = -1  # Not used for Gemini 3 Flash Preview

# API TIMEOUT CONFIGURATION (in milliseconds)
# Note: Google GenAI client appears to use milliseconds, not seconds
API_TIMEOUT = 1800000  # 30 minutes timeout (30 * 60 * 1000 ms) for API calls and file uploads
API_MAX_RETRIES = 5  # Number of retries for 504/timeout errors (increased for large files)
API_RETRY_DELAY = 15  # Initial delay in seconds between retries (will use exponential backoff)

# Primary Google API key for Gemini (paid key required)
GOOGLE_API_KEY = GEMINI_API_KEY

# COST CONFIGURATION (per million tokens)
# Prices are in USD per million tokens
# Gemini 3 Flash pricing: $0.25 per million input tokens, $1.50 per million output tokens (including thinking)

# All steps use Gemini 3 Flash Preview
COST_INPUT_TOKENS = 0.25  # $0.25 per million input tokens (prompt + cached)
COST_OUTPUT_TOKENS = 1.50  # $1.50 per million output tokens (response + thinking)


def _configure_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("workflow_manager")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers when re-imported
    if logger.handlers:
        return logger

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "debug.log"), mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = _configure_logger()

