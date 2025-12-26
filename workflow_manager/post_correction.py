"""
Post-correction module for HTR text using LLM and Bayesian named entity correction.

This module handles:
1. LLM-based diplomatic transcription correction using Gemini 3 Flash Preview (batch API mode)
2. Named entity extraction (forenames, surnames, place names)
3. Bayesian correction of named entities using Pylaia CTC loss and database priors

PERFORMANCE OPTIMIZATIONS:
- CTC loss function instance is reused (not recreated for each call)
- Levenshtein distance has early exit for very different lengths
- Candidate finding has early termination when enough exact matches found
- Model inference is done once per line image and cached
- Image preprocessing only logged if >100ms
- Database queries timed and logged
- All major operations have timing logs for bottleneck identification

=============================================================================
BAYESIAN FRAMEWORK
=============================================================================

We seek: argmax_c P(candidate | image, context)

Using Bayes' rule:
  P(candidate | image) ∝ P(image | candidate) × P(candidate)

In log space (for numerical stability):
  log P(candidate | image) = log P(image | candidate) + log P(candidate) + const

Where:
  - log P(image | candidate) ≈ -CTC_loss(candidate) / len(candidate)
    The CTC loss is the negative log-likelihood. We normalize by character count
    to make scores comparable across different-length candidates.
    
  - log P(candidate) = log((freq(candidate) + α) / (total_freq + α × V))
    Laplace-smoothed prior from database frequency.
    α = smoothing parameter (default 1.0)
    V = vocabulary size (number of unique names)

Final score:
  score = λ_likelihood × normalized_log_likelihood + λ_prior × log_prior

Where λ_likelihood and λ_prior are tunable weights (default 1.0 each).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from google import genai
from google.genai import types

from .settings import BASE_DIR, logger, MODEL_VISION, WORK_DIR, GEMINI_API_KEY, API_TIMEOUT

# Detect GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} (CUDA available)")
else:
    logger.info("No GPU detected, using CPU")

# Try to import laia - may not be available in all environments
try:
    from laia.utils import SymbolsTable
    LAIA_AVAILABLE = True
except ImportError:
    LAIA_AVAILABLE = False
    logger.warning("laia not available - Bayesian correction will use frequency-only mode")

# Try to import rapidfuzz for entity position finding
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# Try to import symspellpy for fast candidate finding
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
    logger.warning("symspellpy not available - candidate finding will use slower brute-force method")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Gemini client will use paid API key from GEMINI_API_KEY environment variable

# Database path for Bayesian correction
DB_PATH = Path(BASE_DIR) / "cp40_records.db"

# Pylaia models directory
PYLAIA_MODELS_DIR = Path(BASE_DIR) / "bootstrap_training_data" / "pylaia_models"
TOP_CANDIDATES = 20

@dataclass
class BayesianConfig:
    """Configuration for Bayesian candidate selection."""
    # Weights for combining likelihood and prior
    likelihood_weight: float = 1.0  # Weight for log P(image|candidate)
    prior_weight: float = 1.0       # Weight for log P(candidate)
    
    # Laplace smoothing parameter for prior
    smoothing_alpha: float = 1.0
    
    # Bonus for original extraction (in log space, added to score)
    original_bonus: float = 0.5
    
    # Penalty per edit distance from original (in log space)
    distance_penalty: float = 1.5
    
    # Penalty for declension mismatch (in log space)
    declension_mismatch_penalty: float = 0.5
    
    # Minimum similarity for fuzzy position matching
    fuzzy_match_threshold: float = 80.0
    
    # Maximum edit distance for candidate generation
    max_edit_distance: int = 2
    
    # Length ratio bounds for candidate filtering
    min_length_ratio: float = 0.7
    max_length_ratio: float = 1.4


# =============================================================================
# GEMINI CLIENT MANAGEMENT
# =============================================================================

_current_api_key: Optional[str] = None
_current_client: Optional[Any] = None


def reset_gemini_flash_client() -> None:
    """Reset the cached Gemini client to force trying new keys on next call."""
    global _current_api_key, _current_client
    _current_api_key = None
    _current_client = None


def get_gemini_flash_client() -> Any:
    """
    Get a working Gemini 3 Flash Preview client using paid API key.
    Caches the working client for subsequent calls.
    """
    global _current_api_key, _current_client
    
    if _current_client is not None:
        return _current_client
    
    # Use paid key from environment variable
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable must be set with a paid API key")
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # Test the client with a minimal request
        response = client.models.generate_content(
            model=MODEL_VISION,  # Use MODEL_VISION constant for consistency
            contents=[types.Part.from_text(text="test")],
            config=types.GenerateContentConfig(max_output_tokens=1)
        )
        # If we get here, the key works
        logger.info(f"  ✓ Using paid API key for Gemini 3 Flash Preview")
        _current_api_key = GEMINI_API_KEY
        _current_client = client
        return client
    except Exception as e:
        logger.error(f"  ✗ Paid key failed: {str(e)[:50]}...")
        raise RuntimeError(f"Failed to initialize Gemini 3 Flash Preview client: {e}")


# =============================================================================
# LLM POST-CORRECTION
# =============================================================================

def build_post_correction_prompt(lines_data: List[Dict], has_line_images: bool = False) -> str:
    """
    Build the prompt for post-correction and named entity extraction.
    Includes character restrictions: only A-Z, a-z, pilcrow (¶), and straight apostrophe (').
    
    Args:
        lines_data: List of line dictionaries with key, htr_text, and optionally bbox
        has_line_images: If True, indicates that individual line images are provided
    """
    lines_json = json.dumps(lines_data, indent=2, ensure_ascii=False)
    
    line_images_note = ""
    if has_line_images:
        line_images_note = """
## Important Context:
**LINE IMAGES PROVIDED**: Individual line images have been extracted and preprocessed. Each line image corresponds to one line of text. The line images are provided in the SAME ORDER as the line data below. The first image corresponds to L01, the second image to L02, and so on.

**TRANSCRIBE FROM LINE IMAGES**: You should transcribe the text visible in each individual line image provided. The line images have been preprocessed and extracted from the original image, so focus on what you see in each line image.

**BOUNDING BOXES ARE FOR DOCUMENT LAYOUT CONTEXT**: The bounding box coordinates [ymin, xmin, ymax, xmax] provided for each line are from the ORIGINAL full-page image. These coordinates are provided to give you a sense of the document layout and help with semantic analysis (understanding context, relationships between lines, etc.). The coordinates do NOT apply to the individual line images you are viewing - they refer to positions in the original full-page image.

**IMAGE-TO-LABEL CORRESPONDENCE**: The line images are provided in order: first image = L01, second image = L02, third image = L03, etc. Match each image to its corresponding line key.

**ONE LINE PER IMAGE**: Each line image corresponds to exactly one line of text - transcribe only what is visible in that specific line image.

**EMPTY LINES**: If a line image contains no text (e.g., blank space, margin, or decorative element), return an empty string ("") for that line key in your JSON response.

"""
    
    prompt = f"""You are an expert tasked with correcting an HTR diplomatic transcription of text in an image of a CP40 case.
You will use your knowledge of Latin and CP40 idioms as well as placement of lines on the image to correct the HTR text
without moving words between lines. You may also correct named entities (forenames, surnames, place names) when you can clearly see OCR/HTR errors.

In addition, on each line you will identify any named entities (forenames, surnames, place names).
{line_images_note}
## Input Lines:
{lines_json}

## Tasks for EACH line:

1. **Diplomatic Correction**: Correct clear OCR/HTR errors in the latin text, including named entities.
   - PRESERVE all scribal abbreviations exactly (e.g., keep "Rob'tus", NOT "Robertus")
   - PRESERVE apostrophes marking abbreviations (use straight apostrophe ')
   - DO NOT modernize or standardize spellings
   - DO NOT move words between lines
   - **YOU MAY CORRECT NAMED ENTITIES**: If you can clearly see that the HTR misread a name or place name in the image, correct it. For example, if HTR says "Joh'es" but you can see "Joh'is" in the image, correct it. If HTR says "London" but you can see "Lond'on" in the image, correct it.
   - When correcting named entities, preserve the original medieval spelling and abbreviations (e.g., keep "Joh'es" not "Johannes", keep "Lond'on" not "Londonium")
   - **CHARACTER RESTRICTION**: Use ONLY the following characters:
     * Letters: A-Z, a-z
     * Pilcrow for line breaks: ¶
     * Straight apostrophe for abbreviations: '
     * DO NOT use any other characters or punctuation marks

2. **Extract Entities** (MANDATORY - extract from the CORRECTED text):
   **THIS IS REQUIRED FOR EVERY LINE**: You MUST identify and extract ALL named entities in each line. The fields "forenames", "surnames", and "placenames" are MANDATORY and MUST be populated if any names are present in the text.
   
   - **Forenames**: Latin abbreviated names (e.g., "Joh'es", "Ric'us", "Will's", "Will'o", "Joh'anne", "Thomas"). Include declension if recognizable.
     * Look for personal given names in Latin or abbreviated form
     * Common patterns: names ending in 'es, 'us, 'i, 'is, 'o, 'anne, or full names like "Thomas"
     * Examples: "Ric'us", "Will'o", "Joh'anne", "Joh'es", "Thomas"
   - **Surnames**: Family names (e.g., "Clanyng", "Ogle", "Kyngesor", "Samfor'", "Barbor", "Hether", "Chitterne")
     * Look for family names that appear after forenames or with "de" (of)
     * Examples: "Barbor", "Hether", "Chitterne", "Tombrell'", "Barbour", "Chapman"
   - **Placenames**: Place names (e.g., "London'", "Holborn'", "Nolbenton'", "Marlebergh", "Faryndon'", "Wiltes'")
     * Look for place names, especially those following "de" (of/from) or appearing as locations
     * Examples: "Marlebergh", "London'", "Faryndon'", "Wiltes'"
   
   **CRITICAL RULES**:
   - Extract entities from the CORRECTED text (not the original HTR text)
   - If a line contains ANY forenames, surnames, or placenames, the corresponding arrays MUST NOT be empty
   - Empty arrays are ONLY acceptable when NO names of that type are present in the line
   - You MUST scan every word in the corrected text to identify all named entities
   - Missing entity extraction is a CRITICAL ERROR - always populate these fields when names are present

## Output Format (JSON array):
[
  {{
    "key": "L01",
    "corrected_text": "...",
    "forenames": [{{"text": "Joh'es", "declension": "genitive"}}],
    "surnames": [{{"text": "Clanyng"}}],
    "placenames": [{{"text": "London'"}}]
  }}
]

## Examples:

**Example 1:**
If the HTR text is: "Thomas Kyngesor de Nolbenton' fuit ad respondend' Will's Samfor'"
And you can see in the image that it should be "Thomas Kyngesor de Nolbenton' fuit ad respondend' Will'o Samfor'"
Then the output should be:
[
  {{
    "key": "L01",
    "corrected_text": "Thomas Kyngesor de Nolbenton' fuit ad respondend' Will'o Samfor'",
    "forenames": [{{"text": "Thomas"}}, {{"text": "Will'o"}}],
    "surnames": [{{"text": "Kyngesor"}}, {{"text": "Samfor'"}}],
    "placenames": [{{"text": "Nolbenton'"}}]
  }}
]

**Example 2:**
If the HTR text is: "Ric'us Barbor de Marlebergh attach' fuit ad respondend' Will'o Hether & Joh'anne ux'i eius"
And the image shows the text is correct, then the output should be:
[
  {{
    "key": "L01",
    "corrected_text": "Ric'us Barbor de Marlebergh attach' fuit ad respondend' Will'o Hether & Joh'anne ux'i eius",
    "forenames": [{{"text": "Ric'us"}}, {{"text": "Will'o"}}, {{"text": "Joh'anne"}}],
    "surnames": [{{"text": "Barbor"}}, {{"text": "Hether"}}],
    "placenames": [{{"text": "Marlebergh"}}]
  }}
]

**Example 3 (with named entity correction):**
If the HTR text is: "Joh'es Clanyng de Lond'on"
But you can see in the image that it should be "Joh'is Clanyng de Lond'on" (the HTR misread 'is as 'es)
Then the output should be:
[
  {{
    "key": "L01",
    "corrected_text": "Joh'is Clanyng de Lond'on",
    "forenames": [{{"text": "Joh'is"}}],
    "surnames": [{{"text": "Clanyng"}}],
    "placenames": [{{"text": "Lond'on"}}]
  }}
]

**CRITICAL REQUIREMENTS**:
1. The fields "forenames", "surnames", and "placenames" are MANDATORY in the JSON schema
2. You MUST extract ALL named entities present in each line
3. If a line contains names, the corresponding arrays MUST NOT be empty - this is a validation requirement
4. Empty arrays are ONLY acceptable when NO names of that type exist in the line
5. Return ONLY valid JSON, no other text."""

    return prompt


def sanitize_custom_id(image_name: str) -> str:
    """
    Sanitize image_name to create a valid custom_id matching pattern: ^[a-zA-Z0-9_-]{1,64}$
    
    Args:
        image_name: Original image name (may contain invalid characters)
    
    Returns:
        Sanitized custom_id with prefix "post_correction_"
    """
    # Replace invalid characters with underscores, keep only alphanumeric, underscore, hyphen
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', image_name)
    # Truncate to ensure total length is <= 64 (prefix "post_correction_" is 17 chars)
    max_name_len = 64 - 17  # Reserve space for "post_correction_" prefix
    if len(sanitized_name) > max_name_len:
        sanitized_name = sanitized_name[:max_name_len]
    return f"post_correction_{sanitized_name}"


def process_single_image_post_correction(
    image_path: str,
    lines: List[Dict[str, Any]],
    image_name: str,
    client: Any,
    out_dir: Optional[str] = None
) -> List[Dict]:
    """
    Process a single image through post-correction using Gemini 3 Flash Preview (non-batch mode).
    
    Args:
        image_path: Path to the image file
        lines: List of line dicts with htr_text, bbox, line_id
        image_name: Name of the image
        client: Gemini API client
        out_dir: Optional output directory containing line images (e.g., htr_work/{basename}/lines/)
    
    Returns:
        List of corrected line dictionaries with entities extracted
    """
    # Prepare lines data for prompt
    lines_data = []
    for idx, line in enumerate(lines, 1):
        entry = {
            "key": f"L{idx:02d}",
            "htr_text": line.get("htr_text", ""),
        }
        if line.get("bbox"):
            entry["bbox"] = line["bbox"]
        lines_data.append(entry)
    
    # Load individual line images if out_dir is provided
    line_image_data = []
    line_image_paths = []
    has_line_images = False
    
    if out_dir:
        basename = os.path.splitext(os.path.basename(image_name))[0]
        work_dir = os.path.join(out_dir, basename)
        lines_dir = os.path.join(work_dir, "lines")
        
        if os.path.exists(lines_dir):
            for idx, line in enumerate(lines, 1):
                line_id = line.get("line_id")
                
                if not line_id:
                    logger.debug(f"Skipping line L{idx:02d}: missing line_id")
                    line_image_paths.append(None)
                    line_image_data.append(None)
                    continue
                
                # Find existing line image
                line_image_path = os.path.join(lines_dir, f"{line_id}.png")
                
                if not os.path.exists(line_image_path):
                    # Try other extensions
                    found = False
                    for ext in ['.jpg', '.jpeg']:
                        alt_path = os.path.join(lines_dir, f"{line_id}{ext}")
                        if os.path.exists(alt_path):
                            line_image_path = alt_path
                            found = True
                            break
                    
                    if not found:
                        logger.debug(f"Line image not found for L{idx:02d} (line_id: {line_id}): {line_image_path}")
                        line_image_paths.append(None)
                        line_image_data.append(None)
                        continue
                
                try:
                    # Read existing line image
                    with open(line_image_path, 'rb') as f:
                        image_data = f.read()
                    
                    line_image_paths.append(line_image_path)
                    line_image_data.append(image_data)
                    has_line_images = True
                    logger.debug(f"Loaded line image for L{idx:02d} (line_id: {line_id})")
                    
                except Exception as e:
                    logger.warning(f"Error loading line image for L{idx:02d} (line_id: {line_id}): {e}")
                    line_image_paths.append(None)
                    line_image_data.append(None)
    
    # Build prompt (indicate if line images are provided)
    prompt_text = build_post_correction_prompt(lines_data, has_line_images=has_line_images)
    
    # Prepare content parts for Gemini API
    parts = [types.Part.from_text(text=prompt_text)]
    
    # Add individual line images if available
    if has_line_images:
        valid_line_count = 0
        for line_idx, (line_img_data, line_img_path) in enumerate(zip(line_image_data, line_image_paths)):
            if line_img_data is not None and line_img_path is not None:
                # Validate image size (Gemini has limits)
                if len(line_img_data) > 20 * 1024 * 1024:  # 20MB limit
                    logger.warning(f"Line image {line_idx+1} for {image_name} is too large ({len(line_img_data) / 1024 / 1024:.2f}MB), skipping")
                    continue
                
                # Add line image part
                line_image_part = types.Part.from_bytes(data=line_img_data, mime_type="image/png")
                parts.append(line_image_part)
                valid_line_count += 1
        
        if valid_line_count > 0:
            logger.debug(f"Added {valid_line_count} line images for {image_name}")
    
    # Add full page image if it exists and no line images were found
    if not has_line_images and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Determine MIME type
            if image_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            else:
                mime_type = 'image/jpeg'  # Default
            
            parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
    
    # Call Gemini API
    logger.info(f"  Running Gemini 3 Flash Preview post-correction for {image_name} ({len(lines_data)} lines)...")
    
    # Log a sample of the prompt for debugging
    logger.debug(f"  Prompt preview (first 500 chars): {prompt_text[:500]}...")
    
    # Define response schema to ensure proper structure
    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "corrected_text": {"type": "string"},
                "forenames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "declension": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                },
                "surnames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                },
                "placenames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                }
            },
            "required": ["key", "corrected_text", "forenames", "surnames", "placenames"]
        }
    }
    
    try:
        # Configure API call parameters
        config_params = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": 0.0,
            "max_output_tokens": 8192,
        }
        
        # Use medium resolution for line images (similar to bootstrap_training)
        if has_line_images:
            config_params["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
        
        # Try with thinking_config first, fall back without it if not supported
        try:
            config_params["thinking_config"] = types.ThinkingConfig(include_thoughts=True, thinking_level="LOW")
            config = types.GenerateContentConfig(**config_params)
            response = client.models.generate_content(
                model=MODEL_VISION,
                contents=parts,
                config=config
            )
        except Exception as e:
            # If thinking_config is not supported, retry without it
            if "INVALID_ARGUMENT" in str(e) or (hasattr(e, 'status_code') and e.status_code == 400):
                logger.debug(f"Thinking config not supported, retrying without it for {image_name}")
                config_params.pop("thinking_config", None)
                config = types.GenerateContentConfig(**config_params)
                response = client.models.generate_content(
                    model=MODEL_VISION,
                    contents=parts,
                    config=config
                )
            else:
                raise
        
        # Extract text from response
        # When using response_mime_type="application/json", the response structure may differ
        response_text = ""
        parsed_data = None
        
        # Try direct text attribute first
        if hasattr(response, 'text') and response.text:
            response_text = response.text
        # Try candidates structure
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            # Check for structured output (when using response_mime_type="application/json")
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    # For structured JSON output, the part might have text directly
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
                    # Or it might be in a structured format
                    elif hasattr(part, 'inline_data'):
                        # Binary JSON data
                        try:
                            parsed_data = json.loads(part.inline_data.data)
                        except (AttributeError, json.JSONDecodeError):
                            pass
        
        # Log raw response for debugging (first 2000 chars)
        if response_text:
            logger.debug(f"  Raw response text (first 2000 chars) for {image_name}:\n{response_text[:2000]}")
        elif parsed_data:
            logger.debug(f"  Parsed response data structure for {image_name}: {type(parsed_data)}")
        
        # If we got parsed_data directly, use it
        if parsed_data:
            parsed_list = parsed_data if isinstance(parsed_data, list) else [parsed_data]
        # Otherwise, try to parse from text
        elif response_text.strip():
            from .utils import clean_json_string
            cleaned_json_str = clean_json_string(response_text)
            if cleaned_json_str:
                try:
                    parsed_data = json.loads(cleaned_json_str)
                    parsed_list = parsed_data if isinstance(parsed_data, list) else [parsed_data]
                except json.JSONDecodeError as e:
                    # Handle "Extra data" errors by extracting just the first valid JSON
                    parsed_list = None  # Initialize in case all parsing attempts fail
                    error_msg = str(e)
                    import re
                    
                    if "Extra data" in error_msg:
                        # Extract the character position where extra data starts
                        char_match = re.search(r'char (\d+)', error_msg)
                        if char_match:
                            try:
                                error_pos = int(char_match.group(1))
                                # Try to parse just up to the error position
                                partial_json = cleaned_json_str[:error_pos]
                                
                                # Find the last complete JSON array by finding matching brackets
                                bracket_count = 0
                                last_valid_pos = -1
                                for i in range(len(partial_json) - 1, -1, -1):
                                    if partial_json[i] == ']':
                                        bracket_count += 1
                                    elif partial_json[i] == '[':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            last_valid_pos = i
                                            break
                                
                                if last_valid_pos != -1:
                                    # Find the matching closing bracket
                                    bracket_count = 0
                                    for i in range(last_valid_pos, len(partial_json)):
                                        if partial_json[i] == '[':
                                            bracket_count += 1
                                        elif partial_json[i] == ']':
                                            bracket_count -= 1
                                            if bracket_count == 0:
                                                try:
                                                    parsed_data = json.loads(partial_json[last_valid_pos:i+1])
                                                    parsed_list = parsed_data if isinstance(parsed_data, list) else [parsed_data]
                                                    logger.info(f"  Extracted valid JSON from response (ignored extra data after char {error_pos})")
                                                    break
                                                except json.JSONDecodeError:
                                                    pass
                                
                                # If bracket matching didn't work, try simple rfind
                                if not parsed_list:
                                    last_bracket = partial_json.rfind(']')
                                    if last_bracket != -1:
                                        # Find the matching opening bracket
                                        bracket_count = 0
                                        for i in range(last_bracket, -1, -1):
                                            if partial_json[i] == ']':
                                                bracket_count += 1
                                            elif partial_json[i] == '[':
                                                bracket_count -= 1
                                                if bracket_count == 0:
                                                    try:
                                                        parsed_data = json.loads(partial_json[i:last_bracket+1])
                                                        parsed_list = parsed_data if isinstance(parsed_data, list) else [parsed_data]
                                                        logger.info(f"  Extracted valid JSON using bracket matching (ignored extra data after char {error_pos})")
                                                        break
                                                    except json.JSONDecodeError:
                                                        pass
                            except (ValueError, json.JSONDecodeError) as parse_err:
                                logger.debug(f"Error parsing at position: {parse_err}")
                                parsed_list = None
                        
                        # If position-based extraction failed, try regex
                        if not parsed_list:
                            json_match = re.search(r'\[[\s\S]*?\]', cleaned_json_str, re.DOTALL)
                            if json_match:
                                try:
                                    parsed_data = json.loads(json_match.group(0))
                                    parsed_list = parsed_data if isinstance(parsed_data, list) else [parsed_data]
                                    logger.info(f"  Extracted JSON array using regex (ignored extra data)")
                                except json.JSONDecodeError:
                                    parsed_list = None
                    else:
                        # Other JSON errors - try regex extraction
                        json_match = re.search(r'\[[\s\S]*?\]', cleaned_json_str, re.DOTALL)
                        if json_match:
                            try:
                                parsed_data = json.loads(json_match.group(0))
                                parsed_list = parsed_data if isinstance(parsed_data, list) else [parsed_data]
                                logger.info(f"  Extracted JSON array using regex after error: {e}")
                            except json.JSONDecodeError:
                                parsed_list = None
                    
                    if not parsed_list:
                        logger.warning(f"Failed to parse JSON for {image_name}: {e}")
                        logger.debug(f"Response text (first 1000 chars): {response_text[:1000]}")
            else:
                logger.warning(f"clean_json_string returned empty for {image_name}")
                logger.debug(f"Raw response text (first 1000 chars): {response_text[:1000]}")
                parsed_list = None
        else:
            # Log the response structure for debugging
            logger.warning(f"No text found in response for {image_name}")
            logger.debug(f"Response structure: {type(response)}, has text: {hasattr(response, 'text')}, "
                        f"has candidates: {hasattr(response, 'candidates')}")
            if hasattr(response, 'candidates') and response.candidates:
                logger.debug(f"Candidate structure: {type(response.candidates[0])}, "
                           f"has content: {hasattr(response.candidates[0], 'content')}")
            parsed_list = None
        
        # If we successfully parsed, return it
        if parsed_list:
            total_forenames = sum(len(l.get('forenames', [])) for l in parsed_list)
            total_surnames = sum(len(l.get('surnames', [])) for l in parsed_list)
            total_placenames = sum(len(l.get('placenames', [])) for l in parsed_list)
            
            logger.info(
                f"  Parsed Gemini JSON for {image_name}: "
                f"lines={len(parsed_list)}, "
                f"forenames={total_forenames}, "
                f"surnames={total_surnames}, "
                f"placenames={total_placenames}"
            )
            
            # Log warning if no entities were extracted but text contains potential names
            if total_forenames == 0 and total_surnames == 0 and total_placenames == 0:
                # Check if any line has text that looks like it might contain names
                has_potential_names = False
                for line in parsed_list:
                    text = line.get('corrected_text', '') or line.get('htr_text', '')
                    # Simple heuristic: check for common name patterns
                    if any(word in text.lower() for word in ['thomas', 'will', 'john', 'richard', 'robert', 'william']):
                        has_potential_names = True
                        break
                    # Check for Latin name patterns
                    if re.search(r"\b[A-Z][a-z]+'?[es|us|i|is]\b", text):
                        has_potential_names = True
                        break
                
                if has_potential_names:
                    logger.warning(
                        f"  ⚠️ No entities extracted for {image_name} but text appears to contain names. "
                        f"Response may not be following entity extraction instructions."
                    )
                    # Log a sample of the response for debugging
                    logger.debug(f"  Sample response text (first 500 chars): {response_text[:500] if response_text else 'No response text'}")
            
            return parsed_list
        
        # Fallback: return uncorrected text with empty extractions
        logger.warning(f"No valid JSON found in response for {image_name}. Using fallback.")
        return [
            {
                "key": line["key"],
                "corrected_text": line.get("htr_text", ""),
                "forenames": [],
                "surnames": [],
                "placenames": []
            }
            for line in lines_data
        ]
        
    except Exception as e:
        logger.error(f"Error calling Gemini API for {image_name}: {e}")
        # Fallback: return uncorrected text with empty extractions
        return [
            {
                "key": line["key"],
                "corrected_text": line.get("htr_text", ""),
                "forenames": [],
                "surnames": [],
                "placenames": []
            }
            for line in lines_data
        ]


def run_non_batch_post_correction(
    image_lines_map: Dict[str, Dict[str, Any]],
    batch_id: str,
    client: Any,
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process post-correction for multiple images using Gemini 3 Flash Preview (non-batch mode).
    Processes each image individually.
    
    Args:
        image_lines_map: Dictionary mapping image_path -> {
            'lines': list of line dicts,
            'image_name': name of the image
        }
        batch_id: Unique identifier for this batch (for logging)
        client: Gemini API client
        out_dir: Optional output directory containing line images (e.g., htr_work/{basename}/lines/)
    
    Returns:
        Dictionary mapping image names to their results (List[Dict] per image)
    """
    if not image_lines_map:
        return {}
    
    results_map = {}
    
    logger.info(f"[{batch_id}] Processing {len(image_lines_map)} images with Gemini 3 Flash Preview (non-batch mode)...")
    
    for image_path, data in image_lines_map.items():
        lines = data['lines']
        image_name = data['image_name']
        
        # Process single image
        result = process_single_image_post_correction(
            image_path=image_path,
            lines=lines,
            image_name=image_name,
            client=client,
            out_dir=out_dir
        )
        
        # Store result using sanitized image name as key
        custom_id = sanitize_custom_id(image_name)
        results_map[custom_id] = result
        
        # Small delay between requests to avoid rate limiting
        time.sleep(0.5)
    
    logger.info(f"[{batch_id}] Completed processing {len(results_map)} images")
    
    return results_map


def run_batch_post_correction(
    image_lines_map: Dict[str, Dict[str, Any]],
    batch_id: str,
    client: Any,
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process post-correction for multiple images using Gemini 3 Flash Preview batch API.
    Processes all images in a single batch job.
    
    Args:
        image_lines_map: Dictionary mapping image_path -> {
            'lines': list of line dicts,
            'image_name': name of the image
        }
        batch_id: Unique identifier for this batch (for logging)
        client: Gemini API client
        out_dir: Optional output directory containing line images (e.g., htr_work/{basename}/lines/)
    
    Returns:
        Dictionary mapping image names to their results (List[Dict] per image)
    """
    if not image_lines_map:
        return {}
    
    import tempfile
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    logger.info(f"[{batch_id}] Processing {len(image_lines_map)} images with Gemini 3 Flash Preview batch API...")
    
    # Create temporary directory for batch files
    temp_dir = Path(tempfile.mkdtemp())
    jsonl_file = temp_dir / "batch_requests.jsonl"
    
    try:
        # Prepare batch requests in JSONL format
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for idx, (image_path, data) in enumerate(image_lines_map.items()):
                lines = data['lines']
                image_name = data['image_name']
                key = sanitize_custom_id(image_name)
                
                # Prepare lines data for prompt
                lines_data = []
                for line_idx, line in enumerate(lines, 1):
                    entry = {
                        "key": f"L{line_idx:02d}",
                        "htr_text": line.get("htr_text", ""),
                    }
                    if line.get("bbox"):
                        entry["bbox"] = line["bbox"]
                    lines_data.append(entry)
                
                # Load individual line images if out_dir is provided
                line_image_paths = []
                has_line_images = False
                
                if out_dir:
                    basename = os.path.splitext(os.path.basename(image_name))[0]
                    work_dir = os.path.join(out_dir, basename)
                    lines_dir = os.path.join(work_dir, "lines")
                    
                    if os.path.exists(lines_dir):
                        logger.info(f"[{batch_id}] Looking for line images in {lines_dir} for {key}...")
                        found_count = 0
                        
                        # Parallelize line image finding
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        
                        def find_line_image(line_idx: int, line: Dict) -> Tuple[int, Optional[str]]:
                            """Find line image path for a single line. Returns (index, path or None)."""
                            line_id = line.get("line_id")
                            
                            if not line_id:
                                return (line_idx - 1, None)  # Convert to 0-based index
                            
                            # Try .png first
                            line_image_path = os.path.join(lines_dir, f"{line_id}.png")
                            
                            if not os.path.exists(line_image_path):
                                # Try other extensions
                                for ext in ['.jpg', '.jpeg']:
                                    alt_path = os.path.join(lines_dir, f"{line_id}{ext}")
                                    if os.path.exists(alt_path):
                                        return (line_idx - 1, alt_path)
                                return (line_idx - 1, None)
                            
                            return (line_idx - 1, line_image_path)
                        
                        # Find all line images in parallel
                        max_workers = min(20, len(lines))  # Limit concurrent file checks
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_idx = {
                                executor.submit(find_line_image, line_idx, line): line_idx
                                for line_idx, line in enumerate(lines, 1)
                            }
                            
                            # Collect results in order
                            results = {}
                            for future in as_completed(future_to_idx):
                                idx, path = future.result()
                                results[idx] = path
                                if path is not None:
                                    found_count += 1
                            
                            # Build line_image_paths in order
                            line_image_paths = [results.get(idx) for idx in range(len(lines))]
                            has_line_images = found_count > 0
                        
                        if found_count > 0:
                            logger.info(f"[{batch_id}] Found {found_count} line images for {key} (out of {len(lines)} lines)")
                        else:
                            logger.warning(f"[{batch_id}] No line images found for {key} in {lines_dir}")
                    else:
                        logger.warning(f"[{batch_id}] Line images directory does not exist for {key}: {lines_dir}")
                
                # Build prompt
                prompt_text = build_post_correction_prompt(lines_data, has_line_images=has_line_images)
                
                # Upload line images if available
                parts = [{"text": prompt_text}]
                uploaded_line_images = []
                
                if has_line_images:
                    # Helper function to upload a single image
                    def upload_single_image(line_idx: int, line_img_path: str) -> Optional[Tuple[int, Any]]:
                        """Upload a single line image and return (index, uploaded_image) or None if failed."""
                        if not line_img_path or not os.path.exists(line_img_path):
                            return None
                        
                        try:
                            # Check file size
                            file_size = os.path.getsize(line_img_path)
                            if file_size > 20 * 1024 * 1024:  # 20MB limit
                                logger.warning(f"[{batch_id}] Line image {line_idx+1} for {key} is too large ({file_size / 1024 / 1024:.2f}MB), skipping")
                                return None
                            
                            # Upload line image file
                            logger.debug(f"[{batch_id}] Uploading line image {line_idx+1} for {key}: {os.path.basename(line_img_path)}")
                            uploaded_image = client.files.upload(
                                file=line_img_path,
                                config=types.UploadFileConfig(mime_type="image/png")
                            )
                            
                            # Wait for file to be active
                            max_wait = 60
                            wait_count = 0
                            while uploaded_image.state.name != "ACTIVE" and wait_count < max_wait:
                                time.sleep(1)
                                uploaded_image = client.files.get(name=uploaded_image.name)
                                wait_count += 1
                            
                            if uploaded_image.state.name != "ACTIVE":
                                logger.warning(f"[{batch_id}] Line image {line_idx+1} for {key} not active after upload")
                                return None
                            
                            logger.debug(f"[{batch_id}] Line image {line_idx+1} for {key} uploaded and active")
                            return (line_idx, uploaded_image)
                        except Exception as e:
                            logger.warning(f"[{batch_id}] Error uploading line image {line_idx+1} for {key}: {e}")
                            return None
                    
                    # Upload all line images in parallel
                    upload_tasks = [
                        (line_idx, line_img_path)
                        for line_idx, line_img_path in enumerate(line_image_paths)
                        if line_img_path is not None
                    ]
                    
                    if upload_tasks:
                        logger.info(f"[{batch_id}] Uploading {len(upload_tasks)} line images for {key}...")
                        max_workers = min(10, len(upload_tasks))
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_task = {
                                executor.submit(upload_single_image, idx, path): (idx, path)
                                for idx, path in upload_tasks
                            }
                            
                            upload_results = {}
                            completed = 0
                            for future in as_completed(future_to_task):
                                result = future.result()
                                if result is not None:
                                    line_idx, uploaded_image = result
                                    upload_results[line_idx] = uploaded_image
                                    completed += 1
                                    if completed % 5 == 0 or completed == len(upload_tasks):
                                        logger.debug(f"[{batch_id}] Uploaded {completed}/{len(upload_tasks)} line images for {key}")
                        
                        # Build parts in original order
                        for line_idx in sorted(upload_results.keys()):
                            uploaded_image = upload_results[line_idx]
                            file_uri = uploaded_image.uri if hasattr(uploaded_image, 'uri') and uploaded_image.uri else uploaded_image.name
                            parts.append({
                                "file_data": {
                                    "mime_type": "image/png",
                                    "file_uri": file_uri
                                }
                            })
                            uploaded_line_images.append(uploaded_image)
                        
                        logger.info(f"[{batch_id}] Successfully uploaded {len(upload_results)} line images for {key} (added to batch request)")
                    else:
                        logger.warning(f"[{batch_id}] No line images to upload for {key}")
                
                if len(parts) == 1:  # Only prompt, no images
                    logger.warning(f"[{batch_id}] No valid line images for {key}, skipping")
                    continue
                
                # Build generation config
                gen_config = {
                    "temperature": 0.0,
                    "max_output_tokens": 32768,
                    "response_mime_type": "application/json",
                    "media_resolution": "MEDIA_RESOLUTION_MEDIUM" if has_line_images else None,
                    "thinking_config": {
                        "include_thoughts": True,
                        "thinking_level": "LOW"
                    }
                }
                
                # Remove None values
                gen_config = {k: v for k, v in gen_config.items() if v is not None}
                
                # Build request object
                num_line_images = len(parts) - 1  # Subtract 1 for the prompt text
                request_obj = {
                    "key": key,
                    "request": {
                        "contents": [{"parts": parts}],
                        "generation_config": gen_config
                    }
                }
                
                f.write(json.dumps(request_obj) + "\n")
                logger.info(f"[{batch_id}] Added batch request for {key} with {num_line_images} line images")
        
        # Check if we have any requests
        if jsonl_file.stat().st_size == 0:
            logger.warning(f"[{batch_id}] No valid requests in batch file")
            return {"results": {}, "token_usage": None}
        
        # Check for cached batch ID first
        def get_batch_cache_path(batch_id: str, output_dir: str = None) -> str:
            """Get the path to the batch ID cache file."""
            safe_id = re.sub(r'[^a-zA-Z0-9]', '_', batch_id)
            cache_dir = output_dir if output_dir else WORK_DIR
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, f"batch_cache_{safe_id}.json")
        
        def get_cached_batch_id(batch_id: str, output_dir: str = None, force: bool = False) -> Optional[str]:
            """Retrieve a cached batch job ID if it exists."""
            cache_path = get_batch_cache_path(batch_id, output_dir)
            if os.path.exists(cache_path) and not force:
                try:
                    with open(cache_path, 'r') as f:
                        cache_data = json.load(f)
                        job_name = cache_data.get('job_name')
                        if job_name:
                            # Verify the batch job still exists and is not completed
                            try:
                                batch_job_check = client.batches.get(name=job_name)
                                completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
                                def get_state_name(batch_job):
                                    if hasattr(batch_job.state, 'name'):
                                        return batch_job.state.name
                                    elif isinstance(batch_job.state, str):
                                        return batch_job.state
                                    else:
                                        return str(batch_job.state)
                                current_state = get_state_name(batch_job_check)
                                if current_state not in completed_states:
                                    logger.info(f"[{batch_id}] Found cached batch ID: {job_name}, resuming polling")
                                    return job_name
                                elif current_state == 'JOB_STATE_SUCCEEDED':
                                    logger.info(f"[{batch_id}] Cached batch {job_name} is already completed ({current_state}), retrieving and processing results")
                                    return job_name
                                else:
                                    logger.info(f"[{batch_id}] Cached batch {job_name} is in completed state ({current_state}), creating new batch")
                            except Exception as e:
                                logger.warning(f"[{batch_id}] Cached batch {job_name} no longer exists: {e}")
                    return None
                except Exception as e:
                    logger.warning(f"[{batch_id}] Failed to read batch cache: {e}")
            return None
        
        def save_batch_id(batch_id: str, job_name: str, output_dir: str = None) -> None:
            """Save a batch job ID to cache."""
            cache_path = get_batch_cache_path(batch_id, output_dir)
            try:
                with open(cache_path, 'w') as f:
                    json.dump({
                        "job_name": job_name,
                        "batch_id": batch_id,
                        "created_at": time.time()
                    }, f, indent=2)
                logger.debug(f"[{batch_id}] Saved batch ID to cache: {job_name}")
            except Exception as e:
                logger.warning(f"[{batch_id}] Failed to save batch cache: {e}")
        
        # Try to resume from cache
        cached_job_name = get_cached_batch_id(batch_id, out_dir, force=False)
        batch_job = None
        
        if cached_job_name:
            try:
                batch_job = client.batches.get(name=cached_job_name)
                logger.info(f"[{batch_id}] Resuming cached batch: {cached_job_name}")
            except Exception as e:
                logger.warning(f"[{batch_id}] Failed to resume cached batch, creating new: {e}")
                batch_job = None
        
        # Upload and create batch if not resuming
        if not batch_job:
            # Upload JSONL file
            logger.info(f"[{batch_id}] Uploading batch file...")
            uploaded_file = client.files.upload(
                file=str(jsonl_file),
                config=types.UploadFileConfig(mime_type='application/jsonl')
            )
            
            # Create batch job
            logger.info(f"[{batch_id}] Creating batch job...")
            batch_job = client.batches.create(
                model=MODEL_VISION,
                src=uploaded_file.name,
                config={"display_name": f"post_correction_{batch_id}"}
            )
            # Save batch ID to cache
            save_batch_id(batch_id, batch_job.name, out_dir)
        
        # Poll for completion
        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
        def get_state_name(batch_job):
            if hasattr(batch_job.state, 'name'):
                return batch_job.state.name
            elif isinstance(batch_job.state, str):
                return batch_job.state
            else:
                return str(batch_job.state)
        
        current_state = get_state_name(batch_job)
        poll_count = 0
        max_polls = 1200  # 10 hours max
        
        while current_state not in completed_states:
            time.sleep(30)
            poll_count += 1
            batch_job = client.batches.get(name=batch_job.name)
            current_state = get_state_name(batch_job)
            logger.info(f"[{batch_id}] Batch status: {current_state} (poll {poll_count}/{max_polls})")
            
            if poll_count >= max_polls:
                logger.error(f"[{batch_id}] Batch job timed out")
                break
        
        if current_state != 'JOB_STATE_SUCCEEDED':
            logger.error(f"[{batch_id}] Batch job failed with state: {current_state}")
            return {"results": {}, "token_usage": None}
        
        # Download results
        logger.info(f"[{batch_id}] Downloading batch results...")
        results_file = temp_dir / "batch_results.jsonl"
        
        if hasattr(batch_job, 'dest') and batch_job.dest and hasattr(batch_job.dest, 'file_name'):
            result_file_name = batch_job.dest.file_name
            file_content_bytes = client.files.download(file=result_file_name)
            
            with open(results_file, 'wb') as f:
                if isinstance(file_content_bytes, bytes):
                    f.write(file_content_bytes)
                else:
                    for chunk in file_content_bytes:
                        f.write(chunk)
        else:
            logger.error(f"[{batch_id}] No result file found")
            return {"results": {}, "token_usage": None}
        
        # Parse results and collect token usage
        results_map = {}
        total_token_usage = {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "response_tokens": 0,
            "thoughts_tokens": 0,
            "total_tokens": 0,
            "non_cached_input": 0
        }
        
        def safe_int(value):
            """Convert value to int, defaulting to 0 if None or invalid."""
            if value is None:
                return 0
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    result_obj = json.loads(line)
                    result_key = result_obj.get("key", "unknown")
                    
                    # Extract token usage from result object
                    um = None
                    if 'response' in result_obj:
                        response_obj = result_obj.get("response", {})
                        if 'usageMetadata' in response_obj:
                            um = response_obj['usageMetadata']
                        elif 'usage_metadata' in response_obj:
                            um = response_obj['usage_metadata']
                    elif 'usageMetadata' in result_obj:
                        um = result_obj['usageMetadata']
                    elif 'usage_metadata' in result_obj:
                        um = result_obj['usage_metadata']
                    
                    if um:
                        # Aggregate token usage
                        total_token_usage["prompt_tokens"] += safe_int(um.get('promptTokenCount') or um.get('prompt_token_count', 0))
                        total_token_usage["cached_tokens"] += safe_int(um.get('cachedContentTokenCount') or um.get('cached_content_token_count', 0))
                        total_token_usage["response_tokens"] += safe_int(um.get('candidatesTokenCount') or um.get('candidates_token_count', 0))
                        total_token_usage["thoughts_tokens"] += safe_int(um.get('thoughtsTokenCount') or um.get('thoughts_token_count', 0))
                        total_token_usage["total_tokens"] += safe_int(um.get('totalTokenCount') or um.get('total_token_count', 0))
                    
                    if result_obj.get("status") == "SUCCEEDED" or "response" in result_obj:
                        response_obj = result_obj.get("response", {})
                        
                        # Extract response text
                        response_text = ""
                        if isinstance(response_obj, dict) and "candidates" in response_obj and response_obj["candidates"]:
                            candidate = response_obj["candidates"][0]
                            if isinstance(candidate, dict) and "content" in candidate:
                                content = candidate["content"]
                                if isinstance(content, dict) and "parts" in content and content["parts"]:
                                    all_parts_text = []
                                    for part in content["parts"]:
                                        if isinstance(part, dict) and "text" in part:
                                            all_parts_text.append(part["text"])
                                    
                                    if all_parts_text:
                                        response_text = all_parts_text[-1] if len(all_parts_text) > 1 else all_parts_text[0]
                        
                        # Parse JSON
                        parsed_list = []
                        if response_text.strip():
                            try:
                                parsed_json = json.loads(response_text.strip())
                                parsed_list = parsed_json if isinstance(parsed_json, list) else [parsed_json]
                            except json.JSONDecodeError:
                                # Try to extract JSON from text
                                from .utils import clean_json_string
                                cleaned_json_str = clean_json_string(response_text)
                                if cleaned_json_str:
                                    try:
                                        parsed_json = json.loads(cleaned_json_str)
                                        parsed_list = parsed_json if isinstance(parsed_json, list) else [parsed_json]
                                    except json.JSONDecodeError:
                                        pass
                        
                        results_map[result_key] = parsed_list if parsed_list else []
                    else:
                        results_map[result_key] = []
                except Exception as e:
                    logger.error(f"[{batch_id}] Error parsing result: {e}")
        
        # Calculate non_cached_input
        total_token_usage["non_cached_input"] = total_token_usage["prompt_tokens"] - total_token_usage["cached_tokens"]
        
        logger.info(f"[{batch_id}] Batch complete: {len(results_map)} results, token usage: {total_token_usage}")
        
        # Return both results and token usage
        return {
            "results": results_map,
            "token_usage": total_token_usage if total_token_usage["total_tokens"] > 0 else None
        }
        
    except Exception as e:
        logger.error(f"[{batch_id}] Error in batch API processing: {e}", exc_info=True)
        return {"results": {}, "token_usage": None}
    finally:
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


def call_llm_for_post_correction(
    lines_data: List[Dict],
    image_path: Optional[str] = None,
    retry_on_error: bool = True,
    out_dir: Optional[str] = None
) -> List[Dict]:
    """
    Call Gemini 3 Flash Preview to correct HTR text and extract entities (single image mode).
    
    Args:
        lines_data: List of line dictionaries with 'key' and 'htr_text'
        image_path: Optional path to image file for vision processing
        retry_on_error: Whether to retry on error (currently unused)
        out_dir: Optional output directory containing line images (e.g., htr_work/{basename}/lines/)
    
    Returns:
        List of corrected line dictionaries with entities extracted
    """
    client = get_gemini_flash_client()
    
    # Prepare lines for processing
    lines = [
        {
            "htr_text": line.get("htr_text", ""),
            "bbox": line.get("bbox"),
            "line_id": line.get("key", f"L{i+1:02d}")
        }
        for i, line in enumerate(lines_data)
    ]
    
    # Use the single image processing function
    result = process_single_image_post_correction(
        image_path=image_path or "",
        lines=lines,
        image_name="single_image",
        client=client,
        out_dir=out_dir
    )
    
    return result


# =============================================================================
# PYLAIA MODEL FUNCTIONS
# =============================================================================

def find_latest_pylaia_model() -> Tuple[Path, Path, Path]:
    """Find the latest Pylaia model from pylaia_models directory."""
    logger.debug(f"Searching for Pylaia models in: {PYLAIA_MODELS_DIR}")
    
    if not PYLAIA_MODELS_DIR.exists():
        logger.error(f"Pylaia models directory does not exist: {PYLAIA_MODELS_DIR}")
        raise FileNotFoundError(f"Pylaia models directory not found: {PYLAIA_MODELS_DIR}")
    
    # Find all model directories
    model_dirs = []
    all_items = list(PYLAIA_MODELS_DIR.iterdir())
    logger.debug(f"  Found {len(all_items)} items in directory")
    
    for item in all_items:
        if item.is_dir() and item.name.startswith("model_v"):
            try:
                version = int(item.name.replace("model_v", ""))
                model_dirs.append((version, item))
                logger.debug(f"  Found model directory: {item.name} (version {version})")
            except ValueError:
                logger.debug(f"  Skipping non-version directory: {item.name}")
                continue
    
    if not model_dirs:
        logger.error(f"No model directories (model_v*) found in {PYLAIA_MODELS_DIR}")
        logger.error(f"  Available items: {[item.name for item in all_items if item.is_dir()]}")
        raise FileNotFoundError(f"No model directories found in {PYLAIA_MODELS_DIR}")
    
    latest_version, latest_dir = max(model_dirs, key=lambda x: x[0])
    logger.debug(f"  Selected latest model: model_v{latest_version} from {latest_dir}")
    
    # Find checkpoint
    experiment_dir = latest_dir / "experiment"
    checkpoint_patterns = ["*-lowest_va_cer.ckpt", "*-last.ckpt"]
    
    checkpoint = None
    search_dirs = [experiment_dir, latest_dir] if experiment_dir.exists() else [latest_dir]
    logger.debug(f"  Searching for checkpoint in: {[str(d) for d in search_dirs]}")
    
    for search_dir in search_dirs:
        for pattern in checkpoint_patterns:
            files = list(search_dir.glob(pattern))
            if files:
                checkpoint = max(files, key=lambda p: p.stat().st_mtime)
                logger.debug(f"  Found checkpoint: {checkpoint} (matching pattern {pattern})")
                break
        if checkpoint:
            break
    
    if not checkpoint:
        logger.error(f"No checkpoint file found in {latest_dir}")
        logger.error(f"  Searched patterns: {checkpoint_patterns}")
        logger.error(f"  Searched directories: {[str(d) for d in search_dirs]}")
        raise FileNotFoundError(f"No checkpoint found in {latest_dir}")
    
    model_file = latest_dir / "model"
    syms_file = latest_dir / "syms.txt"
    
    logger.debug(f"  Checking model file: {model_file} (exists: {model_file.exists()})")
    logger.debug(f"  Checking symbols file: {syms_file} (exists: {syms_file.exists()})")
    
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not syms_file.exists():
        logger.error(f"Symbols file not found: {syms_file}")
        raise FileNotFoundError(f"Syms file not found: {syms_file}")
    
    logger.info(f"Using Pylaia model: model_v{latest_version}")
    logger.info(f"  Checkpoint: {checkpoint.name}")
    logger.info(f"  Model file: {model_file}")
    logger.info(f"  Symbols file: {syms_file}")
    
    return checkpoint, model_file, syms_file


def load_pylaia_model(checkpoint_path: Path, model_arch_path: Path, symbols_path: Path):
    """Load Pylaia model using laia's ModelLoader."""
    if not LAIA_AVAILABLE:
        logger.error("laia package not available - cannot load model")
        raise RuntimeError("laia package not available")
    
    logger.debug(f"Starting model load process...")
    logger.debug(f"  Checkpoint: {checkpoint_path} (exists: {checkpoint_path.exists()})")
    logger.debug(f"  Model arch: {model_arch_path} (exists: {model_arch_path.exists()})")
    logger.debug(f"  Symbols: {symbols_path} (exists: {symbols_path.exists()})")
    
    start_time = time.time()
    logger.info(f"Loading symbols from: {symbols_path}")
    try:
        syms = SymbolsTable(str(symbols_path))
        symbols_time = time.time() - start_time
        logger.info(f"  Symbols loaded in {symbols_time:.2f}s ({len(syms)} symbols)")
    except Exception as e:
        logger.error(f"Failed to load symbols file: {e}")
        raise
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"  Using device: {device_str}")
    
    try:
        logger.debug("Attempting to load model using laia ModelLoader...")
        from laia.common.loader import ModelLoader
        
        train_path = str(model_arch_path.parent)
        model_filename = model_arch_path.name
        logger.debug(f"  Train path: {train_path}")
        logger.debug(f"  Model filename: {model_filename}")
        
        loader = ModelLoader(train_path, filename=model_filename, device=device_str)
        
        # Try to prepare checkpoint - different laia versions have different signatures
        checkpoint = None
        try:
            # First try with 3 positional arguments using train_path as exp_dirpath
            # Signature: prepare_checkpoint(checkpoint, exp_dirpath, monitor)
            checkpoint = loader.prepare_checkpoint(
                str(checkpoint_path), train_path, None
            )
            logger.debug(f"  Prepared checkpoint (3 positional args with train_path): {checkpoint}")
        except (TypeError, AttributeError) as e1:
            logger.debug(f"  prepare_checkpoint with 3 positional args failed ({type(e1).__name__}), trying with keyword args: {e1}")
            try:
                # Try with keyword arguments (some laia versions support this)
                checkpoint = loader.prepare_checkpoint(
                    str(checkpoint_path), 
                    exp_dirpath=train_path, 
                    monitor=None
                )
                logger.debug(f"  Prepared checkpoint (keyword args): {checkpoint}")
            except (TypeError, AttributeError) as e2:
                logger.debug(f"  prepare_checkpoint with keyword args failed ({type(e2).__name__}), trying with single arg: {e2}")
                try:
                    # Try with just checkpoint path (newer laia versions)
                    checkpoint = loader.prepare_checkpoint(str(checkpoint_path))
                    logger.debug(f"  Prepared checkpoint (single arg): {checkpoint}")
                except (TypeError, AttributeError) as e3:
                    logger.debug(f"  prepare_checkpoint with single arg also failed ({type(e3).__name__}), using checkpoint path directly: {e3}")
                    checkpoint = str(checkpoint_path)
        
        if checkpoint is None:
            checkpoint = str(checkpoint_path)
        
        model_start = time.time()
        model = loader.load_by(checkpoint)
        model_time = time.time() - model_start
        logger.info(f"✓ Model loaded using laia ModelLoader (device: {device_str}) in {model_time:.2f}s")
        
    except Exception as e:
        logger.warning(f"  ModelLoader failed ({type(e).__name__}): {e}")
        logger.debug(f"  ModelLoader exception details:", exc_info=True)
        logger.info("  Trying fallback method (PyTorch Lightning)...")
        
        try:
            import pytorch_lightning as pl
        except ImportError as import_err:
            logger.error(f"  PyTorch Lightning not available: {import_err}")
            raise RuntimeError(f"Both ModelLoader and PyTorch Lightning failed. ModelLoader: {e}, Lightning: {import_err}")
        
        # Use GPU if available for map_location
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"  Loading checkpoint with map_location={map_location}")
        model_start = time.time()
        try:
            model = pl.LightningModule.load_from_checkpoint(
                str(checkpoint_path), map_location=map_location, strict=False
            )
            model_time = time.time() - model_start
            logger.info(f"✓ Model loaded using PyTorch Lightning (device: {map_location}) in {model_time:.2f}s")
        except Exception as pl_err:
            logger.error(f"  PyTorch Lightning load failed: {type(pl_err).__name__}: {pl_err}")
            logger.error(f"  Exception details:", exc_info=True)
            raise RuntimeError(f"Both ModelLoader and PyTorch Lightning failed. ModelLoader: {e}, Lightning: {pl_err}")
    
    logger.debug("Setting model to eval mode...")
    model.eval()
    
    # Move model to device if not already there
    if torch.cuda.is_available():
        logger.debug(f"Moving model to device: {DEVICE}")
        model = model.to(DEVICE)
        logger.debug(f"Model moved to {DEVICE}")
    
    # Extract actual model from Lightning wrapper
    for attr in ['model', 'net', 'crnn']:
        if hasattr(model, attr):
            actual_model = getattr(model, attr)
            actual_model.eval()
            return actual_model, syms
    
    return model, syms


def preprocess_line_image(image_path: Path) -> torch.Tensor:
    """Preprocess line image for Pylaia model (grayscale, resize to 128px height)."""
    start_time = time.time()
    img = Image.open(image_path).convert('L')
    
    width, height = img.size
    new_height = 128
    new_width = int(width * (new_height / height))
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    # Move tensor to the same device as the model
    result = tensor.to(DEVICE)
    elapsed = time.time() - start_time
    if elapsed > 0.1:  # Only log if it takes significant time
        logger.debug(f"  Image preprocessing took {elapsed:.3f}s for {image_path.name}")
    return result


def text_to_indices(text: str, syms) -> Optional[List[int]]:
    """
    Convert text to symbol indices for CTC loss computation.
    
    Input text format: Normal text like "Ioh'es Groos" (spaces only between words).
    This function converts each character to its symbol index, mapping spaces to <space> indices.
    
    Note: Pylaia training data uses space-separated tokens in files (e.g., "I o h ' e s <space> G r o o s"),
    but CTC loss works with sequences of indices, not strings. So we convert cleaned text
    directly to indices without adding spaces between characters.
    
    Args:
        text: Cleaned text (spaces only between words, not between every character)
        syms: Symbols table mapping characters to indices
        
    Returns:
        List of symbol indices, or None if any character cannot be mapped
    """
    indices = []
    
    for c in text:
        idx = None
        
        # Try character as-is
        try:
            idx = syms[c]
        except (KeyError, TypeError):
            pass
        
        # Try space variations
        if idx is None and c == ' ':
            try:
                idx = syms['<space>']
            except (KeyError, TypeError):
                pass
        
        # Try case variations
        if idx is None:
            for variant in [c.lower(), c.upper()]:
                try:
                    idx = syms[variant]
                    if idx is not None:
                        break
                except (KeyError, TypeError):
                    pass
        
        # Try apostrophe variations
        if idx is None and c in "'`'":
            for variant in ["'", "'", "`"]:
                try:
                    idx = syms[variant]
                    if idx is not None:
                        break
                except (KeyError, TypeError):
                    pass
        
        if idx is None:
            return None  # Cannot encode this character
        
        indices.append(idx)
    
    return indices if indices else None


def get_model_log_probs(model, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run model forward pass and return log probabilities.
    This should be called ONCE per image, then reused for all candidates.
    """
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Ensure (Time, Batch, Classes) format
    if output.dim() == 3 and output.size(0) == image_tensor.size(0):
        output = output.transpose(0, 1)
    
    result = F.log_softmax(output, dim=2)
    elapsed = time.time() - start_time
    logger.debug(f"  Model inference took {elapsed:.3f}s")
    return result


def calculate_ctc_loss_from_logprobs(
    log_probs: torch.Tensor, 
    syms, 
    text: str
) -> float:
    """
    Calculate CTC loss using pre-computed log probabilities.
    
    Args:
        log_probs: (T, N, C) tensor of log probabilities from model forward pass
        syms: Symbols table mapping characters to indices
        text: Cleaned text (spaces only between words, e.g., "Ioh'es Groos")
              This is converted to indices internally via text_to_indices().
    
    Returns:
        Raw CTC loss (negative log-likelihood), or float('inf') if impossible.
    """
    start_time = time.time()
    indices = text_to_indices(text, syms)
    if indices is None:
        return float('inf')
    
    # Check feasibility (CTC requires T >= S)
    if log_probs.size(0) < len(indices):
        return float('inf')
    
    # Move tensors to the same device as log_probs
    target = torch.tensor(indices, dtype=torch.long, device=log_probs.device)
    input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long, device=log_probs.device)
    target_lengths = torch.tensor([len(indices)], dtype=torch.long, device=log_probs.device)
    
    # Reuse CTCLoss instance if possible (create once, reuse many times)
    if not hasattr(calculate_ctc_loss_from_logprobs, '_ctc_loss_fn'):
        calculate_ctc_loss_from_logprobs._ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=False)
    
    ctc_loss_fn = calculate_ctc_loss_from_logprobs._ctc_loss_fn
    
    try:
        loss = ctc_loss_fn(log_probs, target, input_lengths, target_lengths)
        loss_value = loss.item()
        
        if not np.isfinite(loss_value):
            return float('inf')
        
        elapsed = time.time() - start_time
        if elapsed > 0.01:  # Only log if it takes significant time (>10ms)
            logger.debug(f"  CTC loss calculation took {elapsed:.3f}s for text length {len(text)}")
        
        return loss_value
    except RuntimeError:
        return float('inf')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_htr_text(text: str) -> str:
    """
    Clean raw HTR output by removing inter-character spaces
    and replacing <space> tokens with actual spaces.
    
    This is critical for CTC loss computation - the model expects
    text without the inter-character spaces that PyLaia outputs.
    """
    if not text:
        return ""
    text = text.replace("<space>", "|||SPACE|||")
    text = text.replace(" ", "")
    text = text.replace("|||SPACE|||", " ")
    return " ".join(text.split())


def find_entity_in_text(entity: str, text: str, threshold: float = 80.0) -> Optional[Tuple[int, int]]:
    """
    Find the position of an entity in text using fuzzy matching.
    Returns (start_idx, end_idx) or None if not found.
    """
    if not entity or not text:
        return None
    
    # Try exact match first
    idx = text.find(entity)
    if idx != -1:
        return (idx, idx + len(entity))
    
    # Try case-insensitive exact match
    lower_text = text.lower()
    lower_entity = entity.lower()
    idx = lower_text.find(lower_entity)
    if idx != -1:
        return (idx, idx + len(entity))
    
    if not RAPIDFUZZ_AVAILABLE:
        return None
    
    # Fuzzy matching: slide a window and find best match
    entity_len = len(entity)
    best_score = 0
    best_pos = None
    
    for i in range(len(text) - entity_len + 1):
        window = text[i:i + entity_len]
        score = fuzz.ratio(entity, window)
        if score > best_score and score >= threshold:
            best_score = score
            best_pos = (i, i + entity_len)
    
    # Also try word-level matching
    words = text.split()
    pos = 0
    for word in words:
        word_start = text.find(word, pos)
        if word_start == -1:
            pos += len(word) + 1
            continue
        
        score = fuzz.ratio(entity, word)
        if score > best_score and score >= threshold:
            best_score = score
            best_pos = (word_start, word_start + len(word))
        pos = word_start + len(word)
    
    return best_pos


# =============================================================================
# DATABASE QUERY FOR BAYESIAN CORRECTION
# =============================================================================

class NameDatabase:
    """Helper class for querying cp40_records.db for name candidates."""
    
    def __init__(self, db_path: Path, config: BayesianConfig):
        self.config = config
        self.db_path = db_path
        self.forenames = {}
        self.forename_latin_forms = defaultdict(list)
        self.surnames = {}
        self.placenames = {}
        self.total_forename_freq = 1
        self.total_surname_freq = 1
        self.total_placename_freq = 1
        
        if not db_path.exists():
            logger.warning(f"Database not found at {db_path}. Bayesian correction disabled.")
            return
            
        try:
            self._load_data(db_path)
            # Create SymSpell dictionaries for fast candidate lookup
            if SYMSPELL_AVAILABLE:
                self._create_symspell_dicts()
            else:
                logger.warning("SymSpell not available - using slower brute-force candidate finding")
        except Exception as e:
            logger.warning(f"Failed to load database: {e}. Bayesian correction disabled.")
    
    def _load_data(self, db_path: Path) -> None:
        """Load all names with frequencies from database."""
        start_time = time.time()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        
        try:
            # Forenames and Latin forms
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='forenames'"
            )
            if cursor.fetchone():
                cursor = conn.execute("SELECT id, english_name, frequency FROM forenames")
                self.forenames = {
                    row['english_name']: {'id': row['id'], 'frequency': row['frequency'] or 0}
                    for row in cursor.fetchall()
                }
                
                cursor = conn.execute("""
                    SELECT fl.latin_abbreviated, fl.case_name, f.english_name, f.frequency
                    FROM forename_latin_forms fl
                    JOIN forenames f ON fl.forename_id = f.id
                """)
                for row in cursor.fetchall():
                    self.forename_latin_forms[row['latin_abbreviated']].append({
                        'english_name': row['english_name'],
                        'case_name': row['case_name'],
                        'frequency': row['frequency'] or 0
                    })
            
            # Surnames
            cursor = conn.execute("""
                SELECT s.surname, COUNT(ps.person_id) as frequency
                FROM surnames s
                LEFT JOIN person_surnames ps ON s.id = ps.surname_id
                GROUP BY s.surname
            """)
            self.surnames = {row['surname']: row['frequency'] or 0 for row in cursor.fetchall()}
            
            # Placenames
            cursor = conn.execute("""
                SELECT p.name, COUNT(ep.entry_id) as frequency
                FROM places p
                LEFT JOIN entry_places ep ON p.id = ep.place_id
                GROUP BY p.name
            """)
            self.placenames = {row['name']: row['frequency'] or 0 for row in cursor.fetchall()}
            
            # Calculate totals
            self.total_forename_freq = sum(info['frequency'] for info in self.forenames.values()) or 1
            self.total_surname_freq = sum(self.surnames.values()) or 1
            self.total_placename_freq = sum(self.placenames.values()) or 1
            
            elapsed = time.time() - start_time
            logger.info(f"Loaded {len(self.forenames)} forenames, {len(self.surnames)} surnames, {len(self.placenames)} placenames in {elapsed:.2f}s")
        finally:
            conn.close()
    
    def _get_cache_path(self) -> Path:
        """Get cache file path for SymSpell dictionaries."""
        # Create cache directory if it doesn't exist
        cache_dir = Path(BASE_DIR) / ".symspell_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Generate cache key from db path, max_edit_distance, and db mtime
        db_mtime = self.db_path.stat().st_mtime if self.db_path.exists() else 0
        cache_key = f"{self.db_path.name}_{self.config.max_edit_distance}_{db_mtime}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        return cache_dir / f"symspell_{cache_hash}.pkl"
    
    def _load_symspell_cache(self, cache_path: Path) -> Optional[Dict[str, SymSpell]]:
        """Load SymSpell dictionaries from cache if valid."""
        if not cache_path.exists():
            return None
        
        try:
            # Check if cache is newer than database
            if self.db_path.exists():
                db_mtime = self.db_path.stat().st_mtime
                cache_mtime = cache_path.stat().st_mtime
                if cache_mtime < db_mtime:
                    logger.debug(f"Cache outdated (DB modified), rebuilding dictionaries")
                    return None
            
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            
            # Validate cache structure
            required_keys = ['forename_symspell', 'surname_symspell', 'placename_symspell']
            if all(key in cached for key in required_keys):
                logger.info(f"Loaded SymSpell dictionaries from cache: {cache_path.name}")
                return cached
            else:
                logger.warning("Cache file has invalid structure, rebuilding")
                return None
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, rebuilding dictionaries")
            return None
    
    def _save_symspell_cache(self, cache_path: Path, dicts: Dict[str, SymSpell]) -> None:
        """Save SymSpell dictionaries to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(dicts, f)
            logger.debug(f"Saved SymSpell dictionaries to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _create_symspell_dicts(self) -> None:
        """Create SymSpell dictionaries for fast fuzzy lookup, using cache if available."""
        if not SYMSPELL_AVAILABLE:
            return
        
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        cached_dicts = self._load_symspell_cache(cache_path)
        if cached_dicts:
            self.forename_symspell = cached_dicts['forename_symspell']
            self.surname_symspell = cached_dicts['surname_symspell']
            self.placename_symspell = cached_dicts['placename_symspell']
            return
        
        # Cache miss or invalid - create dictionaries
        start_time = time.time()
        max_dist = self.config.max_edit_distance
        
        # Forename Latin forms
        self.forename_symspell = SymSpell(max_dictionary_edit_distance=max_dist)
        for latin_form, forms in self.forename_latin_forms.items():
            max_freq = max((f['frequency'] for f in forms), default=1)
            self.forename_symspell.create_dictionary_entry(latin_form, max_freq)
        # Also add English names
        for name, info in self.forenames.items():
            self.forename_symspell.create_dictionary_entry(name, info['frequency'])
        
        # Surnames
        self.surname_symspell = SymSpell(max_dictionary_edit_distance=max_dist)
        for surname, freq in self.surnames.items():
            self.surname_symspell.create_dictionary_entry(surname, freq)
        
        # Placenames
        self.placename_symspell = SymSpell(max_dictionary_edit_distance=max_dist)
        for place, freq in self.placenames.items():
            self.placename_symspell.create_dictionary_entry(place, freq)
        
        elapsed = time.time() - start_time
        logger.info(f"Created SymSpell dictionaries in {elapsed:.2f}s")
        
        # Save to cache
        dicts_to_cache = {
            'forename_symspell': self.forename_symspell,
            'surname_symspell': self.surname_symspell,
            'placename_symspell': self.placename_symspell
        }
        self._save_symspell_cache(cache_path, dicts_to_cache)
    
    def _levenshtein(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings.
        
        Uses rapidfuzz if available (much faster), otherwise falls back to pure Python.
        """
        # Early exit for very different lengths (optimization)
        len_diff = abs(len(s1) - len(s2))
        if len_diff > self.config.max_edit_distance:
            return len_diff
        
        # Use rapidfuzz if available (much faster)
        if RAPIDFUZZ_AVAILABLE:
            from rapidfuzz import distance
            return distance.Levenshtein.distance(s1, s2)
        
        # Fallback to pure Python implementation
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        # Standard dynamic programming algorithm
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dels = curr[j] + 1
                subs = prev[j] + (c1 != c2)
                curr.append(min(ins, dels, subs))
            prev = curr
        return prev[-1]
    
    def _get_effective_distance(self, text_len: int) -> int:
        """Get effective edit distance based on text length."""
        if text_len <= 3:
            return 0
        elif text_len <= 5:
            return 1
        else:
            return self.config.max_edit_distance
    
    def _passes_length_filter(self, original_len: int, candidate_len: int) -> bool:
        """Check if candidate passes length ratio filter."""
        if original_len == 0:
            return False
        ratio = candidate_len / original_len
        return self.config.min_length_ratio <= ratio <= self.config.max_length_ratio
    
    def find_forename_candidates(self, text: str, declension: Optional[str] = None, max_candidates: int = 50) -> List[Dict]:
        """
        Find forename candidates using SymSpell (fast) or brute-force Levenshtein (fallback).
        
        Returns candidates sorted by (distance, -frequency) for initial filtering.
        Note: We return more candidates than we'll score, since CTC loss can change rankings.
        """
        if not self.forenames:
            return []
        
        start_time = time.time()
        candidates = []
        effective_dist = self._get_effective_distance(len(text))
        
        # Exact match in Latin forms
        if text in self.forename_latin_forms:
            for form in self.forename_latin_forms[text]:
                # Only include forms that match the declension (if specified)
                if declension is not None and form['case_name'] != declension:
                    continue
                candidates.append({
                    'text': text,
                    'english_name': form['english_name'],
                    'declension': form['case_name'],
                    'distance': 0,
                    'frequency': form['frequency'],
                    'declension_matches': True
                })
        
        # Use SymSpell if available (much faster)
        if SYMSPELL_AVAILABLE and hasattr(self, 'forename_symspell'):
            # SymSpell lookup on original text
            suggestions = self.forename_symspell.lookup(
                text, verbosity=Verbosity.ALL,
                max_edit_distance=effective_dist, include_unknown=True
            )
            
            for suggestion in suggestions:
                if not self._passes_length_filter(len(text), len(suggestion.term)):
                    continue
                
                # Find matching forms
                if suggestion.term in self.forename_latin_forms:
                    for form in self.forename_latin_forms[suggestion.term]:
                        # Only include forms that match the declension (if specified)
                        if declension is not None and form['case_name'] != declension:
                            continue
                        candidates.append({
                            'text': suggestion.term,
                            'english_name': form['english_name'],
                            'declension': form['case_name'],
                            'distance': suggestion.distance,
                            'frequency': form['frequency'],
                            'declension_matches': True
                        })
                elif suggestion.term in self.forenames:
                    # English name match
                    info = self.forenames[suggestion.term]
                    candidates.append({
                        'text': suggestion.term,
                        'english_name': suggestion.term,
                        'declension': None,
                        'distance': suggestion.distance,
                        'frequency': info['frequency'],
                        'declension_matches': True
                    })
        else:
            # Fallback to brute-force Levenshtein (slower)
            # Parallelize fuzzy matching for better performance
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def check_candidate(latin_form: str, forms: List[Dict]) -> List[Dict]:
                """Check a single candidate and return matches."""
                if not self._passes_length_filter(len(text), len(latin_form)):
                    return []
                
                # Normalize I/J for medieval Latin (they're often interchangeable)
                text_norm = text.lower().replace('j', 'i')
                form_norm = latin_form.lower().replace('j', 'i')
                dist = self._levenshtein(text_norm, form_norm)
                # Also check original distance (I/J might be correct, don't penalize)
                dist_original = self._levenshtein(text.lower(), latin_form.lower())
                dist = min(dist, dist_original)  # Use the better (smaller) distance
                
                if dist <= effective_dist and dist > 0:
                    result = []
                    for form in forms:
                        # Only include forms that match the declension (if specified)
                        if declension is not None and form['case_name'] != declension:
                            continue
                        result.append({
                            'text': latin_form,
                            'english_name': form['english_name'],
                            'declension': form['case_name'],
                            'distance': dist,
                            'frequency': form['frequency'],
                            'declension_matches': True
                        })
                    return result
                return []
            
            # Process candidates in parallel
            max_workers = min(20, len(self.forename_latin_forms))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_form = {
                    executor.submit(check_candidate, latin_form, forms): latin_form
                    for latin_form, forms in self.forename_latin_forms.items()
                }
                
                # Collect results with early termination
                for future in as_completed(future_to_form):
                    matches = future.result()
                    candidates.extend(matches)
                    
                    # Early termination: if we have enough candidates with distance 0, stop
                    exact_count = len([c for c in candidates if c['distance'] == 0])
                    if exact_count >= max_candidates:
                        break
        
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            key = (c['text'], c.get('english_name'), c.get('declension'))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        # Return more candidates - CTC scoring will re-rank them
        result = sorted(unique, key=lambda x: (x['distance'], -x['frequency']))[:max_candidates]
        elapsed = time.time() - start_time
        if elapsed > 0.1:  # Only log if it takes significant time
            logger.debug(f"  find_forename_candidates took {elapsed:.3f}s, found {len(result)} candidates")
        return result
    
    def find_surname_candidates(self, text: str, max_candidates: int = 50) -> List[Dict]:
        """
        Find surname candidates using SymSpell (fast) or brute-force Levenshtein (fallback).
        
        Returns candidates sorted by (distance, -frequency) for initial filtering.
        Note: We return more candidates than we'll score, since CTC loss can change rankings.
        """
        if not self.surnames:
            return []
        
        start_time = time.time()
        candidates = []
        effective_dist = self._get_effective_distance(len(text))
        
        # Exact match
        if text in self.surnames:
            candidates.append({
                'text': text,
                'distance': 0,
                'frequency': self.surnames[text]
            })
        
        # Use SymSpell if available (much faster)
        if SYMSPELL_AVAILABLE and hasattr(self, 'surname_symspell'):
            # SymSpell lookup
            suggestions = self.surname_symspell.lookup(
                text, verbosity=Verbosity.ALL,
                max_edit_distance=effective_dist, include_unknown=True
            )
            
            for suggestion in suggestions:
                if not self._passes_length_filter(len(text), len(suggestion.term)):
                    continue
                if suggestion.term in self.surnames:
                    candidates.append({
                        'text': suggestion.term,
                        'distance': suggestion.distance,
                        'frequency': self.surnames[suggestion.term]
                    })
        else:
            # Fallback to brute-force Levenshtein (slower)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def check_candidate(surname: str, freq: int) -> Optional[Dict]:
                """Check a single surname candidate."""
                if not self._passes_length_filter(len(text), len(surname)):
                    return None
                
                # Normalize I/J for medieval Latin (they're often interchangeable)
                text_norm = text.lower().replace('j', 'i')
                surname_norm = surname.lower().replace('j', 'i')
                dist = self._levenshtein(text_norm, surname_norm)
                # Also check original distance (I/J might be correct, don't penalize)
                dist_original = self._levenshtein(text.lower(), surname.lower())
                dist = min(dist, dist_original)  # Use the better (smaller) distance
                
                if dist <= effective_dist:
                    return {
                        'text': surname,
                        'distance': dist,
                        'frequency': freq
                    }
                return None
            
            # Process candidates in parallel
            max_workers = min(20, len(self.surnames))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_surname = {
                    executor.submit(check_candidate, surname, freq): surname
                    for surname, freq in self.surnames.items()
                }
                
                # Collect results with early termination
                for future in as_completed(future_to_surname):
                    candidate = future.result()
                    if candidate:
                        candidates.append(candidate)
                        
                        # Early termination: if we have enough candidates with distance 0, stop
                        exact_count = len([c for c in candidates if c['distance'] == 0])
                        if exact_count >= max_candidates:
                            break
        
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c['text'] not in seen:
                seen.add(c['text'])
                unique.append(c)
        
        # Return more candidates - CTC scoring will re-rank them
        result = sorted(unique, key=lambda x: (x['distance'], -x['frequency']))[:max_candidates]
        elapsed = time.time() - start_time
        if elapsed > 0.1:  # Only log if it takes significant time
            logger.debug(f"  find_surname_candidates took {elapsed:.3f}s, found {len(result)} candidates")
        return result
    
    def find_placename_candidates(self, text: str, max_candidates: int = 50) -> List[Dict]:
        """
        Find placename candidates using SymSpell (fast) or brute-force Levenshtein (fallback).
        
        Returns candidates sorted by (distance, -frequency) for initial filtering.
        Note: We return more candidates than we'll score, since CTC loss can change rankings.
        """
        if not self.placenames:
            return []
        
        start_time = time.time()
        candidates = []
        effective_dist = self._get_effective_distance(len(text))
        
        # Exact match
        if text in self.placenames:
            candidates.append({
                'text': text,
                'distance': 0,
                'frequency': self.placenames[text]
            })
        
        # Use SymSpell if available (much faster)
        if SYMSPELL_AVAILABLE and hasattr(self, 'placename_symspell'):
            # SymSpell lookup
            suggestions = self.placename_symspell.lookup(
                text, verbosity=Verbosity.ALL,
                max_edit_distance=effective_dist, include_unknown=True
            )
            
            for suggestion in suggestions:
                if not self._passes_length_filter(len(text), len(suggestion.term)):
                    continue
                if suggestion.term in self.placenames:
                    candidates.append({
                        'text': suggestion.term,
                        'distance': suggestion.distance,
                        'frequency': self.placenames[suggestion.term]
                    })
        else:
            # Fallback to brute-force Levenshtein (slower)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def check_candidate(place: str, freq: int) -> Optional[Dict]:
                """Check a single placename candidate."""
                if not self._passes_length_filter(len(text), len(place)):
                    return None
                
                # Normalize I/J for medieval Latin (they're often interchangeable)
                text_norm = text.lower().replace('j', 'i')
                place_norm = place.lower().replace('j', 'i')
                dist = self._levenshtein(text_norm, place_norm)
                # Also check original distance (I/J might be correct, don't penalize)
                dist_original = self._levenshtein(text.lower(), place.lower())
                dist = min(dist, dist_original)  # Use the better (smaller) distance
                
                if dist <= effective_dist:
                    return {
                        'text': place,
                        'distance': dist,
                        'frequency': freq
                    }
                return None
            
            # Process candidates in parallel
            max_workers = min(20, len(self.placenames))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_place = {
                    executor.submit(check_candidate, place, freq): place
                    for place, freq in self.placenames.items()
                }
                
                # Collect results with early termination
                for future in as_completed(future_to_place):
                    candidate = future.result()
                    if candidate:
                        candidates.append(candidate)
                        
                        # Early termination: if we have enough candidates with distance 0, stop
                        exact_count = len([c for c in candidates if c['distance'] == 0])
                        if exact_count >= max_candidates:
                            break
        
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c['text'] not in seen:
                seen.add(c['text'])
                unique.append(c)
        
        # Return more candidates - CTC scoring will re-rank them
        result = sorted(unique, key=lambda x: (x['distance'], -x['frequency']))[:max_candidates]
        elapsed = time.time() - start_time
        if elapsed > 0.1:  # Only log if it takes significant time
            logger.debug(f"  find_placename_candidates took {elapsed:.3f}s, found {len(result)} candidates")
        return result


# =============================================================================
# BAYESIAN SELECTOR WITH CTC LOSS
# =============================================================================

class BayesianSelector:
    """
    Bayesian candidate selector using CTC loss as likelihood and DB frequency as prior.
    
    Theory:
    -------
    P(candidate | image) ∝ P(image | candidate) × P(candidate)
    
    log P(candidate | image) = log P(image | candidate) + log P(candidate) + const
    
    Where:
    - log P(image | candidate) ≈ -CTC_loss / len(candidate)
    - log P(candidate) = log((freq + α) / (total + α×V))  [Laplace smoothed]
    """
    
    def __init__(self, model, syms, config: BayesianConfig):
        self.model = model
        self.syms = syms
        self.config = config
    
    def compute_log_prior(self, frequency: int, total_frequency: int, vocab_size: int) -> float:
        """Compute Laplace-smoothed log prior probability."""
        alpha = self.config.smoothing_alpha
        numerator = frequency + alpha
        denominator = total_frequency + alpha * vocab_size
        return log(numerator / denominator)
    
    def compute_log_likelihood(
        self, 
        log_probs: torch.Tensor, 
        text: str, 
        context: str,
        entity_position: Optional[Tuple[int, int]],
        original_length: int
    ) -> Tuple[float, float]:
        """
        Compute normalized log-likelihood from CTC loss using cached log_probs.
        
        Returns (raw_loss, normalized_log_likelihood)
        """
        # Build full text with candidate
        if entity_position:
            start, end = entity_position
            full_text = context[:start] + text + context[end:]
        else:
            full_text = text
        
        # Calculate CTC loss using cached log_probs
        raw_loss = calculate_ctc_loss_from_logprobs(log_probs, self.syms, full_text)
        
        if raw_loss == float('inf'):
            return float('inf'), float('-inf')
        
        # Normalize by candidate length (not original length) to make per-character 
        # scores comparable across different-length candidates.
        # Note: We can't compute CTC loss on just the entity region (CTC requires 
        # the full sequence), so we normalize the full-line loss by entity length.
        candidate_length = len(text) if text else 1
        if candidate_length > 0:
            # Convert loss to log probability and normalize per character
            # This gives us log P(image | candidate) / len(candidate)
            normalized = -raw_loss / candidate_length
        else:
            normalized = float('-inf')
        
        return raw_loss, normalized
    
    def select_best(
        self,
        candidates: List[Dict],
        original_text: str,
        log_probs: Optional[torch.Tensor],
        context: str,
        entity_position: Optional[Tuple[int, int]],
        total_frequency: int,
        vocab_size: int,
        entity_type: str = "entity"
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Select the best candidate using Bayesian scoring.
        
        If log_probs is None (no Pylaia model), falls back to frequency-only mode.
        """
        if not candidates:
            return None, []
        
        start_time = time.time()
        ctc_time = 0.0
        prior_time = 0.0
        
        # Ensure original is in candidates
        original_in_list = any(c['text'].lower() == original_text.lower() for c in candidates)
        if not original_in_list:
            candidates = [{'text': original_text, 'frequency': 0, 'distance': 0}] + candidates
        
        scored = []
        best_score = float('-inf')
        best_candidate = None
        original_length = len(original_text)
        
        for cand in candidates:
            text = cand['text']
            freq = cand.get('frequency', 0)
            distance = cand.get('distance', 0)
            
            # Compute log prior from database frequency
            prior_start = time.time()
            log_prior = self.compute_log_prior(freq, total_frequency, vocab_size)
            prior_time += time.time() - prior_start
            
            # Compute log likelihood from CTC loss (if model available)
            if log_probs is not None and self.model is not None:
                ctc_start = time.time()
                raw_loss, norm_likelihood = self.compute_log_likelihood(
                    log_probs, text, context, entity_position, original_length
                )
                ctc_time += time.time() - ctc_start
            else:
                raw_loss = float('inf')
                norm_likelihood = 0.0  # No visual evidence, rely on prior
                if log_probs is None:
                    logger.debug(f"      log_probs is None for candidate '{text}' - using frequency-only scoring")
                if self.model is None:
                    logger.debug(f"      model is None for candidate '{text}' - using frequency-only scoring")
            
            # Combined score
            score = (
                self.config.likelihood_weight * norm_likelihood +
                self.config.prior_weight * log_prior
            )
            
            # Bonus for original extraction
            is_original = (text.lower() == original_text.lower())
            if is_original:
                score += self.config.original_bonus
            
            # Penalty for edit distance from original
            score -= distance * self.config.distance_penalty
            
            # Penalty for declension mismatch (for forenames)
            if entity_type == 'forename' and not cand.get('declension_matches', True):
                score -= self.config.declension_mismatch_penalty
            
            scored_cand = {
                'text': text,
                'frequency': freq,
                'distance': distance,
                'english_name': cand.get('english_name'),
                'declension': cand.get('declension'),
                'raw_ctc_loss': raw_loss if raw_loss != float('inf') else None,
                'normalized_log_likelihood': norm_likelihood if norm_likelihood != float('-inf') else None,
                'log_prior': log_prior,
                'total_score': score,
                'is_original': is_original
            }
            scored.append(scored_cand)
            
            if score > best_score:
                best_score = score
                best_candidate = scored_cand
        
        # Convert scores to probabilities using softmax
        # Sort by score (descending) for probability computation
        scored_sorted = sorted(scored, key=lambda x: -x['total_score'])
        
        # Extract scores for softmax
        scores = np.array([c['total_score'] for c in scored_sorted])
        
        # Handle extreme values to avoid overflow
        # Subtract max score for numerical stability
        scores_shifted = scores - np.max(scores) if len(scores) > 0 else scores
        
        # Compute softmax probabilities
        exp_scores = np.exp(scores_shifted)
        probabilities = exp_scores / np.sum(exp_scores) if np.sum(exp_scores) > 0 else exp_scores
        
        # Add probability to each candidate
        for i, cand in enumerate(scored_sorted):
            cand['probability'] = float(probabilities[i])
        
        elapsed = time.time() - start_time
        if elapsed > 0.05:  # Only log if it takes significant time (>50ms)
            logger.debug(f"    select_best: {elapsed:.3f}s total (CTC: {ctc_time:.3f}s, prior: {prior_time:.3f}s) for {len(candidates)} candidates")
        
        return best_candidate, scored_sorted


# =============================================================================
# PYLAIA MODEL MANAGER
# =============================================================================

class PylaiaModelManager:
    """Manages Pylaia model loading and caching for CTC loss computation."""
    
    _instance = None
    _model = None
    _syms = None
    _loaded = False
    _load_attempted = False
    
    @classmethod
    def get_instance(cls) -> 'PylaiaModelManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance. Useful for debugging or forcing a reload."""
        cls._instance = None
        cls._model = None
        cls._syms = None
        cls._loaded = False
        cls._load_attempted = False
    
    def load_model(self, force_retry: bool = False) -> bool:
        """Load the Pylaia model. Returns True if successful.
        
        Args:
            force_retry: If True, reset the load attempt flag and try again.
        """
        if self._loaded:
            logger.debug("Pylaia model already loaded, skipping")
            return True
        
        if self._load_attempted and not force_retry:
            logger.warning("Pylaia model load was previously attempted and failed, skipping retry")
            logger.warning("  Use force_retry=True to attempt loading again")
            return False
        
        if force_retry:
            logger.info("Resetting load attempt flag and retrying model load...")
            self._load_attempted = False
        
        logger.info("Attempting to load Pylaia model for Bayesian correction...")
        
        if not LAIA_AVAILABLE:
            logger.warning("✗ laia package not available - using frequency-only Bayesian correction")
            logger.warning("  Install laia package to enable CTC loss computation")
            self._load_attempted = True  # Don't retry if laia is not available
            return False
        
        logger.debug(f"  Checking for models in: {PYLAIA_MODELS_DIR}")
        logger.debug(f"  Directory exists: {PYLAIA_MODELS_DIR.exists()}")
        
        try:
            checkpoint, model_file, syms_file = find_latest_pylaia_model()
            logger.debug(f"  Found checkpoint: {checkpoint}")
            logger.debug(f"  Found model file: {model_file}")
            logger.debug(f"  Found symbols file: {syms_file}")
            
            self._model, self._syms = load_pylaia_model(checkpoint, model_file, syms_file)
            self._loaded = True
            self._load_attempted = False  # Reset on success
            logger.info("✓ Pylaia model loaded successfully for Bayesian correction")
            return True
        except FileNotFoundError as e:
            logger.error(f"✗ Failed to load Pylaia model - files not found: {e}")
            logger.error("  Check that model files exist in bootstrap_training_data/pylaia_models/")
            self._load_attempted = True  # Don't retry if files are missing
            return False
        except Exception as e:
            logger.error(f"✗ Failed to load Pylaia model: {type(e).__name__}: {e}")
            logger.error(f"  Exception details: {e}", exc_info=True)
            logger.warning("  Using frequency-only mode (no CTC loss computation)")
            # Only cache the failure if it's a non-recoverable error
            # For other errors, allow retry on next call
            if isinstance(e, (FileNotFoundError, ImportError, RuntimeError)):
                self._load_attempted = True
            return False
    
    @property
    def model(self):
        return self._model
    
    @property
    def syms(self):
        return self._syms
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


# =============================================================================
# MAIN POST-CORRECTION FUNCTIONS
# =============================================================================

def apply_bayesian_correction_with_ctc(
    llm_results: List[Dict],
    name_db: NameDatabase,
    config: BayesianConfig,
    lines: List[Dict[str, Any]],
    out_dir: str
) -> List[Dict]:
    """
    Apply Bayesian correction to named entities using CTC loss from Pylaia.
    
    Args:
        llm_results: Results from call_llm_for_post_correction
        name_db: NameDatabase instance
        config: BayesianConfig instance
        lines: Original lines with line_id and paths to line images
        out_dir: Output directory containing line images
    
    Returns:
        Updated results with corrected named entities
    """
    total_start = time.time()
    logger.info("Starting Bayesian correction with CTC loss...")
    
    # Initialize Pylaia model manager
    model_load_start = time.time()
    logger.info("Initializing Pylaia model manager...")
    model_manager = PylaiaModelManager.get_instance()
    logger.info("Attempting to load Pylaia model...")
    load_success = model_manager.load_model()
    model_load_time = time.time() - model_load_start
    
    if load_success:
        logger.info(f"✓ Model loaded successfully in {model_load_time:.2f}s")
    else:
        logger.warning(f"✗ Model loading failed or skipped in {model_load_time:.2f}s")
        logger.warning("  Bayesian correction will use frequency-only mode (no CTC loss)")
    
    if model_load_time > 1.0:
        logger.debug(f"  Model loading took {model_load_time:.2f}s")
    
    # Create selector (works with or without model)
    selector = BayesianSelector(model_manager.model, model_manager.syms, config)
    
    # Track statistics for summary
    lines_with_model = 0
    lines_without_model = 0
    entities_with_ctc = 0
    entities_without_ctc = 0
    
    # Map line keys to image paths and HTR text
    line_map = {}
    for idx, line in enumerate(lines, 1):
        key = f"L{idx:02d}"
        line_id = line.get("line_id", "")
        
        # Try to find the line image
        image_path = None
        lines_dir = Path(out_dir) / "lines"
        if lines_dir.exists():
            for ext in ['.png', '.jpg', '.jpeg']:
                potential = lines_dir / f"{line_id}{ext}"
                if potential.exists():
                    image_path = potential
                    break
        
        line_map[key] = {
            'image_path': image_path,
            'htr_text': line.get('htr_text', ''),
            'bbox': line.get('bbox')
        }
    
    # Process each line
    for line_result in llm_results:
        key = line_result.get('key', '')
        line_info = line_map.get(key, {})
        image_path = line_info.get('image_path')
        raw_htr_text = line_info.get('htr_text', '')
        
        # CRITICAL: Clean the HTR text for CTC loss computation
        # Raw HTR has inter-character spaces and <space> tokens that must be cleaned
        cleaned_htr_text = clean_htr_text(raw_htr_text)
        
        # Compute log_probs once per line if model is available
        log_probs = None
        line_inference_start = time.time()
        
        if not model_manager.is_loaded:
            logger.debug(f"  Model not loaded for {key}, skipping log_probs computation")
        elif not image_path:
            logger.debug(f"  No image path for {key}, skipping log_probs computation")
        elif not image_path.exists():
            logger.debug(f"  Image path does not exist for {key}: {image_path}, skipping log_probs computation")
        else:
            try:
                logger.debug(f"  Computing log_probs for {key} from {image_path}")
                image_tensor = preprocess_line_image(image_path)
                log_probs = get_model_log_probs(model_manager.model, image_tensor)
                logger.debug(f"  Successfully computed log_probs for {key}: shape {log_probs.shape}")
            except Exception as e:
                logger.warning(f"  Could not compute log_probs for {key}: {type(e).__name__}: {e}")
                logger.debug(f"  Exception details:", exc_info=True)
        
        line_inference_time = time.time() - line_inference_start
        if line_inference_time > 0.5:
            logger.debug(f"  Line {key} inference took {line_inference_time:.3f}s")
        if log_probs is None:
            logger.debug(f"  log_probs is None for {key} - candidates will use frequency-only scoring")
            lines_without_model += 1
        else:
            lines_with_model += 1
        
        # Process forenames
        forenames_start = time.time()
        corrected_forenames = []
        for fn in line_result.get('forenames', []):
            if not fn.get('text'):
                continue
            
            # Use LLM-extracted entity for candidate search, but extract actual raw HTR text for CTC
            llm_entity_text = fn['text']
            
            # Find position in CLEANED raw HTR text for CTC computation
            entity_position = find_entity_in_text(llm_entity_text, cleaned_htr_text, config.fuzzy_match_threshold)
            
            # Extract the actual text from raw HTR at this position (what Pylaia actually saw)
            raw_htr_entity_text = llm_entity_text  # Default to LLM text if position not found
            if entity_position:
                start, end = entity_position
                raw_htr_entity_text = cleaned_htr_text[start:end]
            
            # Find candidates using LLM-extracted text (might have corrections for matching)
            candidates = name_db.find_forename_candidates(llm_entity_text, fn.get('declension'))
            
            # Use raw HTR entity text as original_text for CTC loss computation
            select_start = time.time()
            best, scored = selector.select_best(
                candidates=candidates,
                original_text=raw_htr_entity_text,  # Use actual raw HTR text for fair comparison
                log_probs=log_probs,
                context=cleaned_htr_text,  # Use cleaned raw HTR text as context for CTC
                entity_position=entity_position,
                total_frequency=name_db.total_forename_freq,
                vocab_size=len(name_db.forenames) or 1,
                entity_type='forename'
            )
            # Track CTC usage
            if log_probs is not None:
                entities_with_ctc += 1
            else:
                entities_without_ctc += 1
            select_time = time.time() - select_start
            if select_time > 0.1:
                logger.debug(f"    select_best for forename '{llm_entity_text}' took {select_time:.3f}s ({len(candidates)} candidates)")
            
            # Build result with full diagnostics like process_htr_with_bayesian_names.py
            result_entry = {
                'original': raw_htr_entity_text,  # Store what was actually in raw HTR
                'position_in_htr': list(entity_position) if entity_position else None,
                'candidates': [
                    {
                        'text': s['text'],
                        'frequency': s['frequency'],
                        'distance': s['distance'],
                        'english_name': s.get('english_name'),
                        'declension': s.get('declension'),
                        'raw_ctc_loss': s.get('raw_ctc_loss'),
                        'normalized_log_likelihood': s.get('normalized_log_likelihood'),
                        'log_prior': s.get('log_prior'),
                        'total_score': s.get('total_score'),
                        'probability': s.get('probability'),  # Probability from softmax over all candidates
                        'is_original': s.get('is_original')
                    }
                    for s in scored[:TOP_CANDIDATES]  # Top candidates with probabilities
                ] if scored else [],
                'declension': fn.get('declension')
            }
            
            if best:
                result_entry['best_candidate'] = {
                    'text': best['text'],
                    'english_name': best.get('english_name'),
                    'declension': best.get('declension'),
                    'frequency': best.get('frequency'),
                    'distance': best.get('distance'),
                    'normalized_log_likelihood': best.get('normalized_log_likelihood'),
                    'log_prior': best.get('log_prior'),
                    'total_score': best.get('total_score'),
                    'probability': best.get('probability')  # Probability from softmax over all candidates
                }
                result_entry['corrected'] = best['text']
                result_entry['confidence'] = 'bayesian_corrected' if best['text'] != raw_htr_entity_text else 'original'
            else:
                result_entry['corrected'] = raw_htr_entity_text
                result_entry['confidence'] = 'original'
            
            corrected_forenames.append(result_entry)
        line_result['forenames'] = corrected_forenames
        
        # Process surnames
        corrected_surnames = []
        for sn in line_result.get('surnames', []):
            if not sn.get('text'):
                continue
            
            # Use LLM-extracted entity for candidate search, but extract actual raw HTR text for CTC
            llm_entity_text = sn['text']
            
            # Find position in CLEANED raw HTR text for CTC computation
            entity_position = find_entity_in_text(llm_entity_text, cleaned_htr_text, config.fuzzy_match_threshold)
            
            # Extract the actual text from raw HTR at this position (what Pylaia actually saw)
            raw_htr_entity_text = llm_entity_text  # Default to LLM text if position not found
            if entity_position:
                start, end = entity_position
                raw_htr_entity_text = cleaned_htr_text[start:end]
            
            # Find candidates using LLM-extracted text (might have corrections for matching)
            candidates = name_db.find_surname_candidates(llm_entity_text)
            # Debug: log if we're looking for a name like Kyngesor
            if 'ynges' in llm_entity_text.lower():
                logger.info(f"    DEBUG: Found {len(candidates)} surname candidates for '{llm_entity_text}'")
                kford = [c for c in candidates if 'ford' in c['text'].lower()]
                if kford:
                    logger.info(f"    DEBUG: *ford candidates: {kford}")
            
            # Use raw HTR entity text as original_text for CTC loss computation
            best, scored = selector.select_best(
                candidates=candidates,
                original_text=raw_htr_entity_text,  # Use actual raw HTR text for fair comparison
                log_probs=log_probs,
                context=cleaned_htr_text,  # Use cleaned raw HTR text as context for CTC
                entity_position=entity_position,
                total_frequency=name_db.total_surname_freq,
                vocab_size=len(name_db.surnames) or 1,
                entity_type='surname'
            )
            # Track CTC usage
            if log_probs is not None:
                entities_with_ctc += 1
            else:
                entities_without_ctc += 1
            
            # Build result with full diagnostics
            result_entry = {
                'original': raw_htr_entity_text,  # Store what was actually in raw HTR
                'position_in_htr': list(entity_position) if entity_position else None,
                'candidates': [
                    {
                        'text': s['text'],
                        'frequency': s['frequency'],
                        'distance': s['distance'],
                        'raw_ctc_loss': s.get('raw_ctc_loss'),
                        'normalized_log_likelihood': s.get('normalized_log_likelihood'),
                        'log_prior': s.get('log_prior'),
                        'total_score': s.get('total_score'),
                        'probability': s.get('probability'),  # Probability from softmax over all candidates
                        'is_original': s.get('is_original')
                    }
                    for s in scored[:TOP_CANDIDATES]  # Top candidates with probabilities
                ] if scored else []
            }
            
            if best:
                result_entry['best_candidate'] = {
                    'text': best['text'],
                    'frequency': best.get('frequency'),
                    'distance': best.get('distance'),
                    'normalized_log_likelihood': best.get('normalized_log_likelihood'),
                    'log_prior': best.get('log_prior'),
                    'total_score': best.get('total_score'),
                    'probability': best.get('probability')  # Probability from softmax over all candidates
                }
                result_entry['corrected'] = best['text']
                result_entry['confidence'] = 'bayesian_corrected' if best['text'].lower() != raw_htr_entity_text.lower() else 'original'
            else:
                result_entry['corrected'] = raw_htr_entity_text
                result_entry['confidence'] = 'original'
            
            corrected_surnames.append(result_entry)
        line_result['surnames'] = corrected_surnames
        
        # Process placenames
        corrected_placenames = []
        for pn in line_result.get('placenames', []):
            if not pn.get('text'):
                continue
            
            # Use LLM-extracted entity for candidate search, but extract actual raw HTR text for CTC
            llm_entity_text = pn['text']
            
            # Find position in CLEANED raw HTR text for CTC computation
            entity_position = find_entity_in_text(llm_entity_text, cleaned_htr_text, config.fuzzy_match_threshold)
            
            # Extract the actual text from raw HTR at this position (what Pylaia actually saw)
            raw_htr_entity_text = llm_entity_text  # Default to LLM text if position not found
            if entity_position:
                start, end = entity_position
                raw_htr_entity_text = cleaned_htr_text[start:end]
            
            # Find candidates using LLM-extracted text (might have corrections for matching)
            candidates = name_db.find_placename_candidates(llm_entity_text)
            
            # Use raw HTR entity text as original_text for CTC loss computation
            best, scored = selector.select_best(
                candidates=candidates,
                original_text=raw_htr_entity_text,  # Use actual raw HTR text for fair comparison
                log_probs=log_probs,
                context=cleaned_htr_text,  # Use cleaned raw HTR text as context for CTC
                entity_position=entity_position,
                total_frequency=name_db.total_placename_freq,
                vocab_size=len(name_db.placenames) or 1,
                entity_type='placename'
            )
            # Track CTC usage
            if log_probs is not None:
                entities_with_ctc += 1
            else:
                entities_without_ctc += 1
            
            # Build result with full diagnostics
            result_entry = {
                'original': raw_htr_entity_text,  # Store what was actually in raw HTR
                'position_in_htr': list(entity_position) if entity_position else None,
                'candidates': [
                    {
                        'text': s['text'],
                        'frequency': s['frequency'],
                        'distance': s['distance'],
                        'raw_ctc_loss': s.get('raw_ctc_loss'),
                        'normalized_log_likelihood': s.get('normalized_log_likelihood'),
                        'log_prior': s.get('log_prior'),
                        'total_score': s.get('total_score'),
                        'probability': s.get('probability'),  # Probability from softmax over all candidates
                        'is_original': s.get('is_original')
                    }
                    for s in scored[:TOP_CANDIDATES]  # Top candidates with probabilities
                ] if scored else []
            }
            
            if best:
                result_entry['best_candidate'] = {
                    'text': best['text'],
                    'frequency': best.get('frequency'),
                    'distance': best.get('distance'),
                    'normalized_log_likelihood': best.get('normalized_log_likelihood'),
                    'log_prior': best.get('log_prior'),
                    'total_score': best.get('total_score'),
                    'probability': best.get('probability')  # Probability from softmax over all candidates
                }
                result_entry['corrected'] = best['text']
                result_entry['confidence'] = 'bayesian_corrected' if best['text'].lower() != raw_htr_entity_text.lower() else 'original'
            else:
                result_entry['corrected'] = raw_htr_entity_text
                result_entry['confidence'] = 'original'
            
            corrected_placenames.append(result_entry)
        line_result['placenames'] = corrected_placenames
    
    total_time = time.time() - total_start
    num_lines = len(llm_results)
    num_entities = sum(
        len(line_result.get('forenames', [])) + 
        len(line_result.get('surnames', [])) + 
        len(line_result.get('placenames', []))
        for line_result in llm_results
    )
    
    # Summary statistics
    logger.info(f"Bayesian correction completed: {num_lines} lines, {num_entities} entities in {total_time:.2f}s ({total_time/num_lines:.3f}s per line)")
    
    if model_manager.is_loaded:
        logger.info(f"  Model usage: {lines_with_model} lines with model, {lines_without_model} lines without model")
        logger.info(f"  CTC usage: {entities_with_ctc} entities with CTC loss, {entities_without_ctc} entities frequency-only")
        if entities_without_ctc > 0:
            logger.warning(f"  ⚠ {entities_without_ctc} entities used frequency-only mode (missing image paths or model inference failed)")
    else:
        logger.warning(f"  ⚠ Model was not loaded - all {num_entities} entities used frequency-only mode")
        logger.warning(f"     Check logs above for model loading errors")
    
    return llm_results


def fill_missing_ctc_losses(
    post_correction_result: Dict[str, Any],
    lines: List[Dict[str, Any]],
    name_db: NameDatabase,
    config: BayesianConfig,
    out_dir: str
) -> Tuple[Dict[str, Any], bool]:
    """
    Fill in missing raw_ctc_loss values in a loaded post-correction result.
    
    This is useful when loading existing post-correction JSON files that were
    created before the model was available or when CTC loss computation failed.
    
    Args:
        post_correction_result: Loaded post-correction result dictionary
        lines: Original lines with line_id and paths to line images
        name_db: NameDatabase instance
        config: BayesianConfig instance
        out_dir: Output directory containing line images
    
    Returns:
        Tuple of (updated post_correction_result, was_updated) where was_updated
        indicates if any CTC loss values were actually filled in.
    """
    # Check if any entities have null raw_ctc_loss
    has_null_ctc = False
    for line_result in post_correction_result.get('lines', []):
        for entity_type in ['forenames', 'surnames', 'placenames']:
            for entity in line_result.get(entity_type, []):
                for candidate in entity.get('candidates', []):
                    if candidate.get('raw_ctc_loss') is None:
                        has_null_ctc = True
                        break
                if has_null_ctc:
                    break
            if has_null_ctc:
                break
        if has_null_ctc:
            break
    
    if not has_null_ctc:
        logger.debug("  No missing CTC loss values found, skipping recomputation")
        return post_correction_result, False
    
    logger.info("  Found missing CTC loss values, recomputing with Pylaia model...")
    updated_count = 0
    
    # Initialize model manager
    model_manager = PylaiaModelManager.get_instance()
    if not model_manager.is_loaded:
        load_success = model_manager.load_model()
        if not load_success:
            logger.warning("  Could not load model for CTC loss recomputation, keeping null values")
            return post_correction_result
    
    # Create selector
    selector = BayesianSelector(model_manager.model, model_manager.syms, config)
    
    # Map line keys to image paths and HTR text
    line_map = {}
    for idx, line in enumerate(lines, 1):
        key = f"L{idx:02d}"
        line_id = line.get("line_id", "")
        
        # Try to find the line image
        image_path = None
        lines_dir = Path(out_dir) / "lines"
        if lines_dir.exists():
            for ext in ['.png', '.jpg', '.jpeg']:
                potential = lines_dir / f"{line_id}{ext}"
                if potential.exists():
                    image_path = potential
                    break
        
        line_map[key] = {
            'image_path': image_path,
            'htr_text': line.get('htr_text', ''),
            'bbox': line.get('bbox')
        }
    
    # Process each line
    for line_result in post_correction_result.get('lines', []):
        key = line_result.get('line_id', '')
        line_info = line_map.get(key, {})
        image_path = line_info.get('image_path')
        raw_htr_text = line_info.get('htr_text', '')
        
        # Clean HTR text for CTC loss computation
        cleaned_htr_text = clean_htr_text(raw_htr_text)
        
        # Compute log_probs once per line if model is available and image exists
        log_probs = None
        if model_manager.is_loaded and image_path and image_path.exists():
            try:
                image_tensor = preprocess_line_image(image_path)
                log_probs = get_model_log_probs(model_manager.model, image_tensor)
            except Exception as e:
                logger.debug(f"  Could not compute log_probs for {key}: {e}")
        
        # Update forenames
        for fn in line_result.get('forenames', []):
            if not fn.get('text'):
                continue
            
            # Check if any candidate has null raw_ctc_loss
            needs_update = any(
                c.get('raw_ctc_loss') is None 
                for c in fn.get('candidates', [])
            )
            
            if not needs_update:
                continue
            
            llm_entity_text = fn['text']
            entity_position = find_entity_in_text(llm_entity_text, cleaned_htr_text, config.fuzzy_match_threshold)
            raw_htr_entity_text = llm_entity_text
            if entity_position:
                start, end = entity_position
                raw_htr_entity_text = cleaned_htr_text[start:end]
            
            # Recompute CTC loss for all candidates
            candidates = name_db.find_forename_candidates(llm_entity_text, fn.get('declension'))
            best, scored = selector.select_best(
                candidates=candidates,
                original_text=raw_htr_entity_text,
                log_probs=log_probs,
                context=cleaned_htr_text,
                entity_position=entity_position,
                total_frequency=name_db.total_forename_freq,
                vocab_size=len(name_db.forenames) or 1,
                entity_type='forename'
            )
            
            # Update candidates with new CTC loss values
            scored_dict = {s['text']: s for s in scored}
            for candidate in fn.get('candidates', []):
                candidate_text = candidate.get('text')
                if candidate_text in scored_dict:
                    new_data = scored_dict[candidate_text]
                    if candidate.get('raw_ctc_loss') is None and new_data.get('raw_ctc_loss') is not None:
                        updated_count += 1
                    candidate['raw_ctc_loss'] = new_data.get('raw_ctc_loss')
                    candidate['normalized_log_likelihood'] = new_data.get('normalized_log_likelihood')
        
        # Update surnames
        for sn in line_result.get('surnames', []):
            if not sn.get('text'):
                continue
            
            needs_update = any(
                c.get('raw_ctc_loss') is None 
                for c in sn.get('candidates', [])
            )
            
            if not needs_update:
                continue
            
            llm_entity_text = sn['text']
            entity_position = find_entity_in_text(llm_entity_text, cleaned_htr_text, config.fuzzy_match_threshold)
            raw_htr_entity_text = llm_entity_text
            if entity_position:
                start, end = entity_position
                raw_htr_entity_text = cleaned_htr_text[start:end]
            
            candidates = name_db.find_surname_candidates(llm_entity_text)
            best, scored = selector.select_best(
                candidates=candidates,
                original_text=raw_htr_entity_text,
                log_probs=log_probs,
                context=cleaned_htr_text,
                entity_position=entity_position,
                total_frequency=name_db.total_surname_freq,
                vocab_size=len(name_db.surnames) or 1,
                entity_type='surname'
            )
            
            scored_dict = {s['text']: s for s in scored}
            for candidate in sn.get('candidates', []):
                candidate_text = candidate.get('text')
                if candidate_text in scored_dict:
                    new_data = scored_dict[candidate_text]
                    if candidate.get('raw_ctc_loss') is None and new_data.get('raw_ctc_loss') is not None:
                        updated_count += 1
                    candidate['raw_ctc_loss'] = new_data.get('raw_ctc_loss')
                    candidate['normalized_log_likelihood'] = new_data.get('normalized_log_likelihood')
        
        # Update placenames
        for pn in line_result.get('placenames', []):
            if not pn.get('text'):
                continue
            
            needs_update = any(
                c.get('raw_ctc_loss') is None 
                for c in pn.get('candidates', [])
            )
            
            if not needs_update:
                continue
            
            llm_entity_text = pn['text']
            entity_position = find_entity_in_text(llm_entity_text, cleaned_htr_text, config.fuzzy_match_threshold)
            raw_htr_entity_text = llm_entity_text
            if entity_position:
                start, end = entity_position
                raw_htr_entity_text = cleaned_htr_text[start:end]
            
            candidates = name_db.find_placename_candidates(llm_entity_text)
            best, scored = selector.select_best(
                candidates=candidates,
                original_text=raw_htr_entity_text,
                log_probs=log_probs,
                context=cleaned_htr_text,
                entity_position=entity_position,
                total_frequency=name_db.total_placename_freq,
                vocab_size=len(name_db.placenames) or 1,
                entity_type='placename'
            )
            
            scored_dict = {s['text']: s for s in scored}
            for candidate in pn.get('candidates', []):
                candidate_text = candidate.get('text')
                if candidate_text in scored_dict:
                    new_data = scored_dict[candidate_text]
                    if candidate.get('raw_ctc_loss') is None and new_data.get('raw_ctc_loss') is not None:
                        updated_count += 1
                    candidate['raw_ctc_loss'] = new_data.get('raw_ctc_loss')
                    candidate['normalized_log_likelihood'] = new_data.get('normalized_log_likelihood')
    
    if updated_count > 0:
        logger.info(f"  Finished recomputing missing CTC loss values: updated {updated_count} candidates")
        return post_correction_result, True
    else:
        logger.info("  Attempted to recompute CTC loss values but none were successfully computed")
        return post_correction_result, False


def process_image_post_correction(
    lines: List[Dict[str, Any]],
    image_name: str,
    name_db: Optional[NameDatabase] = None,
    config: Optional[BayesianConfig] = None,
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single image through post-correction pipeline.
    
    Args:
        lines: List of line dictionaries from merge_htr_data (with htr_text, bbox, line_id)
        image_name: Name of the image being processed
        name_db: Optional NameDatabase instance (created if not provided)
        config: Optional BayesianConfig instance
        out_dir: Output directory containing line images for CTC loss computation
    
    Returns:
        Dictionary with corrected lines and extracted entities
    """
    if config is None:
        config = BayesianConfig()
    
    if name_db is None:
        name_db = NameDatabase(DB_PATH, config)
    
    # Prepare input for LLM
    lines_data = []
    for idx, line in enumerate(lines, 1):
        entry = {
            "key": f"L{idx:02d}",
            "htr_text": line.get("htr_text", ""),
        }
        if line.get("bbox"):
            entry["bbox"] = line["bbox"]
        lines_data.append(entry)
    
    logger.info(f"  Running LLM post-correction for {image_name} ({len(lines_data)} lines)...")
    
    # Call LLM for correction and entity extraction
    llm_results = call_llm_for_post_correction(lines_data, out_dir=out_dir)
    
    # Apply Bayesian correction with CTC loss
    if name_db.forenames or name_db.surnames or name_db.placenames:
        logger.info(f"  Applying Bayesian named entity correction with Pylaia CTC loss...")
        llm_results = apply_bayesian_correction_with_ctc(
            llm_results, name_db, config, lines, out_dir or ""
        )
    
    # Build result
    result = {
        "image_name": image_name,
        "lines": []
    }
    
    # Map results back to original line structure
    result_map = {r['key']: r for r in llm_results}
    
    for idx, line in enumerate(lines, 1):
        key = f"L{idx:02d}"
        llm_line = result_map.get(key, {})
        
        result["lines"].append({
            "line_id": key,
            "original_htr_text": line.get("htr_text", ""),
            "corrected_text": llm_line.get("corrected_text", line.get("htr_text", "")),
            "bbox": line.get("bbox"),
            "forenames": llm_line.get("forenames", []),
            "surnames": llm_line.get("surnames", []),
            "placenames": llm_line.get("placenames", [])
        })
    
    # Rate limiting
    time.sleep(0.5)
    
    return result


def process_post_correction_results(
    batch_results: Dict[str, Any],
    image_lines_map: Dict[str, Dict[str, Any]],
    name_db: NameDatabase,
    config: BayesianConfig,
    out_dir: str
) -> Dict[str, Dict[str, Any]]:
    """
    Process post-correction results and apply Bayesian correction.
    
    Args:
        batch_results: Dictionary mapping image names (via sanitize_custom_id) to LLM results (List[Dict])
        image_lines_map: Dictionary mapping image_path -> {
            'lines': list of line dicts,
            'image_name': name of image
        }
        name_db: NameDatabase instance
        config: BayesianConfig instance
        out_dir: Base output directory containing line images
    
    Returns:
        Dictionary mapping image_name -> post-correction result dict
    """
    processed_results = {}
    
    for image_path, data in image_lines_map.items():
        image_name = data['image_name']
        lines = data['lines']
        batch_key = sanitize_custom_id(image_name)
        
        # Get LLM results for this image
        llm_results = batch_results.get(batch_key, [])
        
        if not llm_results:
            logger.warning(f"No post-correction results found for {image_name}")
            # Fallback: use raw HTR
            llm_results = [
                {
                    "key": f"L{idx:02d}",
                    "corrected_text": line.get("htr_text", ""),
                    "forenames": [],
                    "surnames": [],
                    "placenames": []
                }
                for idx, line in enumerate(lines, 1)
            ]
        
        # Apply Bayesian correction with CTC loss
        basename = os.path.splitext(image_name)[0]
        htr_work_dir = os.path.join(out_dir, basename)
        
        if name_db.forenames or name_db.surnames or name_db.placenames:
            logger.info(f"  Applying Bayesian named entity correction for {image_name}...")
            llm_results = apply_bayesian_correction_with_ctc(
                llm_results, name_db, config, lines, htr_work_dir
            )
        
        # Build result
        result = {
            "image_name": image_name,
            "lines": []
        }
        
        # Map results back to original line structure
        result_map = {r['key']: r for r in llm_results}
        
        for idx, line in enumerate(lines, 1):
            key = f"L{idx:02d}"
            llm_line = result_map.get(key, {})
            
            result["lines"].append({
                "line_id": key,
                "original_htr_text": line.get("htr_text", ""),
                "corrected_text": llm_line.get("corrected_text", line.get("htr_text", "")),
                "bbox": line.get("bbox"),
                "forenames": llm_line.get("forenames", []),
                "surnames": llm_line.get("surnames", []),
                "placenames": llm_line.get("placenames", [])
            })
        
        processed_results[image_name] = result
    
    return processed_results


def get_corrected_lines_for_stitching(post_correction_result: Dict) -> List[Dict]:
    """
    Convert post-correction result to format expected by Step 2a (stitching).
    
    Returns list of dicts with 'transcription' field containing corrected Latin
    with Bayesian-corrected named entities substituted.
    """
    result_lines = []
    
    # Validate input
    if not isinstance(post_correction_result, dict):
        logger.error(f"get_corrected_lines_for_stitching: post_correction_result is not a dict: {type(post_correction_result)}")
        return []
    
    lines = post_correction_result.get("lines", [])
    if not isinstance(lines, list):
        logger.error(f"get_corrected_lines_for_stitching: 'lines' is not a list: {type(lines)}")
        return []
    
    for line in lines:
        # Validate line is a dict
        if not isinstance(line, dict):
            logger.error(f"get_corrected_lines_for_stitching: line is not a dict: {type(line)}, value: {line}")
            continue
        
        corrected_text = line.get("corrected_text", "")
        
        # Substitute Bayesian-corrected named entities into the text
        for entity_type in ['forenames', 'surnames', 'placenames']:
            entities = line.get(entity_type, [])
            if not isinstance(entities, list):
                continue
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                original = entity.get('original', '')
                corrected = entity.get('corrected', '')
                if original and corrected and original != corrected:
                    corrected_text = corrected_text.replace(original, corrected)
        
        # Include full Bayesian correction results with candidates and probabilities
        # This allows Step 2a to reconcile names across images using probability information
        result_lines.append({
            "id": line.get("line_id"),
            "transcription": corrected_text,
            "bbox": line.get("bbox"),
            "entities": {
                "forenames": line.get("forenames", []),  # Includes candidates, probabilities, scores
                "surnames": line.get("surnames", []),    # Includes candidates, probabilities, scores
                "placenames": line.get("placenames", []) # Includes candidates, probabilities, scores
            }
        })
    
    return result_lines
