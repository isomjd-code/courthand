"""
Main workflow for bootstrap Pylaia training with Gemini 3 as teacher.
"""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from ground_truth.query import JSON_QUERY
from line_preprocessor.parser import parse_kraken_json_for_processing
from workflow_manager.kenlm_utils import ensure_kenlm_files
from workflow_manager.settings import (
    API_MAX_RETRIES,
    API_RETRY_DELAY,
    API_TIMEOUT,
    GEMINI_API_KEY,
    BASE_DIR,
    IMAGE_DIR,
    KENLM_MODEL_PATH,
    KENLM_MODEL_WEIGHT,
    KENLM_USE_BINARY,
    KRAKEN_ENV,
    LOG_DIR,
    MODEL_TEXT,
    MODEL_VISION,
    PYLAIA_ARCH,
    PYLAIA_ENV,
    PYLAIA_MODEL,
    PYLAIA_SYMS,
    SURNAME_DB_PATH,
    logger,
)

# Bootstrap training specific paths
BOOTSTRAP_DATA_DIR = os.path.join(BASE_DIR, "bootstrap_training_data")
BOOTSTRAP_CHECKPOINT_FILE = os.path.join(BOOTSTRAP_DATA_DIR, "checkpoint.json")
BOOTSTRAP_STATS_FILE = os.path.join(BOOTSTRAP_DATA_DIR, "statistics.json")
BOOTSTRAP_PYLAIA_MODEL_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "pylaia_models")
BOOTSTRAP_DATASET_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "datasets")
BOOTSTRAP_WORK_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "work")

# Initial Pylaia model (epoch=220) - uses architecture from different location
INITIAL_PYLAIA_MODEL = os.path.join(BASE_DIR, "models", "epoch=220-lowest_va_cer.ckpt")
# Architecture is in a different project directory
# Check if it's a file or directory - if directory, look for model file inside
_latin_model_path = "/home/qj/projects/latin/model"
if os.path.isdir(_latin_model_path):
    # If it's a directory, the model file might be inside or have a different name
    # Try common names
    for name in ["model", "model.pkl", "architecture.pkl"]:
        candidate = os.path.join(_latin_model_path, name)
        if os.path.isfile(candidate):
            INITIAL_PYLAIA_ARCH = candidate
            break
    else:
        # If no file found, use the directory path (Pylaia might handle it)
        INITIAL_PYLAIA_ARCH = _latin_model_path
else:
    INITIAL_PYLAIA_ARCH = _latin_model_path
INITIAL_PYLAIA_SYMS = os.path.join(BASE_DIR, "models", "syms.txt")
# Initial model uses 96px height, subsequent models use 128px
INITIAL_MODEL_IMAGE_HEIGHT = 96
SUBSEQUENT_MODEL_IMAGE_HEIGHT = 128
# Resolution for LLM line images (same as training data)
LLM_LINE_IMAGE_HEIGHT = 128

# Architecture for first retrain (from model_courthand_3090C)
FIRST_RETRAIN_ARCH = os.path.join(BASE_DIR, "model_courthand_3090C", "model")
FIRST_RETRAIN_SYMS = os.path.join(BASE_DIR, "model_courthand_3090C", "syms.txt")

# Lines threshold for retraining
LINES_PER_RETRAIN = 3000
MIN_VALID_TRAINING_LINES = 5000  # Minimum valid lines required before creating a new model version

# Maximum Levenshtein distance/similarity threshold between cleaned htr_text and corrected_text for training
# Lines with distance/similarity exceeding this threshold are filtered out during dataset generation
# This helps ensure training data has reasonable correspondence between HTR output and corrections
# 
# If value < 1.0: Treated as similarity threshold (0.0-1.0), e.g., 0.80 means 80% similarity required
# If value >= 1.0: Treated as absolute distance threshold, e.g., 50 means max 50 character differences
MAX_LEVENSHTEIN_DISTANCE = 0.5  # Similarity threshold: require 50% similarity between HTR and corrected text


def parse_image_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse roll and rotulus numbers from CP40 image filename.
    
    Handles formats like:
    - "CP 40-559 055-a.jpg" -> ("559", "055")
    - "CP40-562 307a.jpg" -> ("562", "307")
    - "CP40-565 481-d-a.jpg" -> ("565", "481")
    
    Args:
        filename: Image filename (with or without path)
        
    Returns:
        Tuple of (roll_number, rotulus_number) or None if parsing fails
    """
    basename = os.path.basename(filename)
    
    # Pattern: CP40-XXX YYY or CP 40-XXX YYY
    patterns = [
        r"CP\s*40[-\s](\d+)\s+(\d+)",  # "CP 40-559 055" or "CP40-559 055"
        r"CP40[-\s](\d+)[-\s](\d+)",  # "CP40-559-055"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            roll = match.group(1)
            rotulus = match.group(2)
            # Remove any trailing letters (like "a", "b", "d" in "055-a")
            rotulus = re.sub(r'[a-z]+$', '', rotulus, flags=re.IGNORECASE)
            return (roll, rotulus)
    
    return None


class BootstrapTrainingManager:
    """
    Bootstrap training manager for Pylaia model using Gemini 3 as teacher.
    
    Workflow:
    1. For each image in input_images:
       a. Use Gemini 2.5 Pro to detect rotation angle (with limited thinking tokens)
       b. Rotate image if necessary
       c. Run Kraken segmentation + Pylaia HTR
       d. Query database for index data
       e. Send to Gemini 3 for correction with bounding boxes
       f. Store corrected transcriptions
    2. Every 3,000 corrected lines, retrain Pylaia model
    3. Checkpoint all intermediate results for resuming
    """
    
    def __init__(self, force: bool = False):
        """
        Initialize the bootstrap training manager.
        
        Args:
            force: If True, reprocesses all images even if results exist.
        """
        self.force = force
        self.state = self._load_checkpoint()
        
        # Create directories
        os.makedirs(BOOTSTRAP_DATA_DIR, exist_ok=True)
        os.makedirs(BOOTSTRAP_PYLAIA_MODEL_DIR, exist_ok=True)
        os.makedirs(BOOTSTRAP_DATASET_DIR, exist_ok=True)
        os.makedirs(BOOTSTRAP_WORK_DIR, exist_ok=True)
        
        # Initialize Gemini 3.0 Flash client for all tasks
        if not GEMINI_API_KEY:
            logger.critical("FATAL: GEMINI_API_KEY not set.")
            sys.exit("FATAL: GEMINI_API_KEY not set.")
        
        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={
                'api_version': 'v1alpha',
                'timeout': API_TIMEOUT
            }
        )
        self.free_client = self.client  # Use same client
        self.uploaded_files_cache = {}
        
        logger.info("Initialized Gemini 3.0 Flash client for bootstrap training")
        
        # Processing queue (non-batch mode - process individually with Gemini 3.0 Flash)
        self.batch_queue = []  # Queue of images waiting for processing
        self.batch_queue_lock = threading.Lock()  # Lock for thread-safe queue access
        self.batch_processing_lock = threading.Lock()  # Lock to ensure only one batch is processed at a time
        
        # Statistics
        self.stats = self._load_statistics()
        self.stats_lock = threading.Lock()  # Lock for thread-safe stats updates
        
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint state for resuming."""
        if os.path.exists(BOOTSTRAP_CHECKPOINT_FILE):
            try:
                with open(BOOTSTRAP_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                state = None
        else:
            state = None
        
        # If no state or state is invalid, create default
        if not state:
            state = {
                "corrected_lines_count": 0,
                "current_model_version": 0,
                "last_retrain_line_count": 0,
                "pylaia_models": {},  # Store checkpoint paths for each model version
            }
        
        # Remove deprecated fields to keep checkpoint small
        # processed_images and pending_images can be reconstructed from disk
        if "processed_images" in state:
            del state["processed_images"]
        if "pending_images" in state:
            del state["pending_images"]
        
        # Detect existing model directories and update current_model_version if needed
        # This handles the case where training was interrupted and checkpoint.json wasn't updated
        # Also populate pylaia_models with checkpoint info for any existing models
        if os.path.exists(BOOTSTRAP_PYLAIA_MODEL_DIR):
            highest_version = 0
            found_models = []
            
            for item in os.listdir(BOOTSTRAP_PYLAIA_MODEL_DIR):
                if item.startswith("model_v") and os.path.isdir(os.path.join(BOOTSTRAP_PYLAIA_MODEL_DIR, item)):
                    try:
                        # Extract version number from "model_v1", "model_v2", etc.
                        version = int(item.replace("model_v", ""))
                        if version > highest_version:
                            highest_version = version
                        found_models.append(version)
                    except ValueError:
                        continue
            
            # Ensure pylaia_models dict exists
            if "pylaia_models" not in state:
                state["pylaia_models"] = {}
            
            # For each found model, check if we have checkpoint info, and populate if missing
            for version in found_models:
                model_key = f"v{version}"
                model_dir = os.path.join(BOOTSTRAP_PYLAIA_MODEL_DIR, f"model_v{version}")
                
                # If we don't have info for this model, or the stored checkpoint doesn't exist, find it
                needs_update = False
                if model_key not in state["pylaia_models"]:
                    needs_update = True
                else:
                    stored_checkpoint = state["pylaia_models"][model_key].get("checkpoint")
                    if not stored_checkpoint or not os.path.exists(stored_checkpoint):
                        needs_update = True
                
                if needs_update:
                    # Find the best checkpoint and model files
                    # Use static method to find checkpoint (can't call instance method in __init__)
                    best_checkpoint = BootstrapTrainingManager._find_best_checkpoint_static(model_dir)
                    model_file = os.path.join(model_dir, "model")
                    syms_file = os.path.join(model_dir, "syms.txt")
                    
                    if best_checkpoint:
                        # Determine image height: v0 uses 96px, v1+ uses 128px
                        image_height = INITIAL_MODEL_IMAGE_HEIGHT if version == 0 else SUBSEQUENT_MODEL_IMAGE_HEIGHT
                        state["pylaia_models"][model_key] = {
                            "checkpoint": best_checkpoint,
                            "model_file": model_file if os.path.exists(model_file) else None,
                            "syms": syms_file if os.path.exists(syms_file) else None,
                            "image_height": image_height,  # Store required line height for this model
                        }
                        logger.info(
                            f"Auto-detected model v{version}: checkpoint={best_checkpoint}, image_height={image_height}px"
                        )
            
            # If we found a higher version than what's in state, update it
            if highest_version > state.get("current_model_version", 0):
                logger.info(
                    f"Detected existing model directories up to v{highest_version}, "
                    f"but checkpoint.json has v{state.get('current_model_version', 0)}. "
                    f"Updating current_model_version to {highest_version}."
                )
                state["current_model_version"] = highest_version
        
        # Note: We don't save checkpoint here because this is called during __init__
        # The state will be saved the next time _save_checkpoint() is called
        return state
    
    def _save_checkpoint(self):
        """Save checkpoint state - only essential fields to keep file small."""
        try:
            # Only save essential state - processed_images and pending_images can be reconstructed from disk
            essential_state = {
                "corrected_lines_count": self.state.get("corrected_lines_count", 0),
                "current_model_version": self.state.get("current_model_version", 0),
                "last_retrain_line_count": self.state.get("last_retrain_line_count", 0),
                "pylaia_models": self.state.get("pylaia_models", {}),
            }
            with open(BOOTSTRAP_CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(essential_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _is_image_processed(self, image_path: str) -> bool:
        """Check if image has been processed by checking if corrections exist on disk."""
        basename = os.path.splitext(os.path.basename(image_path))[0]
        corrected_dir = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines", basename)
        metadata_path = os.path.join(corrected_dir, "metadata.json")
        return os.path.exists(corrected_dir) and os.path.exists(metadata_path)
    
    def _load_statistics(self) -> Dict[str, Any]:
        """Load statistics."""
        if os.path.exists(BOOTSTRAP_STATS_FILE):
            try:
                with open(BOOTSTRAP_STATS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load statistics: {e}")
        
        return {
            "images_processed": 0,
            "images_rotated": 0,
            "total_lines_processed": 0,
            "total_lines_corrected": 0,
            "gemini_rotation_calls": 0,
            "gemini_correction_calls": 0,
            "model_retrains": 0,
        }
    
    def _save_statistics(self):
        """Save statistics."""
        try:
            with open(BOOTSTRAP_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
    
    def _get_rotation_angle(self, image_path: str) -> Optional[int]:
        """
        Use Gemini 2.5 Pro to detect rotation angle.
        
        DISABLED: Rotation detection is currently disabled.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Integer rotation angle (clockwise) or None if detection fails
        """
        # Rotation detection is disabled
        logger.debug(f"Rotation detection disabled, skipping for {image_path}")
        return None
        
        # Use Gemini 2.5 Pro for rotation detection with limited thinking tokens
        try:
            # Upload image using paid client for rotation detection
            file_ref = self._upload_file_with_client(image_path, self.client)
            if not file_ref:
                return None
            
            prompt = """Look at this manuscript image. Determine the clockwise rotation angle (in degrees) needed to achieve normal reading orientation where text is horizontal and readable from left to right.

CRITICAL: Your response must contain ONLY one of these integer values: 0, 90, 180, or 270. Do not include any explanation, thinking, or other text. Just the number.

Examples:
- If image is already correct: 0
- If image needs 90° clockwise rotation: 90
- If image needs 180° clockwise rotation: 180
- If image needs 270° clockwise rotation: 270

Your final answer must be exactly one of these numbers: 0, 90, 180, or 270. Nothing else."""
            
            parts = [
                types.Part.from_text(text=prompt),
                types.Part.from_uri(file_uri=file_ref.uri, mime_type=file_ref.mime_type)
            ]
            
            # Use JSON schema to ensure structured response
            json_schema = {
                "type": "OBJECT",
                "properties": {
                    "rotation_angle": {
                        "type": "NUMBER",
                        "description": "Clockwise rotation angle in degrees. Must be one of: 0, 90, 180, or 270."
                    }
                },
                "required": ["rotation_angle"],
                "description": "Response containing the rotation angle needed to orient the image correctly."
            }
            
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,  # Increased to allow room for thinking tokens + response
                response_mime_type="application/json",
                response_schema=json_schema,
            )
            
            # Use Gemini 2.5 Pro for rotation detection
            rotation_model = "gemini-2.5-pro"
            
            # Log the full request
            logger.info(f"[Rotation API Request] Model: {rotation_model}")
            logger.info(f"[Rotation API Request] Prompt: {prompt}")
            logger.info(f"[Rotation API Request] Image URI: {file_ref.uri}")
            logger.info(f"[Rotation API Request] Config: temperature={config.temperature}, max_output_tokens={config.max_output_tokens}")
            logger.info(f"[Rotation API Request] Using paid client for Gemini 2.5 Pro")
            
            response = self._call_api_with_retry(
                self.client,  # Use paid client for Gemini 2.5 Pro
                rotation_model,  # Use Gemini 2.5 Pro for rotation detection
                parts,
                config
            )
            
            # Log the full response
            logger.info(f"[Rotation API Response] Response object: {response}")
            if response:
                logger.info(f"[Rotation API Response] Has text attr: {hasattr(response, 'text')}")
                if hasattr(response, 'text'):
                    logger.info(f"[Rotation API Response] Text: {response.text}")
                logger.info(f"[Rotation API Response] Has candidates: {hasattr(response, 'candidates')}")
                if hasattr(response, 'candidates'):
                    logger.info(f"[Rotation API Response] Candidates count: {len(response.candidates) if response.candidates else 0}")
                    if response.candidates:
                        for idx, candidate in enumerate(response.candidates):
                            logger.info(f"[Rotation API Response] Candidate {idx}: {candidate}")
                            if hasattr(candidate, 'content'):
                                logger.info(f"[Rotation API Response] Candidate {idx} content: {candidate.content}")
                                if hasattr(candidate.content, 'parts'):
                                    logger.info(f"[Rotation API Response] Candidate {idx} parts: {candidate.content.parts}")
            
            if not response:
                logger.warning("No response from rotation detection API")
                return None
            
            # Try to extract JSON from response
            json_data = None
            
            # First try direct text attribute (should contain JSON)
            if hasattr(response, 'text') and response.text:
                try:
                    json_data = json.loads(response.text.strip())
                    logger.info(f"[Rotation API Response] Parsed JSON from text: {json_data}")
                except json.JSONDecodeError as e:
                    logger.warning(f"[Rotation API Response] Failed to parse JSON from text: {e}")
                    logger.debug(f"[Rotation API Response] Text content: {response.text[:200]}...")
            
            # If not available, try candidates
            if json_data is None and hasattr(response, 'candidates') and response.candidates:
                try:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    try:
                                        json_data = json.loads(part.text.strip())
                                        logger.info(f"[Rotation API Response] Parsed JSON from candidate part: {json_data}")
                                        break
                                    except json.JSONDecodeError:
                                        continue
                except (IndexError, AttributeError, TypeError) as e:
                    logger.debug(f"Error extracting JSON from candidates: {e}")
            
            if json_data is None:
                logger.warning("Could not extract JSON from rotation detection API response")
                logger.warning(f"[Rotation API Response] Full response structure: {dir(response)}")
                return None
            
            # Extract rotation_angle from JSON
            if "rotation_angle" in json_data:
                angle = json_data["rotation_angle"]
                # Convert to int if it's a float
                if isinstance(angle, float):
                    angle = int(round(angle))
                else:
                    angle = int(angle)
                
                # Normalize to 0-359 range first
                angle = angle % 360
                # Round to nearest valid rotation (0, 90, 180, 270)
                angle = self._normalize_rotation_angle(angle)
                logger.info(f"[Rotation API Response] Extracted and normalized angle: {angle}")
                return angle
            else:
                logger.warning(f"JSON response missing 'rotation_angle' field: {json_data}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting rotation angle: {e}", exc_info=True)
            return None
    
    def _normalize_rotation_angle(self, angle: int) -> int:
        """
        Normalize rotation angle to the nearest valid value: 0, 90, 180, or 270.
        
        Args:
            angle: Rotation angle in degrees (0-359)
            
        Returns:
            Normalized angle: 0, 90, 180, or 270
        """
        # Normalize to 0-359 range
        angle = angle % 360
        
        # Round to nearest valid rotation (0, 90, 180, 270)
        valid_angles = [0, 90, 180, 270]
        # Find the closest valid angle
        normalized = min(valid_angles, key=lambda x: min(abs(angle - x), abs(angle - x - 360), abs(angle - x + 360)))
        return normalized
    
    def _rotate_image(self, image_path: str, angle: int) -> bool:
        """
        Rotate image on disk if angle is not 0.
        Only allows rotations of 0, 90, 180, or 270 degrees.
        
        Args:
            image_path: Path to the image file
            angle: Clockwise rotation angle in degrees (must be 0, 90, 180, or 270)
            
        Returns:
            True if rotation was applied, False otherwise
        """
        if angle == 0:
            return False
        
        # Validate angle is one of the allowed values
        if angle not in [0, 90, 180, 270]:
            logger.warning(f"Invalid rotation angle {angle}. Normalizing to nearest valid value.")
            angle = self._normalize_rotation_angle(angle)
            if angle == 0:
                logger.info(f"Normalized angle is 0, skipping rotation")
                return False
        
        try:
            img = Image.open(image_path)
            # PIL rotate uses counter-clockwise, so negate
            rotated = img.rotate(-angle, expand=True)
            rotated.save(image_path, quality=95)
            self.stats["images_rotated"] += 1
            logger.info(f"Rotated {image_path} by {angle} degrees")
            return True
        except Exception as e:
            logger.error(f"Error rotating image {image_path}: {e}")
            return False
    
    def _upload_file_with_client(self, path: str, client: Any) -> Optional[Any]:
        """Upload file to Gemini API with specified client."""
        # Use a cache key that includes client info
        cache_key = f"{path}_{id(client)}"
        if cache_key in self.uploaded_files_cache:
            return self.uploaded_files_cache[cache_key]
        
        logger.info(f"Uploading {path} to Gemini...")
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        
        for attempt in range(API_MAX_RETRIES):
            try:
                f = client.files.upload(
                    file=path,
                    config=types.UploadFileConfig(mime_type=mime)
                )
                
                while f.state.name == "PROCESSING":
                    time.sleep(1)
                    f = client.files.get(name=f.name)
                
                if f.state.name != "ACTIVE":
                    logger.error(f"File upload failed for {path}: {f.state.name}")
                    return None
                
                self.uploaded_files_cache[cache_key] = f
                return f
                
            except Exception as e:
                if attempt < API_MAX_RETRIES - 1:
                    wait_time = API_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Upload retry {attempt + 1}/{API_MAX_RETRIES} in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"File upload failed after {API_MAX_RETRIES} attempts: {e}")
                    return None
        
        return None
    
    def _upload_file(self, path: str) -> Optional[Any]:
        """Upload file to Gemini API with main client (for corrections)."""
        return self._upload_file_with_client(path, self.client)
    
    def cleanup_cloud_files(self) -> None:
        """
        Clean up all uploaded files from Google Gemini cloud storage.
        
        Removes all files that were uploaded for vision model processing.
        Useful for managing cloud storage quotas and costs.
        Called occasionally to prevent accumulation of uploaded files.
        """
        logger.info("--- STARTING CLOUD STORAGE CLEANUP ---")
        deleted_count = 0
        error_count = 0
        
        # Helper function to delete a single file
        def delete_file(client, file_name: str, client_name: str = "main") -> bool:
            """Delete a single file. Returns True if successful, False otherwise."""
            try:
                client.files.delete(name=file_name)
                return True
            except Exception as e:
                logger.debug(f"Error deleting file {file_name} from {client_name} client: {e}")
                return False
        
        # Cleanup files from main client
        try:
            logger.info("Listing files from main client...")
            file_iterator = self.client.files.list()
            files_list = list(file_iterator)  # Convert to list to get count
            total_files = len(files_list)
            logger.info(f"Found {total_files} files to delete from main client")
            
            if total_files > 0:
                # Delete files in parallel
                max_workers = min(20, total_files)  # Limit to 20 concurrent deletions
                logger.info(f"Deleting {total_files} files in parallel (max {max_workers} workers)...")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all deletion tasks
                    future_to_file = {
                        executor.submit(delete_file, self.client, f.name, "main"): f.name
                        for f in files_list
                    }
                    
                    # Track progress as deletions complete
                    completed = 0
                    for future in as_completed(future_to_file):
                        if future.result():
                            deleted_count += 1
                        else:
                            error_count += 1
                        completed += 1
                        if completed % 10 == 0 or completed == total_files:
                            logger.info(f"Deleted {completed}/{total_files} files from main client...")
        except Exception as e:
            logger.error(f"Error listing files from main client: {e}")
        
        # Cleanup files from free client (used for rotation detection)
        try:
            logger.info("Listing files from free client...")
            file_iterator = self.free_client.files.list()
            files_list = list(file_iterator)  # Convert to list to get count
            total_files = len(files_list)
            logger.info(f"Found {total_files} files to delete from free client")
            
            if total_files > 0:
                # Delete files in parallel
                max_workers = min(20, total_files)  # Limit to 20 concurrent deletions
                logger.info(f"Deleting {total_files} files in parallel (max {max_workers} workers)...")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all deletion tasks
                    future_to_file = {
                        executor.submit(delete_file, self.free_client, f.name, "free"): f.name
                        for f in files_list
                    }
                    
                    # Track progress as deletions complete
                    completed = 0
                    for future in as_completed(future_to_file):
                        if future.result():
                            deleted_count += 1
                        else:
                            error_count += 1
                        completed += 1
                        if completed % 10 == 0 or completed == total_files:
                            logger.info(f"Deleted {completed}/{total_files} files from free client...")
        except Exception as e:
            logger.error(f"Error listing files from free client: {e}")
        
        # Clear the cache after cleanup
        self.uploaded_files_cache = {}
        
        logger.info(f"--- CLOUD STORAGE CLEANUP COMPLETE: Deleted {deleted_count} files, {error_count} errors ---")
    
    def _call_api_with_retry(self, client, model_name: str, contents: List[Any], config: Any):
        """Call Gemini API with retry logic."""
        for attempt in range(API_MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                return response
            except Exception as e:
                error_msg = str(e).lower()
                is_timeout = any(keyword in error_msg for keyword in [
                    '504', 'timeout', 'deadline exceeded', 'timed out'
                ])
                
                if is_timeout and attempt < API_MAX_RETRIES - 1:
                    wait_time = API_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"API retry {attempt + 1}/{API_MAX_RETRIES} in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise Exception("API call failed after all retries")
    
    def _run_htr_tools(self, image_path: str) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
        """
        Run Kraken segmentation and Pylaia HTR on image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (kraken_json_path, htr_txt_path, merged_lines)
        """
        basename = os.path.splitext(os.path.basename(image_path))[0]
        work_dir = os.path.join(BOOTSTRAP_DATA_DIR, "htr_work", basename)
        os.makedirs(work_dir, exist_ok=True)
        
        kraken_json = os.path.join(work_dir, "kraken.json")
        lines_dir = os.path.join(work_dir, "lines")
        os.makedirs(lines_dir, exist_ok=True)
        list_txt = os.path.join(work_dir, "img_list.txt")
        htr_res = os.path.join(work_dir, "htr.txt")
        
        # Determine which Pylaia model to use
        model_version = self.state.get("current_model_version", 0)
        
        # If we have a model version > 0, check if it actually exists on disk
        # If it exists, use it (even if current valid lines count is below threshold)
        # The threshold only applies when creating NEW model versions, not using existing ones
        if model_version > 0:
            model_dir = os.path.join(BOOTSTRAP_PYLAIA_MODEL_DIR, f"model_v{model_version}")
            model_exists = False
            
            # Check if model directory exists and has a checkpoint
            if os.path.exists(model_dir):
                # Check for checkpoint in stored state
                model_key = f"v{model_version}"
                if model_key in self.state.get("pylaia_models", {}):
                    stored_checkpoint = self.state["pylaia_models"][model_key].get("checkpoint")
                    if stored_checkpoint and os.path.exists(stored_checkpoint):
                        model_exists = True
                
                # Also check model directory directly
                if not model_exists:
                    # Check root and experiment directory for checkpoints
                    best_checkpoint = self._find_best_checkpoint(model_dir)
                    if best_checkpoint:
                        model_exists = True
            
            if not model_exists:
                logger.warning(
                    f"Model v{model_version} specified but no checkpoint found. "
                    f"Falling back to initial model (v0)."
                )
                model_version = 0
                self.state["current_model_version"] = 0
                self._save_checkpoint()
            else:
                # Model exists - use it regardless of current valid lines count
                # (The minimum lines check only applies when creating NEW versions)
                logger.info(f"Using existing model v{model_version} (checkpoint found on disk)")
                
                # If checkpoint info not in state, find and store it
                model_key = f"v{model_version}"
                if model_key not in self.state.get("pylaia_models", {}):
                    best_checkpoint = self._find_best_checkpoint(model_dir)
                    model_file = os.path.join(model_dir, "model")
                    syms_file = os.path.join(model_dir, "syms.txt")
                    
                    if best_checkpoint:
                        if "pylaia_models" not in self.state:
                            self.state["pylaia_models"] = {}
                        # Determine image height: v0 uses 96px, v1+ uses 128px
                        image_height = INITIAL_MODEL_IMAGE_HEIGHT if model_version == 0 else SUBSEQUENT_MODEL_IMAGE_HEIGHT
                        self.state["pylaia_models"][model_key] = {
                            "checkpoint": best_checkpoint,
                            "model_file": model_file if os.path.exists(model_file) else None,
                            "syms": syms_file if os.path.exists(syms_file) else None,
                            "image_height": image_height,  # Store required line height for this model
                        }
                        logger.info(f"Stored checkpoint info for model v{model_version}: checkpoint={best_checkpoint}, image_height={image_height}px")
                        self._save_checkpoint()
        
        if model_version == 0:
            pylaia_model = INITIAL_PYLAIA_MODEL
            pylaia_arch = INITIAL_PYLAIA_ARCH
            pylaia_syms = INITIAL_PYLAIA_SYMS
            
            # Verify model architecture file exists and is a file (not directory)
            if os.path.isdir(pylaia_arch):
                logger.warning(f"Model architecture path is a directory: {pylaia_arch}")
                # Try to find model file inside
                for name in ["model", "model.pkl", "architecture.pkl"]:
                    candidate = os.path.join(pylaia_arch, name)
                    if os.path.isfile(candidate):
                        pylaia_arch = candidate
                        logger.info(f"Using model file: {pylaia_arch}")
                        break
                else:
                    logger.error(f"Could not find model file in directory {pylaia_arch}")
                    return None, None, []
            elif not os.path.isfile(pylaia_arch):
                logger.error(f"Model architecture file not found: {pylaia_arch}")
                return None, None, []
        else:
            # Use latest trained model - check if we have stored checkpoint info
            model_key = f"v{model_version}"
            
            if model_key in self.state.get("pylaia_models", {}):
                # Use stored checkpoint path (best validation checkpoint)
                model_info = self.state["pylaia_models"][model_key]
                pylaia_model = model_info.get("checkpoint")
                pylaia_arch = model_info.get("model_file")
                pylaia_syms = model_info.get("syms")
                
                if not pylaia_model or not os.path.exists(pylaia_model):
                    logger.warning(f"Stored checkpoint for v{model_version} not found: {pylaia_model}")
                    pylaia_model = None
                else:
                    logger.info(f"Using stored checkpoint for model v{model_version}: {pylaia_model}")
            else:
                logger.debug(f"No stored model info for {model_key} in state, checking model directory...")
                pylaia_model = None
            
            # Fallback: look for checkpoint in model directory
            if not pylaia_model:
                model_dir = os.path.join(
                    BOOTSTRAP_PYLAIA_MODEL_DIR,
                    f"model_v{model_version}"
                )
                # Look for any checkpoint with "lowest_va_cer" in the model directory
                if os.path.exists(model_dir):
                    logger.debug(f"Checking model directory: {model_dir}")
                    
                    # First, check root of model directory
                    for filename in os.listdir(model_dir):
                        if "lowest_va_cer" in filename and filename.endswith(".ckpt"):
                            pylaia_model = os.path.join(model_dir, filename)
                            logger.info(f"Found checkpoint in model directory root: {pylaia_model}")
                            break
                    
                    # If not found in root, check experiment directory (where Pylaia saves checkpoints)
                    if not pylaia_model:
                        experiment_dir = os.path.join(model_dir, "experiment")
                        if os.path.exists(experiment_dir):
                            logger.debug(f"Checking experiment directory: {experiment_dir}")
                            # Look for checkpoints with "lowest_va_cer" in experiment directory
                            best_epoch = -1
                            for filename in os.listdir(experiment_dir):
                                if "lowest_va_cer" in filename and filename.endswith(".ckpt"):
                                    # Extract epoch number to find the most recent best checkpoint
                                    try:
                                        epoch_str = filename.split("=")[1].split("-")[0]
                                        epoch = int(epoch_str)
                                        if epoch > best_epoch:
                                            best_epoch = epoch
                                            pylaia_model = os.path.join(experiment_dir, filename)
                                    except (ValueError, IndexError):
                                        # If we can't parse epoch, use first one found
                                        if pylaia_model is None:
                                            pylaia_model = os.path.join(experiment_dir, filename)
                            
                            if pylaia_model:
                                logger.info(f"Found checkpoint in experiment directory: {pylaia_model} (epoch {best_epoch})")
                            else:
                                logger.warning(f"Model v{model_version} experiment directory exists but no 'lowest_va_cer' checkpoint found. Training may be in progress or failed.")
                        else:
                            logger.warning(f"Model v{model_version} directory exists but no experiment directory found.")
                
                if pylaia_model:
                    pylaia_arch = os.path.join(model_dir, "model")
                    pylaia_syms = os.path.join(model_dir, "syms.txt")
                else:
                    # Final fallback to initial model
                    logger.warning(
                        f"Model v{model_version} not found (checked state and {model_dir}), "
                        f"using initial model. This may indicate retraining hasn't completed yet."
                    )
                    pylaia_model = INITIAL_PYLAIA_MODEL
                    pylaia_arch = INITIAL_PYLAIA_ARCH
                    pylaia_syms = INITIAL_PYLAIA_SYMS
        
        # Check if we need to process
        needs_processing = self.force or not (os.path.exists(htr_res) and os.path.getsize(htr_res) > 0)
        
        if needs_processing:
            # Run Kraken segmentation (only if not already done in parallel phase)
            if not os.path.exists(kraken_json) or os.path.getsize(kraken_json) == 0:
                cmd_kraken = f"source {KRAKEN_ENV} && kraken -i '{image_path}' '{kraken_json}' --device cuda:0 segment -bl"
                self._run_command(cmd_kraken, "Kraken Segmentation", raise_on_error=False)
                
                if not os.path.exists(kraken_json):
                    logger.warning(f"Kraken segmentation failed for {image_path}")
                    return None, None, []
            else:
                logger.debug(f"Kraken segmentation already exists for {os.path.basename(image_path)}, skipping")
            
            # Check if Kraken found any lines
            try:
                with open(kraken_json, 'r') as f:
                    kraken_data = json.load(f)
                    if not kraken_data.get('lines'):
                        logger.warning(f"No lines found in Kraken segmentation for {image_path}")
                        return kraken_json, htr_res, []
            except Exception as e:
                logger.warning(f"Error checking Kraken JSON: {e}")
            
            # Preprocess lines with appropriate height for model version
            # Check if stored in model metadata, otherwise use version-based logic
            model_key = f"v{model_version}" if model_version > 0 else None
            if model_key and model_key in self.state.get("pylaia_models", {}):
                stored_height = self.state["pylaia_models"][model_key].get("image_height")
                if stored_height:
                    image_height = stored_height
                    logger.debug(f"Using stored image_height={image_height}px for model v{model_version}")
                else:
                    # Fallback to version-based logic
                    image_height = SUBSEQUENT_MODEL_IMAGE_HEIGHT if model_version > 0 else INITIAL_MODEL_IMAGE_HEIGHT
            else:
                # Use version-based logic
                if model_version == 0:
                    # Initial model uses 96px height
                    image_height = INITIAL_MODEL_IMAGE_HEIGHT
                else:
                    # Subsequent models use 128px height
                    image_height = SUBSEQUENT_MODEL_IMAGE_HEIGHT
            
            # Create a temporary preprocessing script with the correct height
            # Use a temporary location instead of work_dir to avoid clutter
            import tempfile
            temp_script_dir = os.path.join(BOOTSTRAP_DATA_DIR, "temp_scripts")
            os.makedirs(temp_script_dir, exist_ok=True)
            preprocess_script = os.path.join(temp_script_dir, f"preprocess_with_height_{basename}.py")
            with open(preprocess_script, 'w', encoding='utf-8') as f:
                f.write(f"""#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '{BASE_DIR}')

# CRITICAL: Override FINAL_LINE_HEIGHT in config BEFORE any imports
# This must happen before line_preprocessor_greyscale modules are imported
import line_preprocessor_greyscale.config
line_preprocessor_greyscale.config.FINAL_LINE_HEIGHT = {image_height}

# Import after setting config
from line_preprocessor_greyscale.runner import _load_image, _expand_polygons, _save_line_image
from line_preprocessor.parser import parse_kraken_json_for_processing
from line_preprocessor_greyscale.processing import initial_line_extraction, process_line_image_greyscale
from PIL import Image
from tqdm import tqdm

def custom_main(image_path, json_path, output_dir, pylaia_list_path, height):
    page_image = _load_image(image_path)
    lines_to_process = parse_kraken_json_for_processing(json_path)
    if not lines_to_process:
        print("No text lines found in the Kraken JSON. Exiting.", file=sys.stderr)
        sys.exit(0)
    
    print(f"Found {{len(lines_to_process)}} lines to process. Expanding polygons...")
    expanded_lines = _expand_polygons(lines_to_process)
    
    print(f"Processing lines with height={{height}}px...")
    os.makedirs(output_dir, exist_ok=True)
    with open(pylaia_list_path, "w", encoding="utf-8") as pylaia_list_file:
        for line_data in tqdm(expanded_lines, desc="Processing lines"):
            try:
                initial_result = initial_line_extraction(
                    page_image,
                    line_data["polygon"],
                    line_data["baseline"],
                    padding=10,
                )
                if not initial_result:
                    continue
                
                line_rect_img, line_polygon_coords, line_baseline_points = initial_result
                # Explicitly pass height parameter
                final_image = process_line_image_greyscale(
                    line_rect_img,
                    line_polygon_coords,
                    line_baseline_points,
                    final_canvas_height=height,
                    line_id_for_debug=line_data["id"],
                )
                if final_image:
                    abs_path = _save_line_image(final_image, output_dir, line_data["id"])
                    pylaia_list_file.write(f"{{abs_path}}\\n")
            except Exception as exc:
                print(f"    - FATAL WARNING: Unhandled exception on line {{line_data['id']}}: {{exc}}", file=sys.stderr)
    
    print(f"Pylaia input file list saved to: {{pylaia_list_path}}")

if __name__ == '__main__':
    custom_main('{image_path}', '{kraken_json}', '{lines_dir}', '{list_txt}', {image_height})
""")
            
            os.chmod(preprocess_script, 0o755)
            
            cmd_preprocess = (
                f"source {PYLAIA_ENV} && "
                f"python3 '{preprocess_script}'"
            )
            self._run_command(cmd_preprocess, "Preprocess Lines", raise_on_error=False)
            
            # Clean up temporary script after use
            try:
                if os.path.exists(preprocess_script):
                    os.remove(preprocess_script)
                    logger.debug(f"Removed temporary script: {preprocess_script}")
            except Exception as e:
                logger.debug(f"Could not remove temporary script {preprocess_script}: {e}")
            
            if os.path.exists(list_txt) and os.path.getsize(list_txt) > 0:
                # Run Pylaia decode - reuse exact format from workflow_manager/workflow.py
                # For initial model, if model_arch is a directory, we need to handle it differently
                if model_version == 0 and os.path.isdir(pylaia_arch):
                    # If it's a directory, set train_path to the directory and model_filename to just "model"
                    train_path = pylaia_arch
                    model_filename = "model"
                    cmd_parts = [
                        f"source {PYLAIA_ENV} &&",
                        "pylaia-htr-decode-ctc",
                        "--trainer.accelerator gpu",
                        "--trainer.devices 1",
                        f"--common.checkpoint '{pylaia_model}'",
                        f"--common.train_path '{train_path}'",
                        f"--common.model_filename '{model_filename}'",
                        "--decode.include_img_ids true",
                        "--decode.print_word_confidence_score true",
                    ]
                else:
                    # Use full paths for both checkpoint and model_filename (no train_path)
                    # This matches workflow_manager/workflow.py format exactly
                    cmd_parts = [
                        f"source {PYLAIA_ENV} &&",
                        "pylaia-htr-decode-ctc",
                        "--trainer.accelerator gpu",
                        "--trainer.devices 1",
                        f"--common.checkpoint '{pylaia_model}'",
                        f"--common.model_filename '{pylaia_arch}'",
                        "--decode.include_img_ids true",
                        "--decode.print_word_confidence_score true",
                    ]
                
                # Add language model if available
                if KENLM_MODEL_PATH and os.path.exists(KENLM_MODEL_PATH):
                    # Determine which format to use
                    if KENLM_USE_BINARY:
                        # Try binary format first
                        binary_path = KENLM_MODEL_PATH.replace('.arpa', '.klm')
                        if os.path.exists(binary_path):
                            lm_path = binary_path
                        else:
                            # Fall back to ARPA if binary doesn't exist
                            lm_path = KENLM_MODEL_PATH
                            logger.warning(f"Binary KenLM model not found at {binary_path}, using ARPA format")
                    else:
                        lm_path = KENLM_MODEL_PATH
                    
                    # Generate tokens and lexicon files from symbols file if needed
                    # Always regenerate to ensure they include <ctc> and are up-to-date
                    try:
                        tokens_path, lexicon_path = ensure_kenlm_files(pylaia_syms, force_regenerate=True)
                        cmd_parts.extend([
                            f"--decode.use_language_model true",
                            f"--decode.language_model_path '{lm_path}'",
                            f"--decode.tokens_path '{tokens_path}'",
                            f"--decode.lexicon_path '{lexicon_path}'",
                            f"--decode.language_model_weight {KENLM_MODEL_WEIGHT}",
                        ])
                        logger.info(f"Using KenLM language model: {lm_path} (weight: {KENLM_MODEL_WEIGHT})")
                    except Exception as e:
                        logger.warning(f"Failed to generate KenLM support files: {e}. Language model disabled.")
                
                # Add syms and list files
                cmd_parts.append(f"'{pylaia_syms}' '{list_txt}' > '{htr_res}'")
                cmd_decode = " ".join(cmd_parts)
                self._run_command(cmd_decode, "PyLaia Decode", raise_on_error=False)
            else:
                with open(htr_res, 'w') as f:
                    f.write("")
        
        # Merge HTR data with geometry
        merged_lines = self._merge_htr_data(image_path, kraken_json, htr_res)
        return kraken_json, htr_res, merged_lines
    
    def _merge_htr_data(
        self, image_path: str, kraken_json: str, htr_txt: str
    ) -> List[Dict[str, Any]]:
        """
        Merge HTR recognition text with layout geometry data.
        
        Reuses logic from workflow_manager but adapted for bootstrap training.
        """
        try:
            with open(kraken_json, 'r') as f:
                layout = json.load(f)
                layout_lines = layout.get('lines', [])
        except Exception as e:
            logger.error(f"Error reading Kraken JSON: {e}")
            return []
        
        # Map ID to Geometry (including baseline from Kraken)
        geo_map = {}
        for l in layout_lines:
            boundary = l.get('boundary')
            baseline = l.get('baseline', [])  # Get baseline from Kraken JSON
            bbox = None
            if boundary and isinstance(boundary, list) and len(boundary) > 0:
                try:
                    xs = [pt[0] for pt in boundary]
                    ys = [pt[1] for pt in boundary]
                    bbox = [int(min(ys)), int(min(xs)), int(max(ys)), int(max(xs))]
                except Exception:
                    pass
            
            # Convert baseline to string format (matching parse_kraken_json_for_processing)
            baseline_str = None
            if baseline and isinstance(baseline, list) and len(baseline) >= 2:
                baseline_str = " ".join(f"{int(point[0])},{int(point[1])}" for point in baseline)
            
            geo_map[l['id']] = {
                "bbox": bbox,
                "polygon": boundary,
                "baseline": baseline_str,  # Baseline as string format (for initial_line_extraction)
                "baseline_coords": baseline  # Baseline as coordinates (for storage)
            }
        
        structured = []
        if os.path.exists(htr_txt):
            with open(htr_txt, 'r') as f:
                htr_lines = f.readlines()
                for line in htr_lines:
                    # Parse line format: <filename> ['score1', 'score2', ...] <text>
                    confidence_match = re.search(
                        r"\[(['\"]?[\d.]+['\"]?[\s,]*)+(['\"]?[\d.]+['\"]?)\]",
                        line
                    )
                    
                    if confidence_match:
                        conf_str = confidence_match.group(0)
                        conf_start = confidence_match.start()
                        conf_end = confidence_match.end()
                        
                        filename_part = line[:conf_start].strip()
                        text_part = line[conf_end:].strip()
                        
                        conf_values = re.findall(r"['\"]?([\d.]+)['\"]?", conf_str)
                        confidence_scores = [float(c) for c in conf_values]
                        
                        match = re.search(r'([^\s]+\.(png|jpg|jpeg))', filename_part, re.IGNORECASE)
                        if match:
                            filename_path = match.group(1)
                        else:
                            parts = filename_part.split(' ', 1)
                            if len(parts) > 0:
                                filename_path = parts[0]
                            else:
                                continue
                    else:
                        match = re.search(r'(\.png|\.jpg|\.jpeg)\s', line, re.IGNORECASE)
                        if match:
                            split_idx = match.end() - 1
                            filename_path = line[:split_idx].strip()
                            text_part = line[split_idx:].strip()
                        else:
                            continue
                        confidence_scores = []
                    
                    # Extract line ID from filename (e.g., "line_001.png" -> "001")
                    line_id_match = re.search(r'([^/\\]+)\.(png|jpg|jpeg)', filename_path, re.IGNORECASE)
                    if not line_id_match:
                        continue
                    
                    line_id = line_id_match.group(1)
                    geo = geo_map.get(line_id, {})
                    
                    structured.append({
                        "line_id": line_id,
                        "htr_text": text_part,
                        "bbox": geo.get("bbox"),
                        "polygon": geo.get("polygon"),
                        "baseline": geo.get("baseline"),  # Baseline as string (for processing)
                        "baseline_coords": geo.get("baseline_coords"),  # Baseline as coordinates (for storage)
                    })
        
        # Sort by y-coordinate (top to bottom)
        structured.sort(key=lambda x: x["bbox"][0] if x["bbox"] else float('inf'))
        return structured
    
    def _run_command(self, cmd: str, description: str, raise_on_error: bool = False, show_output: bool = False):
        """
        Run shell command with error handling.
        
        Args:
            cmd: Command to run
            description: Description for logging
            raise_on_error: If True, raise exception on failure. If False, only log error.
            show_output: If True, stream output in real-time (for long-running commands like training)
        """
        logger.info(f"Running {description}...")
        try:
            # Set environment variables like train_model.sh does
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            
            if show_output:
                # For verbose output, stream directly to console
                logger.info(f"Streaming output for {description}...")
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    executable="/bin/bash",
                    env=env,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output line by line
                for line in process.stdout:
                    print(line, end='', flush=True)
                
                process.wait()
                returncode = process.returncode
                stdout = ""
                stderr = ""
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    executable="/bin/bash",
                    env=env
                )
                returncode = result.returncode
                stdout = result.stdout
                stderr = result.stderr
            
            if returncode != 0:
                error_msg = stderr if stderr else stdout
                logger.error(f"{description} failed (return code {returncode}):")
                if error_msg:
                    # Log last few lines of error for context
                    error_lines = error_msg.strip().split('\n')
                    for line in error_lines[-10:]:  # Last 10 lines
                        logger.error(f"  {line}")
                    # Also log full error to a file for debugging
                    error_log_path = os.path.join(LOG_DIR, f"{description.lower().replace(' ', '_')}_error.log")
                    try:
                        with open(error_log_path, 'w', encoding='utf-8') as f:
                            f.write(f"Command: {cmd}\n")
                            f.write(f"Return code: {returncode}\n")
                            f.write(f"\nSTDERR:\n{stderr}\n")
                            f.write(f"\nSTDOUT:\n{stdout}\n")
                        logger.error(f"Full error output saved to: {error_log_path}")
                    except Exception as log_err:
                        logger.debug(f"Could not save error log: {log_err}")
                else:
                    logger.error(f"  No error output captured")
                
                if raise_on_error:
                    raise RuntimeError(f"{description} failed with return code {returncode}")
            else:
                if not show_output and stdout:
                    # Log successful output summary (last few lines)
                    output_lines = stdout.strip().split('\n')
                    if len(output_lines) > 0:
                        logger.info(f"{description} completed successfully")
                        # Show last few lines of output
                        for line in output_lines[-5:]:
                            logger.debug(f"  {line}")
        except Exception as e:
            logger.error(f"Error running {description}: {e}")
            if raise_on_error:
                raise
    
    def _get_database_index_data(self, roll: str, rotulus: str) -> Optional[Dict[str, Any]]:
        """
        Query database for index data matching roll and rotulus.
        
        Args:
            roll: Roll number
            rotulus: Rotulus number
            
        Returns:
            JSON data from database or None
        """
        if not os.path.exists(SURNAME_DB_PATH):
            logger.warning(f"Database not found at {SURNAME_DB_PATH}")
            return None
        
        try:
            conn = sqlite3.connect(SURNAME_DB_PATH)
            cursor = conn.cursor()
            
            # Find matching case
            # Use DISTINCT to avoid duplicate cases from multiple references
            lookup_query = """
                SELECT DISTINCT c.CaseID, c.CaseRot, r.reference
                FROM TblCase c
                JOIN TblReference r ON c.DocID = r.docid
                WHERE CAST(r.reference AS TEXT) = ? 
                AND CAST(c.CaseRot AS TEXT) = ?
            """
            
            cursor.execute(lookup_query, (roll, rotulus))
            matches = cursor.fetchall()
            
            if not matches:
                logger.warning(f"No database entry found for roll {roll}, rotulus {rotulus}")
                conn.close()
                return None
            
            # Get JSON data for first match (after filtering duplicates)
            case_id = matches[0][0]
            cursor.execute(JSON_QUERY, (case_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if not row or not row[0]:
                return None
            
            return json.loads(row[0])
            
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            return None
    
    def _prepare_batch_request(
        self,
        image_path: str,
        lines: List[Dict[str, Any]],
        htr_text: str,
        index_data: Optional[Dict[str, Any]],
        max_lines_per_request: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare a request for Gemini 3.0 Flash correction (non-batch mode).
        
        Args:
            image_path: Path to the image
            lines: List of line data with bounding boxes
            htr_text: Combined HTR text (not used, but kept for compatibility)
            index_data: Database index data
            max_lines_per_request: Maximum lines allowed per image (default 50)
            
        Returns:
            Dictionary with request data or None if preparation fails or image has too many lines
        """
        # Check if corrections already exist before uploading
        basename = os.path.splitext(os.path.basename(image_path))[0]
        corrected_dir = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines", basename)
        metadata_path = os.path.join(corrected_dir, "metadata.json")
        has_corrections = os.path.exists(corrected_dir) and os.path.exists(metadata_path)
        
        if has_corrections:
            logger.info(f"Corrections already exist for {basename}, skipping batch request preparation (checked {metadata_path})")
            return None
        
        # Skip images with more than max_lines_per_request lines
        if len(lines) > max_lines_per_request:
            logger.warning(f"Skipping {basename}: image has {len(lines)} lines (max allowed: {max_lines_per_request})")
            return None
        
        def _sanitize_batch_key(name: str) -> str:
            """
            Sanitize a batch key/custom_id to satisfy Anthropic regex:
            ^[a-zA-Z0-9_-]{1,64}$
            """
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            return safe[:64]

        try:
            # Preprocess HTR text: remove spaces between characters and replace <space> with actual space
            def clean_htr_text(text: str) -> str:
                """
                Remove spaces between characters and replace <space> tokens with actual spaces.
                
                HTR output format: "I o h ' i <space> C l a n y n g"
                Should become: "Ioh'i Clanyng"
                """
                if not text:
                    return ""
                # First, replace <space> tokens with a temporary marker to preserve word boundaries
                # Use a marker that won't appear in the text
                text = text.replace("<space>", "|||SPACE_MARKER|||")
                # Remove all regular spaces (which were between individual characters)
                text = text.replace(" ", "")
                # Replace the marker back with actual spaces
                text = text.replace("|||SPACE_MARKER|||", " ")
                # Clean up any multiple spaces
                text = " ".join(text.split())
                return text
            
            # Prepare index data
            index_json = json.dumps(index_data, indent=2, ensure_ascii=False) if index_data else "No index data available"
            
            # Prepare line data
            line_data = []
            for idx, line in enumerate(lines, 1):
                key = f"L{idx:02d}"
                bbox = line.get("bbox")
                if bbox:
                    # Clean the HTR text before sending to Gemini
                    raw_htr_text = line.get("htr_text", "")
                    cleaned_htr_text = clean_htr_text(raw_htr_text)
                    # Format as [ymin, xmin, ymax, xmax]
                    line_data.append({
                        "key": key,
                        "bbox": bbox,
                        "htr_text": cleaned_htr_text
                    })
            
            line_data_json = json.dumps(line_data, indent=2, ensure_ascii=False)
            
            # Get image dimensions for resolution info
            image_width = None
            image_height = None
            try:
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            except Exception as e:
                logger.debug(f"Could not get image dimensions for {image_path}: {e}")
            
            # Determine MIME type
            if image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif image_path.lower().endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"
            
            # Build response schema for Gemini
            properties = {}
            for idx, line in enumerate(lines, 1):
                key = f"L{idx:02d}"
                # Only include description on the first line to avoid repetition
                if idx == 1:
                    properties[key] = {
                        "type": "string",
                        "description": f"Corrected transcription text for line {idx}. Only allowed characters: A-Z, a-z, & (ampersand), ' (apostrophe), ¶ (pilcrow), and spaces. Return abbreviated transcription, not expanded."
                    }
                else:
                    properties[key] = {
                        "type": "string"
                    }
            
            response_schema = {
                "type": "object",
                "properties": properties
            }
            
            # Load existing line images from Pylaia processing
            # Line images are stored in htr_work/{basename}/lines/{line_id}.png
            work_dir = os.path.join(BOOTSTRAP_DATA_DIR, "htr_work", basename)
            lines_dir = os.path.join(work_dir, "lines")
            
            line_image_paths = []
            line_image_data = []
            
            for idx, line in enumerate(lines, 1):
                key = f"L{idx:02d}"
                line_id = line.get("line_id")
                
                if not line_id:
                    logger.warning(f"Skipping line {key}: missing line_id")
                    line_image_paths.append(None)
                    line_image_data.append(None)
                    continue
                
                # Find existing line image
                line_image_path = os.path.join(lines_dir, f"{line_id}.png")
                
                if not os.path.exists(line_image_path):
                    logger.warning(f"Line image not found for {key} (line_id: {line_id}): {line_image_path}")
                    line_image_paths.append(None)
                    line_image_data.append(None)
                    continue
                
                try:
                    # Read existing line image
                    with open(line_image_path, 'rb') as f:
                        image_data = f.read()
                    
                    line_image_paths.append(line_image_path)
                    line_image_data.append(image_data)
                    logger.debug(f"Loaded existing line image for {key} (line_id: {line_id})")
                    
                except Exception as e:
                    logger.error(f"Error loading line image for {key} (line_id: {line_id}): {e}")
                    line_image_paths.append(None)
                    line_image_data.append(None)
            
            # Format schema for prompt
            schema_json = json.dumps(response_schema, indent=2, ensure_ascii=False)
            resolution_info = f"{image_width}x{image_height} pixels" if image_width and image_height else "unknown resolution"
            
            # Build prompt
            prompt = f"""Please correct the transcription using the imperfect HTR transcript, the individual line images provided, and the index data from the database.

## Image Information:
- **Original Image Resolution**: {resolution_info}
- **Number of Lines**: {len(lines)}
- **Line Images**: Each line has been extracted and preprocessed as a separate image at 128px height

## Important Context:
**BOUNDING BOXES ARE FOR DOCUMENT LAYOUT CONTEXT**: The bounding box coordinates [ymin, xmin, ymax, xmax] provided for each line are from the ORIGINAL full-page image. These coordinates are provided to give you a sense of the document layout and help with semantic analysis (understanding context, relationships between lines, etc.). The coordinates do NOT apply to the individual line images you are viewing - they refer to positions in the original full-page image.

**TRANSCRIBE FROM LINE IMAGES**: You should transcribe the text visible in each individual line image provided. The line images have been preprocessed and extracted from the original image, so focus on what you see in each line image.

**LINE IMAGE ORDER**: The line images are provided in the SAME ORDER as the line data below. The first image corresponds to L01, the second image to L02, and so on. Each line image directly corresponds to the line with the matching key (L01, L02, etc.) in the line data.

## Critical Instructions:
1. **TRANSCRIBE FROM LINE IMAGES**: Transcribe the text visible in each individual line image provided
2. **IMAGE-TO-LABEL CORRESPONDENCE**: The line images are provided in order: first image = L01, second image = L02, third image = L03, etc. Match each image to its corresponding line key
3. **BOUNDING BOXES FOR CONTEXT**: The bounding box coordinates are provided for document layout context and semantic analysis - they refer to the original full-page image, not the line images
4. **ONE LINE PER IMAGE**: Each line image corresponds to exactly one line of text - transcribe only what is visible in that specific line image
5. **EMPTY LINES**: If a line image contains no text (e.g., blank space, margin, or decorative element), return an empty string ("") for that line key in your JSON response

## Transcription Rules:
1. **CRITICAL: RETURN ABBREVIATED TRANSCRIPTION, NOT EXPANDED**: You MUST return the abbreviated transcription exactly as it appears in the manuscript, with abbreviations marked by apostrophes. DO NOT expand abbreviations into full words. For example, if the manuscript shows "p'", return "p'" NOT "per". If it shows "q'", return "q'" NOT "quod". The transcription must match the abbreviated form visible in the image.
2. Correct the transcription for each line, preserving abbreviations exactly as written
3. Use a single straight apostrophe (') to indicate abbreviated letters
4. Use ONLY these characters: A-Z, a-z, &, ' and pilcrow (¶)
5. **DO NOT EXPAND ABBREVIATIONS** - mark them with apostrophes and keep them in their abbreviated form
6. Return a JSON object with the same keys as the input lines, mapping each key to the corrected text
7. For lines with no text in the bounding box, use an empty string: "L05": ""

## Index Data from Database:
{index_json}

## HTR Lines with Bounding Boxes:
Each entry includes:
- "key": Line identifier (e.g., "L01", "L02")
- "bbox": Bounding box coordinates [ymin, xmin, ymax, xmax] from the original full-page image (for layout context only)
- "htr_text": The imperfect HTR transcription for reference

{line_data_json}

Return a JSON object mapping each line key to its corrected text. Transcribe from the individual line images provided. If a line image contains no text, return an empty string for that line.

**CRITICAL REMINDER**: You MUST return the ABBREVIATED transcription, NOT the expanded form. Keep all abbreviations as they appear in the manuscript with apostrophes (e.g., "p'", "q'", "n'", etc.). DO NOT expand them into full words.

**IMPORTANT**: Return a single JSON object (not an array). Example format:
{{
  "L01": "corrected text here",
  "L02": "another corrected line",
  "L03": "third line",
  "L04": "",
  ...
}}

The response must be a JSON object where each key is a line identifier (L01, L02, etc.) and each value is the corrected transcription string in ABBREVIATED form (or empty string if no text is present in the bounding box).

## Expected Response Schema:
The response must conform to the following JSON schema:
{schema_json}

**Note**: The schema defines the structure of your response. Each property key (L01, L02, etc.) corresponds to a line identifier, and the value must be a string containing the corrected transcription in abbreviated form."""
                
            # Prepare image for Gemini API
            if not os.path.exists(image_path):
                logger.warning(f"Image not found for request: {image_path}")
                return None
            
            batch_key = _sanitize_batch_key(basename)
            
            return {
                "key": batch_key,
                "image_path": image_path,
                "lines": lines,
                "line_image_paths": line_image_paths,
                "line_image_data": line_image_data,
                "prompt": prompt,
                "mime_type": mime_type,
                "response_schema": response_schema
            }
            
        except Exception as e:
            logger.error(f"Error preparing batch request for {image_path}: {e}", exc_info=True)
            return None
    
    def _run_batch_corrections(self, batch_requests: List[Dict[str, Any]], batch_id: str) -> Dict[str, Any]:
        """
        Submit batch job for Gemini 3 corrections.
        
        Args:
            batch_requests: List of request dictionaries from _prepare_batch_request
            batch_id: Unique identifier for this batch
            
        Returns:
            Dictionary mapping batch keys to result dictionaries containing:
            - "corrections": Dictionary mapping line keys to corrected text
            - "raw_thoughts": Full raw response text including thoughts
            - "token_usage": Token usage information
        """
        if not batch_requests:
            return {}
        
        self.stats["gemini_correction_calls"] += len(batch_requests)
        
        # Create a unique state filename for this batch
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', batch_id)
        state_file = os.path.join(BOOTSTRAP_WORK_DIR, f"batch_state_{safe_id}.json")
        batch_job = None
        
        # 1. ATTEMPT RESUME
        if os.path.exists(state_file) and not self.force:
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                job_name = state.get('job_name')
                if job_name:
                    logger.info(f"[{batch_id}] Found interrupted state. Resuming Job: {job_name}")
                    try:
                        batch_job = self.client.batches.get(name=job_name)
                        logger.info(f"[{batch_id}] Successfully resumed batch job: {batch_job.name}, state: {batch_job.state.name}")
                    except Exception as e:
                        logger.warning(f"[{batch_id}] Batch job {job_name} not found or inaccessible: {e}")
                        batch_job = None
                else:
                    logger.warning(f"[{batch_id}] State file exists but no job_name found")
                    batch_job = None
            except Exception as e:
                logger.warning(f"[{batch_id}] Failed to resume state (creating new): {e}")
                batch_job = None
        
        # 2. CREATE NEW BATCH
        if not batch_job:
            timestamp = int(time.time())
            jsonl_filename = os.path.join(BOOTSTRAP_WORK_DIR, f"batch_{safe_id}_{timestamp}.jsonl")
            
            logger.info(f"[{batch_id}] Creating Batch JSONL with {len(batch_requests)} requests...")
            
            # Write JSONL file
            with open(jsonl_filename, 'w', encoding='utf-8') as f:
                for item in batch_requests:
                    line = {
                        "key": item['key'],
                        "request": item['request']
                    }
                    f.write(json.dumps(line) + "\n")
            
            logger.info(f"[{batch_id}] Uploading JSONL...")
            batch_input_file = self.client.files.upload(
                file=jsonl_filename,
                config=types.UploadFileConfig(mime_type='application/jsonl')
            )
            
            logger.info(f"[{batch_id}] Submitting Batch Job...")
            batch_job = self.client.batches.create(
                model=MODEL_VISION,
                src=batch_input_file.name,
            )
            logger.info(f"[{batch_id}] Job Created: {batch_job.name}")
            
            # Save state with batch ID for resumption
            try:
                with open(state_file, 'w') as f:
                    json.dump({
                        "job_name": batch_job.name,
                        "batch_id": batch_id,
                        "timestamp": timestamp,
                        "request_count": len(batch_requests),
                        "request_keys": [req['key'] for req in batch_requests]
                    }, f, indent=2)
                logger.info(f"[{batch_id}] Saved batch state to {state_file}")
            except Exception as e:
                logger.error(f"Could not save batch state: {e}")
        
        # 3. WAIT FOR COMPLETION (with resumable polling)
        completed_states = ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']
        poll_count = 0
        start_time = time.time()
        logger.info(f"[{batch_id}] Batch job submitted. Job name: {batch_job.name}")
        logger.info(f"[{batch_id}] Waiting for batch job to complete (this may take several minutes)...")
        
        while batch_job.state.name not in completed_states:
            poll_count += 1
            elapsed = int(time.time() - start_time)
            logger.info(f"[{batch_id}] Status: {batch_job.state.name} (poll #{poll_count}, {elapsed}s elapsed). Waiting 60s...")
            
            # Update state file with current status
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                state['last_poll'] = time.time()
                state['last_status'] = batch_job.state.name
                state['poll_count'] = poll_count
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not update batch state: {e}")
            
            time.sleep(60)
            try:
                batch_job = self.client.batches.get(name=batch_job.name)
            except Exception as e:
                logger.error(f"[{batch_id}] Error polling batch job: {e}")
                logger.info(f"[{batch_id}] Batch state saved. Resume by running again with same batch_id")
                raise
        
        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            logger.error(f"[{batch_id}] Batch Job Failed: {batch_job.state.name}")
            return {}
        
        # 4. DOWNLOAD RESULTS
        logger.info(f"[{batch_id}] Downloading results from batch job...")
        results_map = {}
        total_thoughts_tokens = 0
        total_response_tokens = 0
        total_prompt_tokens = 0
        
        if batch_job.dest and batch_job.dest.file_name:
            content = self.client.files.download(file=batch_job.dest.file_name).decode('utf-8')
            
            # Parse JSONL results
            for line in content.strip().split('\n'):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    key = result.get('key')
                    response_data = result.get('response', {})
                    
                    if not key or not response_data:
                        continue
                    
                    # Extract usage metadata
                    um = None
                    if 'usageMetadata' in response_data:
                        um = response_data['usageMetadata']
                    elif 'usage_metadata' in response_data:
                        um = response_data['usage_metadata']
                    elif 'usageMetadata' in result:
                        um = result['usageMetadata']
                    elif 'usage_metadata' in result:
                        um = result['usage_metadata']
                    
                    # Extract token usage
                    def safe_int(value):
                        if value is None:
                            return 0
                        try:
                            return int(value)
                        except (ValueError, TypeError):
                            return 0
                    
                    item_token_usage = {
                        "prompt_tokens": 0,
                        "response_tokens": 0,
                        "thoughts_tokens": 0,
                        "total_tokens": 0
                    }
                    
                    if um:
                        thoughts_tokens = safe_int(um.get('thoughts_token_count') or um.get('thoughtsTokenCount', 0))
                        response_tokens = safe_int(um.get('candidates_token_count') or um.get('candidatesTokenCount', 0))
                        prompt_tokens = safe_int(um.get('prompt_token_count') or um.get('promptTokenCount', 0))
                        total = safe_int(um.get('total_token_count') or um.get('totalTokenCount', 0))
                        
                        item_token_usage = {
                            "prompt_tokens": prompt_tokens,
                            "response_tokens": response_tokens,
                            "thoughts_tokens": thoughts_tokens,
                            "total_tokens": total
                        }
                        
                        total_thoughts_tokens += thoughts_tokens
                        total_response_tokens += response_tokens
                        total_prompt_tokens += prompt_tokens
                    
                    # Extract full raw text including thoughts
                    raw_thoughts = ""
                    text = None
                    
                    if 'candidates' in response_data and response_data['candidates']:
                        candidate = response_data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            # Join all parts to get full text including thoughts
                            parts = candidate['content']['parts']
                            all_text_parts = []
                            for part in parts:
                                if 'text' in part and part['text']:
                                    all_text_parts.append(part['text'])
                            
                            # Combine all text parts
                            raw_thoughts = "".join(all_text_parts)
                            
                            # For response text, use the last part (usually the actual response after thoughts)
                            # But if there's only one part or response_tokens is 0, use all text
                            if all_text_parts:
                                if len(all_text_parts) > 1 and item_token_usage.get('response_tokens', 0) > 0:
                                    # Use last part as response (after thoughts)
                                    text = all_text_parts[-1]
                                else:
                                    # Use all text if only one part or no response tokens
                                    text = raw_thoughts
                    elif 'text' in response_data:
                        text = response_data['text']
                        raw_thoughts = text
                    
                    if not text and not raw_thoughts:
                        logger.warning(f"[{batch_id}] No text found in response for {key}")
                        continue
                    
                    # Parse JSON from response text or raw_thoughts
                    # Since response_mime_type is "application/json", the response should be directly parseable
                    corrections = {}
                    json_text = None
                    
                    # First, try to parse the response text directly as JSON (since response_mime_type is set)
                    # When response_mime_type is "application/json", the JSON should be directly in the text
                    if text:
                        try:
                            # Try parsing directly - when response_mime_type is "application/json", 
                            # the response should be valid JSON
                            parsed_json = json.loads(text.strip())
                            if isinstance(parsed_json, dict) and any(k.startswith('L') for k in parsed_json.keys()):
                                corrections = parsed_json
                                logger.debug(f"[{batch_id}] Successfully parsed JSON directly from response for {key}")
                        except (json.JSONDecodeError, AttributeError):
                            # If direct parsing fails, try extraction methods
                            pass
                    
                    # Helper: Extract JSON object from text, handling nested structures
                    def extract_json_from_text(content: str) -> Optional[str]:
                        if not content:
                            return None
                        
                        # Try markdown code block first
                        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            return json_match.group(1).strip()
                        
                        # Try to find JSON object with balanced braces
                        brace_positions = []
                        for i, char in enumerate(content):
                            if char == '{':
                                brace_positions.append(i)
                            elif char == '}':
                                if brace_positions:
                                    start = brace_positions.pop()
                                    if not brace_positions:  # Found complete top-level object
                                        candidate = content[start:i+1]
                                        if '"L' in candidate or '"l' in candidate.lower():
                                            return candidate
                        
                        # If no balanced JSON found, try simpler patterns
                        json_match = re.search(r'\{[^{}]*"L\d+".*?\}', content, re.DOTALL)
                        if json_match:
                            return json_match.group(0)
                        
                        # Try to find any JSON object (last resort - might be incomplete)
                        start_idx = content.find('{')
                        if start_idx != -1:
                            brace_count = 0
                            for i in range(start_idx, len(content)):
                                if content[i] == '{':
                                    brace_count += 1
                                elif content[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        candidate = content[start_idx:i+1]
                                        try:
                                            json.loads(candidate)
                                            return candidate
                                        except Exception:
                                            pass
                        
                        return None
                    
                    json_text = None
                    
                    # If direct parsing didn't work, try extraction methods
                    if not corrections:
                        # First try to extract JSON from the text (response after thoughts)
                        if text:
                            json_text = extract_json_from_text(text)
                        
                        # If no JSON found in text, try extracting from raw_thoughts
                        if not json_text and raw_thoughts:
                            json_text = extract_json_from_text(raw_thoughts)
                        
                        # Try to parse the extracted JSON (only if we didn't already get corrections from direct parsing)
                        if json_text and not corrections:
                            try:
                                parsed_json = json.loads(json_text)
                                # Handle both array format [{"L01": "..."}, {"L02": "..."}] and object format {"L01": "...", "L02": "..."}
                                if isinstance(parsed_json, list):
                                    # Convert array of objects to single object
                                    corrections = {}
                                    for item in parsed_json:
                                        if isinstance(item, dict):
                                            corrections.update(item)
                                else:
                                    corrections = parsed_json
                            except json.JSONDecodeError as e:
                                logger.warning(f"[{batch_id}] Failed to parse JSON for {key}: {e}")
                                logger.debug(f"[{batch_id}] JSON text (first 500 chars): {json_text[:500]}")
                                # Try to extract just the JSON part more carefully
                                # Look for the largest valid JSON object
                                try:
                                    # Find all potential JSON objects
                                    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_text, re.DOTALL)
                                    for obj in reversed(json_objects):  # Try largest first
                                        try:
                                            parsed_obj = json.loads(obj)
                                            # Handle both array and object formats
                                            if isinstance(parsed_obj, list):
                                                corrections = {}
                                                for item in parsed_obj:
                                                    if isinstance(item, dict):
                                                        corrections.update(item)
                                            else:
                                                corrections = parsed_obj
                                            
                                            if corrections and isinstance(corrections, dict) and any(k.startswith('L') for k in corrections.keys()):
                                                logger.info(f"[{batch_id}] Successfully parsed JSON from {key} using fallback extraction")
                                                break
                                        except json.JSONDecodeError:
                                            continue
                                except Exception:
                                    pass
                                
                                if not corrections:
                                    logger.warning(f"[{batch_id}] Could not extract valid JSON for {key}")
                                    # Save raw thoughts for debugging
                                    if raw_thoughts:
                                        debug_dir = os.path.join(BOOTSTRAP_DATA_DIR, "debug_failed_json")
                                        os.makedirs(debug_dir, exist_ok=True)
                                        debug_path = os.path.join(debug_dir, f"{key}_raw_thoughts.log")
                                        with open(debug_path, 'w', encoding='utf-8') as f:
                                            f.write(raw_thoughts)
                                        logger.info(f"[{batch_id}] Saved raw thoughts to {debug_path} for debugging")
                        else:
                            logger.warning(f"[{batch_id}] No JSON text found in response or thoughts for {key}")
                            # Log what we actually received
                            if text:
                                logger.debug(f"[{batch_id}] Response text (first 200 chars): {text[:200]}")
                            if raw_thoughts:
                                logger.debug(f"[{batch_id}] Raw thoughts (first 200 chars): {raw_thoughts[:200]}")
                                # Check if there's any JSON-like content
                                if '{' in raw_thoughts or '[' in raw_thoughts:
                                    logger.debug(f"[{batch_id}] Found braces in raw_thoughts, but extraction failed")
                                # Save raw thoughts for debugging
                                debug_dir = os.path.join(BOOTSTRAP_DATA_DIR, "debug_failed_json")
                                os.makedirs(debug_dir, exist_ok=True)
                                debug_path = os.path.join(debug_dir, f"{key}_raw_thoughts.log")
                                with open(debug_path, 'w', encoding='utf-8') as f:
                                    f.write(raw_thoughts)
                                logger.info(f"[{batch_id}] Saved raw thoughts to {debug_path} for debugging")
                            else:
                                logger.warning(f"[{batch_id}] No text or raw_thoughts available for {key}")
                    
                    results_map[key] = {
                        "corrections": corrections,
                        "raw_thoughts": raw_thoughts,
                        "token_usage": item_token_usage
                    }
                    
                    if item_token_usage.get('thoughts_tokens', 0) > 0:
                        logger.info(f"[{batch_id}] {key}: {item_token_usage.get('thoughts_tokens', 0):,} thoughts tokens, {item_token_usage.get('response_tokens', 0):,} response tokens")
                    
                except Exception as e:
                    logger.error(f"[{batch_id}] Error parsing result line: {e}", exc_info=True)
        
        logger.info(f"[{batch_id}] Batch complete: {total_prompt_tokens:,} prompt, {total_response_tokens:,} response, {total_thoughts_tokens:,} thoughts tokens")
        
        # Note: Cleanup of uploaded files is done by the caller after results are saved to disk
        
        # Cleanup state file on success
        if os.path.exists(state_file):
            try:
                os.remove(state_file)
                logger.info(f"[{batch_id}] Removed batch state file")
            except Exception as e:
                logger.warning(f"Could not remove state file: {e}")
        
        return results_map

    def _run_batch_corrections_claude(
        self, batch_requests: List[Dict[str, Any]], batch_id: str
    ) -> Dict[str, Any]:
        """
        Process corrections using Gemini 3.0 Flash in non-batch mode (replaces Claude Opus 4.5).

        Args:
            batch_requests: List of request dictionaries from _prepare_batch_request
            batch_id: Unique identifier for this batch

        Returns:
            Dictionary mapping batch keys to result dictionaries containing:
            - "corrections": Dictionary mapping line keys to corrected text
            - "raw_thoughts": Full raw response text
            - "token_usage": Token usage information
        """
        if not batch_requests:
            return {}

        self.stats["gemini_correction_calls"] += len(batch_requests)
        
        logger.info(f"[{batch_id}] Processing {len(batch_requests)} images with Gemini 3.0 Flash (non-batch mode)...")
        
        results_map: Dict[str, Any] = {}
        total_prompt_tokens = 0
        total_output_tokens = 0
        
        for idx, request_data in enumerate(batch_requests, 1):
            key = request_data.get('key', f'item_{idx}')
            image_path = request_data.get('image_path')
            prompt = request_data.get('prompt')
            mime_type = request_data.get('mime_type', 'image/png')
            response_schema = request_data.get('response_schema')
            line_image_data = request_data.get('line_image_data', [])
            line_image_paths = request_data.get('line_image_paths', [])
            temp_dir = request_data.get('temp_dir')
            
            if not prompt:
                logger.warning(f"[{batch_id}] Skipping {key}: missing prompt")
                continue
            
            if not response_schema:
                logger.warning(f"[{batch_id}] Skipping {key}: missing response_schema")
                continue
            
            # Validate response_schema structure
            if not isinstance(response_schema, dict) or "properties" not in response_schema:
                logger.warning(f"[{batch_id}] Skipping {key}: invalid response_schema structure")
                continue
            
            num_properties = len(response_schema.get("properties", {}))
            if num_properties == 0:
                logger.warning(f"[{batch_id}] Skipping {key}: response_schema has no properties")
                continue
            
            # Log schema info for debugging
            logger.debug(f"[{batch_id}] Processing {idx}/{len(batch_requests)}: {key} (schema has {num_properties} properties, prompt length: {len(prompt)} chars, {len(line_image_data)} line images)")
            
            # Skip response_schema for very large requests (>200 properties) as it may cause API errors
            # We'll rely on prompt instructions and parse JSON from text response
            use_schema = num_properties <= 200
            if not use_schema:
                logger.info(f"[{batch_id}] Skipping response_schema for {key} (too many properties: {num_properties}), will parse JSON from text")
                response_schema = None
            
            try:
                # Prepare content parts with individual line images
                parts = [types.Part.from_text(text=prompt)]
                
                # Add each line image
                valid_line_count = 0
                for line_idx, (line_img_data, line_img_path) in enumerate(zip(line_image_data, line_image_paths)):
                    if line_img_data is not None and line_img_path is not None:
                        # Validate image size (Gemini has limits)
                        if len(line_img_data) > 20 * 1024 * 1024:  # 20MB limit
                            logger.warning(f"[{batch_id}] Line image {line_idx+1} for {key} is too large ({len(line_img_data) / 1024 / 1024:.2f}MB), skipping")
                            continue
                        
                        # Add line image part
                        line_image_part = types.Part.from_bytes(data=line_img_data, mime_type="image/png")
                        parts.append(line_image_part)
                        valid_line_count += 1
                
                if valid_line_count == 0:
                    logger.warning(f"[{batch_id}] No valid line images for {key}, skipping")
                    continue
                
                logger.debug(f"[{batch_id}] Added {valid_line_count} line images for {key}")
                
                # Call Gemini API with temperature 0.0
                # Try with thinking_config first, fall back without it if not supported
                config_params = {
                    "temperature": 0.0,
                    "max_output_tokens": 8192,
                    "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,  # Set medium resolution for line images
                }
                
                # Only include response_schema and response_mime_type if schema is being used
                if response_schema:
                    config_params["response_mime_type"] = "application/json"
                    config_params["response_schema"] = response_schema
                else:
                    # Without schema, still request JSON but let the model return it as text
                    config_params["response_mime_type"] = "application/json"
                
                # Try with thinking_config first (may not be supported for all models)
                try:
                    config = types.GenerateContentConfig(
                        **config_params,
                        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level="LOW")
                    )
                    response = self.client.models.generate_content(
                        model=MODEL_VISION,
                        contents=parts,
                        config=config
                    )
                except ClientError as e:
                    # Log detailed error information
                    error_details = str(e)
                    # Try to extract more details from ClientError
                    if hasattr(e, 'response_json'):
                        try:
                            error_details = json.dumps(e.response_json, indent=2)
                        except:
                            pass
                    elif hasattr(e, 'status_code'):
                        error_details = f"Status {e.status_code}: {error_details}"
                    
                    logger.warning(f"[{batch_id}] Error with thinking_config for {key}: {error_details}")
                    
                    # If thinking_config is not supported, retry without it
                    if "INVALID_ARGUMENT" in str(e) or (hasattr(e, 'status_code') and e.status_code == 400):
                        logger.info(f"[{batch_id}] Retrying without thinking_config for {key}")
                        try:
                            config = types.GenerateContentConfig(**config_params)
                            response = self.client.models.generate_content(
                                model=MODEL_VISION,
                                contents=parts,
                                config=config
                            )
                        except ClientError as retry_e:
                            # If it still fails, try without response_schema (unstructured output)
                            retry_error = str(retry_e)
                            if hasattr(retry_e, 'response_json'):
                                try:
                                    retry_error = json.dumps(retry_e.response_json, indent=2)
                                except:
                                    pass
                            logger.warning(f"[{batch_id}] Failed with response_schema for {key}, trying without structured output: {retry_error}")
                            
                            # Try one more time without response_schema and without response_mime_type
                            # (response_mime_type="application/json" may require a schema)
                            try:
                                config_no_schema = types.GenerateContentConfig(
                                    temperature=0.0,
                                    max_output_tokens=8192,
                                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,  # Set medium resolution for line images
                                )
                                response = self.client.models.generate_content(
                                    model=MODEL_VISION,
                                    contents=parts,
                                    config=config_no_schema
                                )
                                logger.info(f"[{batch_id}] Successfully called API without response_schema for {key}")
                            except ClientError as final_e:
                                # Final failure - log and re-raise
                                final_error = str(final_e)
                                if hasattr(final_e, 'response_json'):
                                    try:
                                        final_error = json.dumps(final_e.response_json, indent=2)
                                    except:
                                        pass
                                logger.error(f"[{batch_id}] Final failure for {key} (even without schema): {final_error}")
                                raise
                    else:
                        # Re-raise if it's a different error
                        raise
                
                # Extract response text
                response_text = ""
                parsed_data = None
                
                if hasattr(response, 'text') and response.text:
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                            elif hasattr(part, 'inline_data'):
                                try:
                                    import base64
                                    parsed_data = json.loads(base64.b64decode(part.inline_data.data).decode('utf-8'))
                                except Exception:
                                    pass
                
                # Parse JSON corrections
                corrections = {}
                if parsed_data:
                    corrections = parsed_data if isinstance(parsed_data, dict) else {}
                elif response_text.strip():
                    try:
                        parsed_json = json.loads(response_text.strip())
                        corrections = parsed_json if isinstance(parsed_json, dict) else {}
                    except json.JSONDecodeError:
                        # Try to extract JSON from text
                        import re
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                        if json_match:
                            try:
                                corrections = json.loads(json_match.group(0))
                            except json.JSONDecodeError:
                                logger.warning(f"[{batch_id}] Failed to parse JSON for {key}")
                
                # Extract token usage
                usage_metadata = None
                if hasattr(response, 'usage_metadata'):
                    usage_metadata = response.usage_metadata
                elif hasattr(response, 'usageMetadata'):
                    usage_metadata = response.usageMetadata
                
                item_token_usage = {
                    "prompt_tokens": 0,
                    "response_tokens": 0,
                    "thoughts_tokens": 0,
                    "total_tokens": 0
                }
                
                if usage_metadata:
                    item_token_usage = {
                        "prompt_tokens": int(getattr(usage_metadata, 'prompt_token_count', 0) or 0),
                        "response_tokens": int(getattr(usage_metadata, 'candidates_token_count', 0) or 0),
                        "thoughts_tokens": int(getattr(usage_metadata, 'thoughts_token_count', 0) or 0),
                        "total_tokens": int(getattr(usage_metadata, 'total_token_count', 0) or 0)
                    }
                    total_prompt_tokens += item_token_usage["prompt_tokens"]
                    total_output_tokens += item_token_usage["response_tokens"]
                
                results_map[key] = {
                    "corrections": corrections,
                    "raw_thoughts": response_text,
                    "token_usage": item_token_usage
                }
                
                logger.info(f"[{batch_id}] Completed {key}: {len(corrections)} corrections")
                
            except Exception as e:
                logger.error(f"[{batch_id}] Error processing {key}: {e}", exc_info=True)
                results_map[key] = {
                    "corrections": {},
                    "raw_thoughts": "",
                    "token_usage": {
                        "prompt_tokens": 0,
                        "response_tokens": 0,
                        "thoughts_tokens": 0,
                        "total_tokens": 0
                    }
                }
        
        logger.info(
            f"[{batch_id}] Completed processing. Total tokens: "
            f"{total_prompt_tokens + total_output_tokens} "
            f"(prompt={total_prompt_tokens}, output={total_output_tokens})"
        )
        
        return results_map
    
    def _run_batch_corrections_gemini_batch(
        self, batch_requests: List[Dict[str, Any]], batch_id: str
    ) -> Dict[str, Any]:
        """
        Process corrections using Gemini 3.0 Flash batch API (true batch mode with 10 images).
        
        This method uses Gemini's batch API to process multiple images in a single batch job,
        which is more efficient than processing them individually.

        Args:
            batch_requests: List of request dictionaries from _prepare_batch_request
            batch_id: Unique identifier for this batch

        Returns:
            Dictionary mapping batch keys to result dictionaries containing:
            - "corrections": Dictionary mapping line keys to corrected text
            - "raw_thoughts": Full raw response text
            - "token_usage": Token usage information
        """
        if not batch_requests:
            return {}
        
        import tempfile
        import base64
        from pathlib import Path

        self.stats["gemini_correction_calls"] += len(batch_requests)
        
        logger.info(f"[{batch_id}] Processing {len(batch_requests)} images with Gemini 3.0 Flash batch API...")
        
        # Create temporary directory for batch files
        temp_dir = Path(tempfile.mkdtemp())
        jsonl_file = temp_dir / "batch_requests.jsonl"
        
        try:
            # Prepare batch requests in JSONL format
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for idx, request_data in enumerate(batch_requests):
                    key = request_data.get('key', f'item_{idx}')
                    image_path = request_data.get('image_path')
                    prompt = request_data.get('prompt')
                    mime_type = request_data.get('mime_type', 'image/jpeg')
                    response_schema = request_data.get('response_schema')
                    
                    if not prompt or not image_path:
                        logger.warning(f"[{batch_id}] Skipping {key}: missing prompt or image_path")
                        continue
                    
                    if not response_schema:
                        logger.warning(f"[{batch_id}] Skipping {key}: missing response_schema")
                        continue
                    
                    # Validate response_schema structure
                    if not isinstance(response_schema, dict) or "properties" not in response_schema:
                        logger.warning(f"[{batch_id}] Skipping {key}: invalid response_schema structure")
                        continue
                    
                    num_properties = len(response_schema.get("properties", {}))
                    if num_properties == 0:
                        logger.warning(f"[{batch_id}] Skipping {key}: response_schema has no properties")
                        continue
                    
                    # Skip response_schema for very large requests (>200 properties)
                    use_schema = num_properties <= 200
                    if not use_schema:
                        logger.info(f"[{batch_id}] Skipping response_schema for {key} (too many properties: {num_properties}), will parse JSON from text")
                        response_schema = None
                    
                    try:
                        # Load line images and upload them
                        line_image_data = request_data.get('line_image_data', [])
                        line_image_paths = request_data.get('line_image_paths', [])
                        
                        if not line_image_data or not any(line_image_data):
                            logger.warning(f"[{batch_id}] No line images for {key}, skipping")
                            continue
                        
                        # Helper function to upload a single image and wait for it to be active
                        def upload_single_image(line_idx: int, line_img_path: str, line_img_data: bytes) -> Optional[Tuple[int, Any]]:
                            """Upload a single line image and return (index, uploaded_image) or None if failed."""
                            if line_img_data is None or line_img_path is None:
                                return None
                            
                            # Validate image size (Gemini has limits)
                            if len(line_img_data) > 20 * 1024 * 1024:  # 20MB limit
                                logger.warning(f"[{batch_id}] Line image {line_idx+1} for {key} is too large ({len(line_img_data) / 1024 / 1024:.2f}MB), skipping")
                                return None
                            
                            # Upload line image file
                            logger.debug(f"[{batch_id}] Uploading line image {line_idx+1} for {key}...")
                            try:
                                uploaded_image = self.client.files.upload(
                                    file=line_img_path,
                                    config=types.UploadFileConfig(mime_type="image/png")
                                )
                                
                                # Wait for file to be active (required before using in batch)
                                max_wait = 60  # Wait up to 60 seconds
                                wait_count = 0
                                while uploaded_image.state.name != "ACTIVE" and wait_count < max_wait:
                                    time.sleep(1)
                                    uploaded_image = self.client.files.get(name=uploaded_image.name)
                                    wait_count += 1
                                
                                if uploaded_image.state.name != "ACTIVE":
                                    logger.warning(f"[{batch_id}] Line image {line_idx+1} for {key} not active after upload, skipping")
                                    return None
                                
                                return (line_idx, uploaded_image)
                                
                            except Exception as e:
                                logger.warning(f"[{batch_id}] Error uploading line image {line_idx+1} for {key}: {e}")
                                return None
                        
                        # Upload all line images in parallel
                        parts = [{"text": prompt}]
                        uploaded_line_images = []
                        
                        # Prepare list of images to upload with their indices
                        upload_tasks = [
                            (line_idx, line_img_path, line_img_data)
                            for line_idx, (line_img_data, line_img_path) in enumerate(zip(line_image_data, line_image_paths))
                            if line_img_data is not None and line_img_path is not None
                        ]
                        
                        # Use ThreadPoolExecutor to upload images in parallel
                        max_workers = min(10, len(upload_tasks))  # Limit to 10 concurrent uploads
                        if upload_tasks:
                            logger.debug(f"[{batch_id}] Uploading {len(upload_tasks)} line images for {key} in parallel (max {max_workers} workers)...")
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                # Submit all upload tasks
                                future_to_task = {
                                    executor.submit(upload_single_image, idx, path, data): (idx, path, data)
                                    for idx, path, data in upload_tasks
                                }
                                
                                # Collect results as they complete, preserving order
                                upload_results = {}
                                for future in as_completed(future_to_task):
                                    result = future.result()
                                    if result is not None:
                                        line_idx, uploaded_image = result
                                        upload_results[line_idx] = uploaded_image
                            
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
                        
                        if len(parts) == 1:  # Only prompt, no images
                            logger.warning(f"[{batch_id}] No valid line images uploaded for {key}, skipping")
                            continue
                        
                        logger.debug(f"[{batch_id}] Uploaded {len(uploaded_line_images)} line images for {key}")
                        
                        # Build generation config
                        # For JSONL batch format, use generation_config (not 'config' like inline requests)
                        # Note: response_schema with dict format doesn't work in JSONL batch API
                        # (Pydantic models only work for inline requests, not JSONL files)
                        # We'll request JSON format and parse from text response
                        # Increase max_output_tokens significantly to handle large JSON responses
                        # Some images have 30+ lines, each with long transcriptions
                        gen_config = {
                            "temperature": 0.0,
                            "max_output_tokens": 32768,  # Increased from 8192 to handle large responses
                            "response_mime_type": "application/json",
                            "media_resolution": "MEDIA_RESOLUTION_MEDIUM",  # Set medium resolution for line images
                            "thinking_config": {
                                "include_thoughts": True,
                                "thinking_level": "LOW"
                            }
                        }
                        
                        # Don't include response_schema for JSONL batch API - it causes INVALID_ARGUMENT errors
                        # The batch API doesn't support dict-format schemas in JSONL files
                        # We'll parse JSON from the text response instead
                        
                        # Build request object in the correct format: {"key": "...", "request": {...}}
                        # The request should be a GenerateContentRequest
                        # Note: contents should be an array with parts, no "role" field needed
                        request_obj = {
                            "key": key,
                            "request": {
                                "contents": [{"parts": parts}],
                                "generation_config": gen_config
                            }
                        }
                        
                        # Log first request for debugging
                        if idx == 0:
                            # Log the full generation_config to see if response_schema is the issue
                            logger.debug(f"[{batch_id}] Generation config keys: {list(gen_config.keys())}")
                            if "response_schema" in gen_config:
                                schema_props_count = len(gen_config["response_schema"].get("properties", {}))
                                logger.debug(f"[{batch_id}] Response schema has {schema_props_count} properties")
                            logger.debug(f"[{batch_id}] Sample request structure (first request): {json.dumps(request_obj, indent=2)[:1000]}")
                        
                        f.write(json.dumps(request_obj) + "\n")
                        logger.debug(f"[{batch_id}] Added {key} to batch JSONL")
                        
                    except Exception as e:
                        logger.error(f"[{batch_id}] Error preparing request for {key}: {e}", exc_info=True)
                        continue
            
            # Check if we have any requests
            if jsonl_file.stat().st_size == 0:
                logger.warning(f"[{batch_id}] No valid requests in batch file")
                return {}
            
            # Save a copy of the JSONL for debugging
            debug_jsonl = temp_dir / "batch_requests_debug.jsonl"
            try:
                import shutil
                shutil.copy2(jsonl_file, debug_jsonl)
                logger.info(f"[{batch_id}] Saved debug JSONL to: {debug_jsonl}")
                
                # Also log the complete first request (not truncated)
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        try:
                            parsed = json.loads(first_line)
                            # Log the complete structure, especially file_data and generation_config
                            logger.debug(f"[{batch_id}] Complete first request structure:")
                            logger.debug(f"[{batch_id}] Keys in request: {list(parsed.get('request', {}).keys())}")
                            if 'contents' in parsed.get('request', {}):
                                contents = parsed['request']['contents']
                                if contents and 'parts' in contents[0]:
                                    parts = contents[0]['parts']
                                    logger.debug(f"[{batch_id}] Number of parts: {len(parts)}")
                                    for i, part in enumerate(parts):
                                        logger.debug(f"[{batch_id}] Part {i} keys: {list(part.keys())}")
                                        if 'file_data' in part:
                                            logger.debug(f"[{batch_id}] Part {i} file_data: {part['file_data']}")
                            if 'generation_config' in parsed.get('request', {}):
                                gen_config = parsed['request']['generation_config']
                                logger.debug(f"[{batch_id}] Generation config keys: {list(gen_config.keys())}")
                                if 'response_schema' in gen_config:
                                    schema = gen_config['response_schema']
                                    logger.debug(f"[{batch_id}] Response schema type: {schema.get('type')}, properties count: {len(schema.get('properties', {}))}")
                        except Exception as e:
                            logger.warning(f"[{batch_id}] Could not parse first line for debugging: {e}")
            except Exception as e:
                logger.warning(f"[{batch_id}] Could not save debug JSONL: {e}")
            
            # Upload JSONL file
            logger.info(f"[{batch_id}] Uploading batch file with {len(batch_requests)} requests...")
            uploaded_file = self.client.files.upload(
                file=str(jsonl_file),
                config=types.UploadFileConfig(mime_type='application/jsonl')
            )
            logger.info(f"[{batch_id}] File uploaded: {uploaded_file.name}")
            
            # Create batch job
            logger.info(f"[{batch_id}] Creating batch job...")
            batch_job = self.client.batches.create(
                model=MODEL_VISION,
                src=uploaded_file.name,
                config={
                    "display_name": f"bootstrap_corrections_{batch_id}"
                }
            )
            # Get state name helper
            def get_state_name(batch_job):
                if hasattr(batch_job.state, 'name'):
                    return batch_job.state.name
                elif isinstance(batch_job.state, str):
                    return batch_job.state
                else:
                    return str(batch_job.state)
            
            logger.info(f"[{batch_id}] Batch job created: {batch_job.name}, Status: {get_state_name(batch_job)}")
            
            # Poll for completion
            poll_count = 0
            max_polls = 1200  # 10 hours max (30 seconds * 1200 = 10 hours)
            completed_states = set([
                'JOB_STATE_SUCCEEDED',
                'JOB_STATE_FAILED',
                'JOB_STATE_CANCELLED',
                'JOB_STATE_EXPIRED',
            ])
            
            # Get state name - handle both string and object formats
            def get_state_name(batch_job):
                if hasattr(batch_job.state, 'name'):
                    return batch_job.state.name
                elif isinstance(batch_job.state, str):
                    return batch_job.state
                else:
                    return str(batch_job.state)
            
            current_state = get_state_name(batch_job)
            while current_state not in completed_states:
                time.sleep(30)
                poll_count += 1
                batch_job = self.client.batches.get(name=batch_job.name)
                current_state = get_state_name(batch_job)
                logger.info(f"[{batch_id}] Batch status: {current_state} (poll {poll_count}/{max_polls})")
                
                if poll_count >= max_polls:
                    logger.error(f"[{batch_id}] Batch job timed out after {max_polls} polls")
                    break
            
            if current_state != 'JOB_STATE_SUCCEEDED':
                logger.error(f"[{batch_id}] Batch job failed with state: {current_state}")
                # Return empty results for all requests
                results_map = {}
                for request_data in batch_requests:
                    key = request_data.get('key', 'unknown')
                    results_map[key] = {
                        "corrections": {},
                        "raw_thoughts": "",
                        "token_usage": {
                            "prompt_tokens": 0,
                            "response_tokens": 0,
                            "thoughts_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                return results_map
            
            # Download results
            logger.info(f"[{batch_id}] Downloading batch results...")
            results_file = temp_dir / "batch_results.jsonl"
            
            # For file-based batch jobs, results are in batch_job.dest.file_name
            if hasattr(batch_job, 'dest') and batch_job.dest and hasattr(batch_job.dest, 'file_name'):
                result_file_name = batch_job.dest.file_name
                logger.info(f"[{batch_id}] Results are in file: {result_file_name}")
                
                # Download the result file using the files API
                file_content_bytes = self.client.files.download(file=result_file_name)
                
                # Write to local file
                with open(results_file, 'wb') as f:
                    if isinstance(file_content_bytes, bytes):
                        f.write(file_content_bytes)
                    else:
                        # If it's a stream, write chunks
                        for chunk in file_content_bytes:
                            f.write(chunk)
            else:
                logger.error(f"[{batch_id}] No result file found in batch job destination")
                # Return empty results
                results_map = {}
                for request_data in batch_requests:
                    key = request_data.get('key', 'unknown')
                    results_map[key] = {
                        "corrections": {},
                        "raw_thoughts": "",
                        "token_usage": {
                            "prompt_tokens": 0,
                            "response_tokens": 0,
                            "thoughts_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                return results_map
            
            # Parse results
            results_map = {}
            total_prompt_tokens = 0
            total_output_tokens = 0
            total_thoughts_tokens = 0
            
            with open(results_file, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    if not line.strip():
                        continue
                    
                    line_count += 1
                    try:
                        result_obj = json.loads(line)
                        
                        # Log first result for debugging
                        if line_count == 1:
                            logger.debug(f"[{batch_id}] Sample result structure (first line): {json.dumps(result_obj, indent=2)[:1000]}")
                            logger.debug(f"[{batch_id}] Result keys: {list(result_obj.keys())}")
                        
                        # Batch API returns results with "key" field matching the input key
                        result_key = result_obj.get("key", result_obj.get("custom_id", "unknown"))
                        
                        # Check status - could be "status" or check for "response" field
                        status = result_obj.get("status")
                        has_response = "response" in result_obj
                        has_error = "error" in result_obj
                        
                        logger.debug(f"[{batch_id}] Result for {result_key}: status={status}, has_response={has_response}, has_error={has_error}")
                        
                        if status == "SUCCEEDED" or (has_response and not has_error):
                            response_obj = result_obj.get("response", {})
                            
                            # Extract response text and raw thoughts - try multiple formats
                            response_text = ""  # Just the JSON response (for parsing)
                            raw_thoughts = ""   # All text including thoughts (for saving to gemini_thoughts.log)
                            
                            # Try to get text from response - check multiple possible structures
                            # Format 1: response.text (direct)
                            if isinstance(response_obj, dict) and "text" in response_obj:
                                response_text = response_obj["text"]
                                raw_thoughts = response_obj["text"]
                            
                            # Format 2: response.candidates[0].content.parts[].text (most common for batch API)
                            elif isinstance(response_obj, dict) and "candidates" in response_obj and response_obj["candidates"]:
                                candidate = response_obj["candidates"][0]
                                if isinstance(candidate, dict):
                                    if "content" in candidate:
                                        content = candidate["content"]
                                        if isinstance(content, dict):
                                            if "parts" in content and content["parts"]:
                                                # Extract text from all parts (includes thoughts + response)
                                                all_parts_text = []
                                                for part in content["parts"]:
                                                    if isinstance(part, dict) and "text" in part:
                                                        part_text = part["text"]
                                                        all_parts_text.append(part_text)
                                                        raw_thoughts += part_text
                                                
                                                # For response_text, use the last part (usually the JSON response after thoughts)
                                                # But if there's only one part, use all text
                                                if all_parts_text:
                                                    if len(all_parts_text) > 1:
                                                        # Last part is usually the actual response (JSON)
                                                        response_text = all_parts_text[-1]
                                                    else:
                                                        # Only one part - use it for both
                                                        response_text = all_parts_text[0]
                                            elif "text" in content:
                                                response_text = content["text"]
                                                raw_thoughts = content["text"]
                                    # Some formats might have text directly in candidate
                                    elif "text" in candidate:
                                        response_text = candidate["text"]
                                        raw_thoughts = candidate["text"]
                            
                            # Format 3: Check if response_obj itself is a string (unlikely but possible)
                            elif isinstance(response_obj, str):
                                response_text = response_obj
                                raw_thoughts = response_obj
                            
                            # Format 4: Try accessing .text attribute if it's an object
                            elif hasattr(response_obj, 'text'):
                                response_text = str(response_obj.text)
                                raw_thoughts = str(response_obj.text)
                            
                            # If raw_thoughts is empty but response_text is not, use response_text
                            if not raw_thoughts and response_text:
                                raw_thoughts = response_text
                            
                            # Log if we couldn't extract text
                            if not response_text:
                                logger.warning(f"[{batch_id}] Could not extract response text for {result_key}. Response structure: {type(response_obj)}")
                                if isinstance(response_obj, dict):
                                    logger.debug(f"[{batch_id}] Response keys: {list(response_obj.keys())}")
                                    if "candidates" in response_obj:
                                        logger.debug(f"[{batch_id}] Candidates structure: {type(response_obj['candidates'])}, length: {len(response_obj['candidates']) if isinstance(response_obj['candidates'], list) else 'N/A'}")
                                        if response_obj["candidates"] and isinstance(response_obj["candidates"][0], dict):
                                            logger.debug(f"[{batch_id}] First candidate keys: {list(response_obj['candidates'][0].keys())}")
                                            if "content" in response_obj["candidates"][0]:
                                                logger.debug(f"[{batch_id}] Content keys: {list(response_obj['candidates'][0]['content'].keys())}")
                                                if "parts" in response_obj["candidates"][0]["content"]:
                                                    parts = response_obj["candidates"][0]["content"]["parts"]
                                                    logger.debug(f"[{batch_id}] Parts count: {len(parts) if isinstance(parts, list) else 'N/A'}")
                                                    if parts and isinstance(parts[0], dict):
                                                        logger.debug(f"[{batch_id}] First part keys: {list(parts[0].keys())}")
                                                        if "text" in parts[0]:
                                                            logger.debug(f"[{batch_id}] First part text (first 200 chars): {str(parts[0]['text'])[:200]}")
                            
                            # Parse JSON corrections
                            corrections = {}
                            if response_text.strip():
                                logger.debug(f"[{batch_id}] Extracted response text for {result_key} (length: {len(response_text)}, first 200 chars: {response_text[:200]})")
                                
                                # Check if response was truncated (finishReason: MAX_TOKENS)
                                finish_reason = None
                                if isinstance(response_obj, dict) and "candidates" in response_obj and response_obj["candidates"]:
                                    candidate = response_obj["candidates"][0]
                                    if isinstance(candidate, dict) and "finishReason" in candidate:
                                        finish_reason = candidate["finishReason"]
                                        if finish_reason == "MAX_TOKENS":
                                            logger.warning(f"[{batch_id}] Response for {result_key} was truncated (MAX_TOKENS). JSON may be incomplete.")
                                
                                try:
                                    parsed_json = json.loads(response_text.strip())
                                    corrections = parsed_json if isinstance(parsed_json, dict) else {}
                                    logger.debug(f"[{batch_id}] Successfully parsed JSON for {result_key}: {len(corrections)} keys")
                                except json.JSONDecodeError as e:
                                    logger.warning(f"[{batch_id}] JSON decode error for {result_key}: {e}")
                                    if finish_reason == "MAX_TOKENS":
                                        logger.warning(f"[{batch_id}] Response was truncated. Attempting to extract partial JSON...")
                                    
                                    # Try to extract and fix truncated JSON
                                    # Find the last complete key-value pair before the truncation
                                    import re
                                    # Try to find all complete "L##": "..." pairs
                                    pattern = r'"L\d+":\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
                                    matches = re.findall(pattern, response_text)
                                    if matches:
                                        # Reconstruct JSON from complete pairs
                                        corrections = {}
                                        for match in re.finditer(r'"L(\d+)":\s*"([^"\\]*(?:\\.[^"\\]*)*)"', response_text):
                                            key = f"L{match.group(1)}"
                                            value = match.group(2).replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                                            corrections[key] = value
                                        logger.info(f"[{batch_id}] Extracted {len(corrections)} complete line corrections from truncated JSON for {result_key}")
                                    else:
                                        # Fallback: try simple regex extraction
                                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                                        if json_match:
                                            try:
                                                corrections = json.loads(json_match.group(0))
                                                logger.info(f"[{batch_id}] Successfully parsed JSON using regex extraction for {result_key}: {len(corrections)} keys")
                                            except json.JSONDecodeError as e2:
                                                logger.warning(f"[{batch_id}] Failed to parse JSON even with regex for {result_key}: {e2}")
                            else:
                                logger.warning(f"[{batch_id}] No response text extracted for {result_key}")
                            
                            # Extract token usage - batch API uses camelCase (usageMetadata) not snake_case
                            usage_metadata = result_obj.get("usage_metadata") or response_obj.get("usageMetadata", {})
                            item_token_usage = {
                                "prompt_tokens": int(usage_metadata.get("promptTokenCount") or usage_metadata.get("prompt_token_count", 0) or 0),
                                "response_tokens": int(usage_metadata.get("candidatesTokenCount") or usage_metadata.get("candidates_token_count", 0) or 0),
                                "thoughts_tokens": int(usage_metadata.get("thoughtsTokenCount") or usage_metadata.get("thoughts_token_count", 0) or 0),
                                "total_tokens": int(usage_metadata.get("totalTokenCount") or usage_metadata.get("total_token_count", 0) or 0)
                            }
                            
                            total_prompt_tokens += item_token_usage["prompt_tokens"]
                            total_output_tokens += item_token_usage["response_tokens"]
                            total_thoughts_tokens += item_token_usage["thoughts_tokens"]
                            
                            results_map[result_key] = {
                                "corrections": corrections,
                                "raw_thoughts": raw_thoughts if raw_thoughts else response_text,  # Use raw_thoughts (all text) or fallback to response_text
                                "token_usage": item_token_usage
                            }
                            
                            logger.info(f"[{batch_id}] Completed {result_key}: {len(corrections)} corrections")
                        else:
                            # Request failed or status unclear
                            error_info = result_obj.get("error", {})
                            error_message = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                            logger.warning(f"[{batch_id}] Request {result_key} failed: status={status}, error={error_message}")
                            
                            # Log full result for debugging if status is None (unexpected)
                            if status is None:
                                logger.debug(f"[{batch_id}] Full result object for {result_key}: {json.dumps(result_obj, indent=2)[:500]}")
                            results_map[result_key] = {
                                "corrections": {},
                                "raw_thoughts": "",
                                "token_usage": {
                                    "prompt_tokens": 0,
                                    "response_tokens": 0,
                                    "thoughts_tokens": 0,
                                    "total_tokens": 0
                                }
                            }
                    except Exception as e:
                        logger.error(f"[{batch_id}] Error parsing result line: {e}", exc_info=True)
            
            # Ensure we have results for all requests (fill in missing ones)
            for request_data in batch_requests:
                key = request_data.get('key', 'unknown')
                if key not in results_map:
                    logger.warning(f"[{batch_id}] No result found for {key}, adding empty result")
                    results_map[key] = {
                        "corrections": {},
                        "raw_thoughts": "",
                        "token_usage": {
                            "prompt_tokens": 0,
                            "response_tokens": 0,
                            "thoughts_tokens": 0,
                            "total_tokens": 0
                        }
                    }
            
            logger.info(
                f"[{batch_id}] Batch complete: {total_prompt_tokens:,} prompt, "
                f"{total_output_tokens:,} response, {total_thoughts_tokens:,} thoughts tokens"
            )
            
            return results_map
            
        except Exception as e:
            logger.error(f"[{batch_id}] Error in batch API processing: {e}", exc_info=True)
            # Return empty results for all requests
            results_map = {}
            for request_data in batch_requests:
                key = request_data.get('key', 'unknown')
                results_map[key] = {
                    "corrections": {},
                    "raw_thoughts": "",
                    "token_usage": {
                        "prompt_tokens": 0,
                        "response_tokens": 0,
                        "thoughts_tokens": 0,
                        "total_tokens": 0
                    }
                }
            return results_map
        finally:
            # Cleanup temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"[{batch_id}] Could not cleanup temp directory: {e}")
    
    def _get_gemini_corrections(
        self,
        image_path: str,
        lines: List[Dict[str, Any]],
        htr_text: str,
        index_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, str]]:
        """
        Queue a request for batch processing (legacy method name kept for compatibility).
        This now adds requests to the batch queue instead of processing immediately.
        
        Args:
            image_path: Path to the image
            lines: List of line data with bounding boxes
            htr_text: Combined HTR text (not used, but kept for compatibility)
            index_data: Database index data
            
        Returns:
            None (requests are queued for batch processing)
        """
        # Prepare batch request
        batch_request = self._prepare_batch_request(image_path, lines, htr_text, index_data)
        if batch_request:
            with self.batch_queue_lock:
                self.batch_queue.append(batch_request)
            logger.info(f"Queued batch request for {image_path} ({len(lines)} lines)")
        else:
            logger.warning(f"Failed to prepare batch request for {image_path}")
        
        return None  # Will be processed in batch
    
    def _save_corrected_lines(
        self,
        image_path: str,
        lines: List[Dict[str, Any]],
        corrections: Dict[str, str]
    ):
        """
        Save corrected lines for Pylaia training.
        
        Args:
            image_path: Path to the source image
            lines: Original line data with polygons
            corrections: Dictionary of corrected text by line key
        """
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines", basename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "image_path": image_path,
            "image_basename": basename,
            "lines": []
        }
        
        for idx, line in enumerate(lines, 1):
            key = f"L{idx:02d}"
            if key in corrections:
                corrected_text = corrections[key]
                metadata["lines"].append({
                    "line_id": line["line_id"],
                    "key": key,
                    "htr_text": line.get("htr_text", ""),
                    "corrected_text": corrected_text,
                    "polygon": line.get("polygon"),
                    "bbox": line.get("bbox"),
                    "baseline": line.get("baseline"),  # Save baseline from Kraken
                    "baseline_coords": line.get("baseline_coords"),  # Save baseline coordinates
                })
                self.state["corrected_lines_count"] += 1
                self.stats["total_lines_corrected"] += 1
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self._save_checkpoint()
        self._save_statistics()
    
    def _should_retrain(self) -> bool:
        """Check if we should retrain the model."""
        corrected_count = self.state["corrected_lines_count"]
        last_retrain_count = self.state["last_retrain_line_count"]
        
        return (corrected_count - last_retrain_count) >= LINES_PER_RETRAIN
    
    def _retrain_pylaia_model(self, force: bool = False):
        """
        Retrain Pylaia model with accumulated corrected lines.
        
        Args:
            force: If True, bypass the minimum valid lines check and retrain anyway
        """
        logger.info("Starting Pylaia model retraining...")
        
        # Check if we have enough valid training lines before creating a new model version
        from .dataset_generator import count_valid_training_lines
        valid_lines_count = count_valid_training_lines(
            corrected_lines_dir=BOOTSTRAP_DATA_DIR,
            max_levenshtein_distance=MAX_LEVENSHTEIN_DISTANCE,
        )
        
        logger.info(f"Found {valid_lines_count} valid training lines (minimum required: {MIN_VALID_TRAINING_LINES})")
        
        if valid_lines_count < MIN_VALID_TRAINING_LINES:
            if force:
                logger.warning(
                    f"⚠️  FORCE MODE: Not enough valid training lines ({valid_lines_count} < {MIN_VALID_TRAINING_LINES}), "
                    f"but proceeding anyway as requested."
                )
            else:
                logger.warning(
                    f"Not enough valid training lines ({valid_lines_count} < {MIN_VALID_TRAINING_LINES}). "
                    f"Skipping retraining and continuing to use current model (v{self.state['current_model_version']}). "
                    f"Will retry when more lines are available. Use --force to bypass this check."
                )
                # Don't increment model version, don't retrain
                # Continue using the current model (which may be v0/initial model)
                # However, update last_retrain_line_count to prevent repeated checks
                # We'll check again when we have 3000 more new lines
                self.state["last_retrain_line_count"] = self.state["corrected_lines_count"]
                self._save_checkpoint()
                return
        
        # We have enough lines, proceed with retraining
        self.stats["model_retrains"] += 1
        
        # Increment model version at the start of training (not at the end)
        # This ensures the version is incremented even if training fails
        self.state["current_model_version"] += 1
        model_version = self.state["current_model_version"]
        logger.info(f"Incrementing model version to v{model_version} at start of training")
        self._save_checkpoint()  # Save checkpoint immediately with new version
        
        # Create model directory
        model_dir = os.path.join(BOOTSTRAP_PYLAIA_MODEL_DIR, f"model_v{model_version}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate dataset from corrected lines
        dataset_dir = os.path.join(BOOTSTRAP_DATASET_DIR, f"dataset_v{model_version}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        logger.info("Generating training dataset...")
        from .dataset_generator import generate_training_dataset
        generate_training_dataset(
            corrected_lines_dir=BOOTSTRAP_DATA_DIR,
            output_dir=dataset_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
            image_height=128,
            max_levenshtein_distance=MAX_LEVENSHTEIN_DISTANCE,
        )
        
        # Find the most recent model checkpoint to use as starting point
        latest_checkpoint, latest_model_file, latest_syms = self._find_latest_model_checkpoint()
        
        # Determine architecture source
        if latest_model_file and os.path.exists(latest_model_file):
            # Use the latest model's architecture as starting point
            arch_source = latest_model_file
            syms_source = latest_syms if latest_syms and os.path.exists(latest_syms) else INITIAL_PYLAIA_SYMS
            logger.info(f"Using latest model architecture from: {arch_source}")
            if latest_checkpoint:
                logger.info(f"Will start training from checkpoint: {latest_checkpoint}")
        elif model_version == 1:
            # First retrain: use architecture from model_courthand_3090C
            arch_source = FIRST_RETRAIN_ARCH
            syms_source = FIRST_RETRAIN_SYMS
            # Fallback to initial if not found
            if not os.path.exists(arch_source):
                arch_source = INITIAL_PYLAIA_ARCH
                syms_source = INITIAL_PYLAIA_SYMS
        else:
            # Subsequent retrains: use previous model architecture
            prev_model_dir = os.path.join(
                BOOTSTRAP_PYLAIA_MODEL_DIR,
                f"model_v{model_version - 1}"
            )
            arch_source = os.path.join(prev_model_dir, "model")
            syms_source = os.path.join(prev_model_dir, "syms.txt")
            
            # Fallback to initial if not found
            if not os.path.exists(arch_source):
                arch_source = INITIAL_PYLAIA_ARCH
                syms_source = INITIAL_PYLAIA_SYMS
        
        # Copy symbols first (following train_model.sh pattern)
        # IMPORTANT: Use dataset symbols file since it contains all characters from the actual training data
        model_syms_path = os.path.join(model_dir, "syms.txt")
        dataset_syms = os.path.join(dataset_dir, "syms.txt")
        
        # Prioritize dataset symbols (contains all characters from tokenized text)
        if os.path.exists(dataset_syms):
            shutil.copy2(dataset_syms, model_syms_path)
            logger.info(f"Using symbols from dataset: {dataset_syms}")
        elif os.path.exists(syms_source):
            # Fallback to source symbols if dataset doesn't have them yet
            shutil.copy2(syms_source, model_syms_path)
            logger.warning(f"Using symbols from source model (dataset symbols not found): {syms_source}")
        else:
            logger.error(f"Could not find syms.txt in {dataset_dir} or {syms_source}")
            return
        
        # Check if model exists to determine resume behavior (following train_model.sh)
        model_file = os.path.join(model_dir, "model")
        model_exists = os.path.exists(model_file)
        
        # Check if there are valid checkpoints in experiment directory
        experiment_dir = os.path.join(model_dir, "experiment")
        has_checkpoints = False
        if os.path.exists(experiment_dir):
            # Look for checkpoint files
            checkpoint_files = [f for f in os.listdir(experiment_dir) if f.endswith('.ckpt')]
            has_checkpoints = len(checkpoint_files) > 0
        
        # Determine if we should use the latest checkpoint as starting point
        starting_checkpoint = None
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            # Copy the latest checkpoint to the new model's experiment directory to use as starting point
            os.makedirs(experiment_dir, exist_ok=True)
            # Copy checkpoint with a name that Pylaia will recognize
            checkpoint_name = os.path.basename(latest_checkpoint)
            target_checkpoint = os.path.join(experiment_dir, checkpoint_name)
            if not os.path.exists(target_checkpoint):
                shutil.copy2(latest_checkpoint, target_checkpoint)
                logger.info(f"Copied latest checkpoint to new model directory: {target_checkpoint}")
            starting_checkpoint = target_checkpoint
        
        if model_exists and has_checkpoints:
            logger.info(f"Existing model and checkpoints found. Will resume training.")
            do_create_model = False
            resume_flag = "true"
        elif model_exists and not has_checkpoints:
            if starting_checkpoint:
                logger.info(f"Existing model found. Will start training from latest checkpoint: {starting_checkpoint}")
                do_create_model = False
                resume_flag = "true"
            else:
                logger.info(f"Existing model found but no checkpoints. Will start fresh training (resume=false).")
                do_create_model = False
                resume_flag = "false"
                # Clean up empty experiment directory
                if os.path.exists(experiment_dir):
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    backup_dir = os.path.join(model_dir, f"experiment_backup_{timestamp}")
                    shutil.move(experiment_dir, backup_dir)
                    logger.info(f"Backed up empty experiment directory to {backup_dir}")
        else:
            if starting_checkpoint:
                logger.info(f"No existing model found. Will create new model architecture and start from latest checkpoint: {starting_checkpoint}")
                do_create_model = True
                resume_flag = "true"
            else:
                logger.info(f"No existing model found. Will create new model architecture.")
                do_create_model = True
                resume_flag = "false"
            # Backup existing experiment directory if it exists (following train_model.sh)
            if os.path.exists(experiment_dir) and not starting_checkpoint:
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join(model_dir, f"experiment_backup_{timestamp}")
                shutil.move(experiment_dir, backup_dir)
                logger.info(f"Backed up existing experiment directory to {backup_dir}")
        
        # Copy architecture if it exists and we're not creating new (for subsequent retrains)
        if not do_create_model and os.path.exists(arch_source):
            shutil.copy2(arch_source, model_file)
        
        # Create training config (following train_model.sh structure)
        train_config_path = os.path.join(model_dir, "train_config.yaml")
        with open(train_config_path, 'w', encoding='utf-8') as f:
            f.write(f"""syms: {model_syms_path}
img_dirs:
  - {os.path.join(dataset_dir, 'images')}
  - {dataset_dir}/
tr_txt_table: {os.path.join(dataset_dir, 'train.txt')}
va_txt_table: {os.path.join(dataset_dir, 'val.txt')}

data:
  batch_size: 12
  num_workers: 4
  color_mode: L

train:
  delimiters: ["<space>"]
  early_stopping_patience: 40
  checkpoint_k: 5
  resume: {resume_flag}
  augment_training: true

common:
  train_path: {model_dir}
  model_filename: model
  monitor: va_cer


logging:
  level: INFO
  filepath: train-crnn.log

trainer:
  max_epochs: 500
  accelerator: gpu
  devices: [0]
  precision: 16
  # Verbose training options (only supported options)
  progress_bar_refresh_rate: 1
  log_every_n_steps: 10
  check_val_every_n_epoch: 1

optimizer:
  learning_rate: 0.0001
  name: Adam
""")
        
        # Create model architecture config (only if creating new model, following train_model.sh)
        if do_create_model:
            create_config_path = os.path.join(model_dir, "create_config.yaml")
            with open(create_config_path, 'w', encoding='utf-8') as f:
                f.write(f"""syms: {model_syms_path}
fixed_input_height: 128
adaptive_pooling: avg
common:
  train_path: {model_dir}
  model_filename: model
crnn:
  # 4-Layer CNN optimized for Latin Court Hand
  # Increased initial depth (32) to capture complex abbreviation glyphs/texture
  cnn_num_features: [32, 64, 128, 256]
  cnn_kernel_size: [3, 3, 3, 3]
  cnn_stride: [1, 1, 1, 1]
  cnn_dilation: [1, 1, 1, 1]
  cnn_activation: [LeakyReLU, LeakyReLU, LeakyReLU, LeakyReLU]
  cnn_batchnorm: [true, true, true, true]
  
  # Anisotropic Pooling: [[2, 2], [2, 2], [2, 1], [2, 1]]
  # Reduces Height by 16 (128->8), but Width only by 4.
  # Essential for distinguishing dense minims (i, n, m, u) in court hand.
  cnn_poolsize: [[2, 2], [2, 2], [2, 1], [2, 1]]
  
  # Added dropout to deeper layers to prevent overfitting on parchment noise
  cnn_dropout: [0.0, 0.0, 0.2, 0.2]
  
  rnn_type: LSTM
  rnn_layers: 3
  rnn_units: 512
  rnn_dropout: 0.5
  lin_dropout: 0.5

""")

            
            # Create model architecture (following train_model.sh)
            logger.info("Creating Model Architecture...")
            cmd_create = (
                f"source {PYLAIA_ENV} && "
                f"pylaia-htr-create-model --config '{create_config_path}'"
            )
            # Model creation is critical - raise exception on failure
            self._run_command(cmd_create, "Create Model Architecture", raise_on_error=True)
        else:
            logger.info("Skipping model creation (using existing architecture)")
        
        # Run training
        # Set ulimit like train_model.sh does (increase file descriptor limit)
        cmd_train = (
            f"ulimit -n 4096 2>/dev/null || true && "
            f"source {PYLAIA_ENV} && "
            f"pylaia-htr-train-ctc --config '{train_config_path}'"
        )
        # Training is critical - raise exception on failure
        # For training, show output in real-time for better visibility
        logger.info("Starting training with verbose output...")
        logger.info(f"Training log will be saved to: {os.path.join(model_dir, 'experiment', 'train-crnn.log')}")
        self._run_command(cmd_train, "Train Pylaia Model", raise_on_error=True, show_output=True)
        
        # Find and select the best checkpoint (lowest validation CER)
        best_checkpoint = self._find_best_checkpoint(model_dir)
        if best_checkpoint:
            # Copy best checkpoint to canonical location in model directory
            canonical_checkpoint = os.path.join(model_dir, os.path.basename(best_checkpoint))
            if best_checkpoint != canonical_checkpoint:
                shutil.copy2(best_checkpoint, canonical_checkpoint)
                logger.info(f"Selected best checkpoint: {canonical_checkpoint}")
            else:
                logger.info(f"Best checkpoint already in place: {canonical_checkpoint}")
            
            # Store checkpoint path in state for use in subsequent HTR
            # All retrained models (v1+) use 128px height
            self.state["pylaia_models"][f"v{model_version}"] = {
                "checkpoint": canonical_checkpoint,
                "model_file": model_file,
                "syms": model_syms_path,
                "image_height": SUBSEQUENT_MODEL_IMAGE_HEIGHT,  # Store required line height (128px for v1+)
            }
        else:
            logger.warning(f"Could not find best checkpoint in {model_dir}/experiment")
        
        self.state["last_retrain_line_count"] = self.state["corrected_lines_count"]
        self._save_checkpoint()
        self._save_statistics()
        
        logger.info(f"Model retraining completed. New version: {model_version}")
        
        # After retraining, regenerate HTR on all previously processed images
        logger.info("Regenerating HTR on all previously processed images with new model...")
        self._regenerate_htr_with_latest_model()
    
    def _find_latest_model_checkpoint(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Find the most recent model checkpoint to use as starting point for training.
        
        Returns:
            Tuple of (checkpoint_path, model_file_path, syms_file_path) or (None, None, None) if not found
        """
        # Look for model directories
        model_dirs = []
        if os.path.exists(BOOTSTRAP_PYLAIA_MODEL_DIR):
            for item in os.listdir(BOOTSTRAP_PYLAIA_MODEL_DIR):
                if item.startswith("model_v") and os.path.isdir(os.path.join(BOOTSTRAP_PYLAIA_MODEL_DIR, item)):
                    try:
                        version = int(item.replace("model_v", ""))
                        model_dirs.append((version, item))
                    except ValueError:
                        continue
        
        if not model_dirs:
            # No bootstrap models found, check if we can use initial model
            if os.path.exists(INITIAL_PYLAIA_MODEL):
                logger.info(f"No bootstrap models found, will use initial model: {INITIAL_PYLAIA_MODEL}")
                return INITIAL_PYLAIA_MODEL, INITIAL_PYLAIA_ARCH, INITIAL_PYLAIA_SYMS
            return None, None, None
        
        # Sort by version number (descending) and find the first complete model
        model_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Iterate through models from latest to oldest to find the first complete one
        for version, dir_name in model_dirs:
            model_dir = os.path.join(BOOTSTRAP_PYLAIA_MODEL_DIR, dir_name)
            logger.debug(f"Checking model directory: {model_dir}")
            
            # Find the best checkpoint (highest epoch with "lowest_va_cer")
            # Search both root and experiment directories to find the best one
            checkpoint = None
            best_epoch = -1
            
            # Check root directory
            if os.path.exists(model_dir):
                try:
                    for filename in os.listdir(model_dir):
                        # Skip directories
                        filepath = os.path.join(model_dir, filename)
                        if os.path.isdir(filepath):
                            continue
                        if "lowest_va_cer" in filename and filename.endswith(".ckpt"):
                            try:
                                epoch_str = filename.split("=")[1].split("-")[0]
                                epoch = int(epoch_str)
                                if epoch > best_epoch:
                                    best_epoch = epoch
                                    checkpoint = filepath
                                    logger.debug(f"Found checkpoint in root: {checkpoint} (epoch {epoch})")
                            except (ValueError, IndexError):
                                if checkpoint is None:
                                    checkpoint = filepath
                                    logger.debug(f"Found checkpoint in root (couldn't parse epoch): {checkpoint}")
                except OSError as e:
                    logger.warning(f"Error reading model directory {model_dir}: {e}")
            
            # Check experiment directory (always check to find the best checkpoint)
            experiment_dir = os.path.join(model_dir, "experiment")
            if os.path.exists(experiment_dir):
                try:
                    for filename in os.listdir(experiment_dir):
                        if "lowest_va_cer" in filename and filename.endswith(".ckpt"):
                            try:
                                epoch_str = filename.split("=")[1].split("-")[0]
                                epoch = int(epoch_str)
                                if epoch > best_epoch:
                                    best_epoch = epoch
                                    checkpoint = os.path.join(experiment_dir, filename)
                                    logger.debug(f"Found checkpoint in experiment: {checkpoint} (epoch {epoch})")
                            except (ValueError, IndexError):
                                if checkpoint is None:
                                    checkpoint = os.path.join(experiment_dir, filename)
                                    logger.debug(f"Found checkpoint in experiment (couldn't parse epoch): {checkpoint}")
                except OSError as e:
                    logger.warning(f"Error reading experiment directory {experiment_dir}: {e}")
            
            model_file = os.path.join(model_dir, "model")
            syms_file = os.path.join(model_dir, "syms.txt")
            
            # Verify all required files exist - if this model is complete, use it
            if checkpoint and os.path.exists(checkpoint) and os.path.exists(model_file):
                if os.path.exists(syms_file):
                    logger.info(f"Found complete model v{version}: checkpoint={checkpoint}, model={model_file}, syms={syms_file}")
                    return checkpoint, model_file, syms_file
                else:
                    logger.warning(f"Found model v{version} but syms.txt missing: checkpoint={checkpoint}, model={model_file}")
                    # Still return it, syms can be generated from dataset
                    logger.info(f"Using model v{version}: checkpoint={checkpoint}, model={model_file}")
                    return checkpoint, model_file, None
            else:
                # This model is incomplete, try the next one
                missing = []
                if not checkpoint:
                    missing.append("checkpoint")
                elif not os.path.exists(checkpoint):
                    missing.append(f"checkpoint file ({checkpoint})")
                if not os.path.exists(model_file):
                    missing.append(f"model file ({model_file})")
                logger.debug(f"Model v{version} incomplete (missing: {', '.join(missing)}), trying previous version...")
                continue
        
        # No complete bootstrap models found, fall back to initial model
        logger.warning(f"No complete bootstrap models found, falling back to initial model")
        if os.path.exists(INITIAL_PYLAIA_MODEL):
            return INITIAL_PYLAIA_MODEL, INITIAL_PYLAIA_ARCH, INITIAL_PYLAIA_SYMS
        
        return None, None, None
    
    @staticmethod
    def _find_best_checkpoint_static(model_dir: str) -> Optional[str]:
        """
        Static version of _find_best_checkpoint for use during initialization.
        
        Find the best checkpoint (lowest validation CER) in the experiment directory.
        
        Args:
            model_dir: Model directory containing experiment subdirectory
            
        Returns:
            Path to best checkpoint or None
        """
        experiment_dir = os.path.join(model_dir, "experiment")
        if not os.path.exists(experiment_dir):
            return None
        
        # Look for checkpoints with "lowest_va_cer" in the name
        # Pylaia saves the best checkpoint with this naming pattern
        best_checkpoint = None
        best_epoch = -1
        
        try:
            for filename in os.listdir(experiment_dir):
                if "lowest_va_cer" in filename and filename.endswith(".ckpt"):
                    # Extract epoch number from filename like "epoch=79-lowest_va_cer.ckpt"
                    try:
                        epoch_str = filename.split("=")[1].split("-")[0]
                        epoch = int(epoch_str)
                        if epoch > best_epoch:
                            best_epoch = epoch
                            best_checkpoint = os.path.join(experiment_dir, filename)
                    except (ValueError, IndexError):
                        # If we can't parse epoch, still consider it
                        if best_checkpoint is None:
                            best_checkpoint = os.path.join(experiment_dir, filename)
        except OSError:
            return None
        
        return best_checkpoint
    
    def _find_best_checkpoint(self, model_dir: str) -> Optional[str]:
        """
        Find the best checkpoint (lowest validation CER) in the experiment directory.
        
        Args:
            model_dir: Model directory containing experiment subdirectory
            
        Returns:
            Path to best checkpoint or None
        """
        best_checkpoint = self._find_best_checkpoint_static(model_dir)
        if best_checkpoint:
            logger.info(f"Found best checkpoint: {best_checkpoint}")
        else:
            # Fallback: look for any checkpoint file
            logger.warning("No 'lowest_va_cer' checkpoint found, looking for any checkpoint...")
            experiment_dir = os.path.join(model_dir, "experiment")
            if os.path.exists(experiment_dir):
                try:
                    for filename in os.listdir(experiment_dir):
                        if filename.endswith(".ckpt"):
                            checkpoint_path = os.path.join(experiment_dir, filename)
                            # Prefer "last" checkpoint as fallback
                            if "last" in filename:
                                best_checkpoint = checkpoint_path
                                break
                            elif best_checkpoint is None:
                                best_checkpoint = checkpoint_path
                except OSError as e:
                    logger.error(f"Error reading experiment directory: {e}")
        
        return best_checkpoint
    
    def _regenerate_htr_with_latest_model(self):
        """
        Regenerate HTR results for all previously processed images using the latest model.
        
        This function is called after retraining to update all existing HTR results
        with the newly trained model.
        """
        # Find the latest model (just trained)
        latest_checkpoint, latest_model_file, latest_syms = self._find_latest_model_checkpoint()
        
        if not latest_checkpoint or not latest_model_file:
            logger.warning("Could not find latest model checkpoint for HTR regeneration. Skipping.")
            return
        
        if not os.path.exists(latest_checkpoint):
            logger.warning(f"Latest checkpoint not found: {latest_checkpoint}. Skipping HTR regeneration.")
            return
        
        if not os.path.exists(latest_model_file):
            logger.warning(f"Latest model file not found: {latest_model_file}. Skipping HTR regeneration.")
            return
        
        syms_file = latest_syms if latest_syms and os.path.exists(latest_syms) else INITIAL_PYLAIA_SYMS
        if not os.path.exists(syms_file):
            logger.warning(f"Symbols file not found: {syms_file}. Skipping HTR regeneration.")
            return
        
        logger.info(f"Regenerating HTR with model checkpoint: {latest_checkpoint}")
        logger.info(f"  Model file: {latest_model_file}")
        logger.info(f"  Symbols: {syms_file}")
        
        # Find all HTR work directories that have existing kraken.json
        HTR_WORK_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "htr_work")
        if not os.path.exists(HTR_WORK_DIR):
            logger.info("No HTR work directory found. Nothing to regenerate.")
            return
        
        work_dirs = []
        for item in os.listdir(HTR_WORK_DIR):
            item_path = os.path.join(HTR_WORK_DIR, item)
            if os.path.isdir(item_path):
                kraken_json = os.path.join(item_path, "kraken.json")
                if os.path.exists(kraken_json):
                    work_dirs.append(item)
        
        if not work_dirs:
            logger.info("No images with existing HTR work found. Nothing to regenerate.")
            return
        
        logger.info(f"Found {len(work_dirs)} images with existing HTR work to regenerate")
        
        # Process each image
        success_count = 0
        fail_count = 0
        
        for basename in sorted(work_dirs):
            logger.info(f"Regenerating HTR for: {basename}")
            if self._regenerate_htr_for_image(basename, latest_checkpoint, latest_model_file, syms_file):
                success_count += 1
            else:
                fail_count += 1
        
        logger.info(f"HTR regeneration completed: {success_count} successful, {fail_count} failed")
    
    def _regenerate_htr_for_image(self, basename: str, checkpoint: str, model_file: str, syms_file: str) -> bool:
        """
        Regenerate HTR results for a single image.
        
        Args:
            basename: Image basename (e.g., "CP 40-559 055-a")
            checkpoint: Path to PyLaia checkpoint
            model_file: Path to PyLaia model architecture file
            syms_file: Path to symbols file
            
        Returns:
            True if processing was successful
        """
        HTR_WORK_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "htr_work")
        CORRECTED_LINES_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines")
        
        work_dir = os.path.join(HTR_WORK_DIR, basename)
        kraken_json = os.path.join(work_dir, "kraken.json")
        lines_dir = os.path.join(work_dir, "lines")
        list_txt = os.path.join(work_dir, "img_list.txt")
        htr_res = os.path.join(work_dir, "htr.txt")
        
        # Find the original image
        INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "input_images")
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            path = os.path.join(INPUT_IMAGES_DIR, basename + ext)
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            # Try to get image path from kraken.json
            try:
                with open(kraken_json, 'r') as f:
                    kraken_data = json.load(f)
                    image_path = kraken_data.get("imagename")
            except:
                pass
        
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Cannot find image for {basename}")
            return False
        
        # Check kraken.json exists
        if not os.path.exists(kraken_json):
            logger.warning(f"No kraken.json found for {basename}")
            return False
        
        # Check if we need to regenerate line images
        # Only regenerate if they don't exist or have wrong height
        need_regenerate_lines = True
        if os.path.exists(lines_dir) and os.path.exists(list_txt) and os.path.getsize(list_txt) > 0:
            # Check if all line images have the correct height (128px for bootstrap models v1+)
            try:
                all_correct_height = True
                with open(list_txt, 'r') as f:
                    for line_path in f:
                        line_path = line_path.strip()
                        if not line_path:
                            continue
                        # Handle relative paths
                        if not os.path.isabs(line_path):
                            line_path = os.path.join(lines_dir, os.path.basename(line_path))
                        if os.path.exists(line_path):
                            try:
                                with Image.open(line_path) as img:
                                    width, height = img.size
                                    if height != SUBSEQUENT_MODEL_IMAGE_HEIGHT:
                                        all_correct_height = False
                                        break
                            except Exception:
                                all_correct_height = False
                                break
                        else:
                            all_correct_height = False
                            break
                
                if all_correct_height:
                    need_regenerate_lines = False
                    logger.debug(f"Line images for {basename} already have correct height ({SUBSEQUENT_MODEL_IMAGE_HEIGHT}px). Skipping regeneration.")
            except Exception as e:
                logger.debug(f"Error checking line image heights for {basename}: {e}. Will regenerate.")
                need_regenerate_lines = True
        
        # Regenerate line images if needed
        if need_regenerate_lines:
            # Remove existing lines directory and recreate
            import shutil
            if os.path.exists(lines_dir):
                shutil.rmtree(lines_dir)
            os.makedirs(lines_dir, exist_ok=True)
            
            # Preprocess lines (extract line images)
            try:
                from line_preprocessor_greyscale.config import FINAL_LINE_HEIGHT
                from line_preprocessor_greyscale.runner import _load_image, _expand_polygons, _save_line_image
                from line_preprocessor_greyscale.processing import initial_line_extraction, process_line_image_greyscale
                from line_preprocessor.parser import parse_kraken_json_for_processing
                
                # Override height for bootstrap models (v1+ use 128px)
                import line_preprocessor_greyscale.config
                line_preprocessor_greyscale.config.FINAL_LINE_HEIGHT = SUBSEQUENT_MODEL_IMAGE_HEIGHT
                
                page_image = _load_image(image_path)
                lines_to_process = parse_kraken_json_for_processing(kraken_json)
                if not lines_to_process:
                    logger.warning(f"No text lines found in the Kraken JSON for {basename}")
                    return False
                
                logger.debug(f"Found {len(lines_to_process)} lines to process. Expanding polygons...")
                expanded_lines = _expand_polygons(lines_to_process)
                
                logger.debug(f"Processing lines with height={SUBSEQUENT_MODEL_IMAGE_HEIGHT}px...")
                with open(list_txt, "w", encoding="utf-8") as pylaia_list_file:
                    for line_data in expanded_lines:
                        try:
                            initial_result = initial_line_extraction(
                                page_image,
                                line_data["polygon"],
                                line_data["baseline"],
                                padding=10,
                            )
                            if not initial_result:
                                continue
                            
                            line_rect_img, line_polygon_coords, line_baseline_points = initial_result
                            final_image = process_line_image_greyscale(
                                line_rect_img,
                                line_polygon_coords,
                                line_baseline_points,
                                final_canvas_height=SUBSEQUENT_MODEL_IMAGE_HEIGHT,
                                line_id_for_debug=line_data["id"],
                            )
                            if final_image:
                                abs_path = _save_line_image(final_image, lines_dir, line_data["id"])
                                pylaia_list_file.write(f"{abs_path}\n")
                        except Exception as exc:
                            logger.warning(f"Unhandled exception on line {line_data['id']}: {exc}")
                
                logger.debug(f"Pylaia input file list saved to: {list_txt}")
            except Exception as e:
                logger.error(f"Error preprocessing lines for {basename}: {e}")
                return False
        
        # Check if we have lines to process
        if not os.path.exists(list_txt) or os.path.getsize(list_txt) == 0:
            logger.warning(f"No lines extracted for {basename}")
            return False
        
        # Run PyLaia decode
        cmd_parts = [
            f"source {PYLAIA_ENV} &&",
            "pylaia-htr-decode-ctc",
            "--trainer.accelerator gpu",
            "--trainer.devices 1",
            f"--common.checkpoint '{checkpoint}'",
            f"--common.model_filename '{model_file}'",
            "--decode.include_img_ids true",
            "--decode.print_word_confidence_score true",
        ]
        
        # Add language model if available
        if KENLM_MODEL_PATH and os.path.exists(KENLM_MODEL_PATH):
            # Determine which format to use
            if KENLM_USE_BINARY:
                # Try binary format first
                binary_path = KENLM_MODEL_PATH.replace('.arpa', '.klm')
                if os.path.exists(binary_path):
                    lm_path = binary_path
                else:
                    # Fall back to ARPA if binary doesn't exist
                    lm_path = KENLM_MODEL_PATH
                    logger.warning(f"Binary KenLM model not found at {binary_path}, using ARPA format")
            else:
                lm_path = KENLM_MODEL_PATH
            
            # Generate tokens and lexicon files from symbols file if needed
            # Always regenerate to ensure they include <ctc> and are up-to-date
            try:
                tokens_path, lexicon_path = ensure_kenlm_files(syms_file, force_regenerate=True)
                cmd_parts.extend([
                    f"--decode.use_language_model true",
                    f"--decode.language_model_path '{lm_path}'",
                    f"--decode.tokens_path '{tokens_path}'",
                    f"--decode.lexicon_path '{lexicon_path}'",
                    f"--decode.language_model_weight {KENLM_MODEL_WEIGHT}",
                ])
                logger.info(f"Using KenLM language model: {lm_path} (weight: {KENLM_MODEL_WEIGHT})")
            except Exception as e:
                logger.warning(f"Failed to generate KenLM support files: {e}. Language model disabled.")
        
        # Add syms and list files
        cmd_parts.append(f"'{syms_file}' '{list_txt}' > '{htr_res}'")
        cmd_decode = " ".join(cmd_parts)
        
        try:
            result = subprocess.run(
                cmd_decode,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            if result.returncode != 0:
                logger.warning(f"PyLaia decode failed for {basename}: {result.stderr[:500]}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning(f"PyLaia decode timed out for {basename}")
            return False
        except Exception as e:
            logger.warning(f"PyLaia decode error for {basename}: {e}")
            return False
        
        # Parse HTR results and update metadata.json
        htr_results = self._parse_htr_results(htr_res)
        if htr_results:
            self._update_metadata_json(basename, htr_results)
        
        return True
    
    def _parse_htr_results(self, htr_txt_path: str) -> Dict[str, str]:
        """
        Parse HTR results file and return a mapping of line_id -> htr_text.
        
        The htr.txt format is:
        /path/to/lines/uuid.png ['conf1', 'conf2', ...] c h a r <space> c h a r ...
        """
        results = {}
        
        if not os.path.exists(htr_txt_path) or os.path.getsize(htr_txt_path) == 0:
            return results
        
        with open(htr_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse the line - format: path [confidences] characters
                parts = line.split()
                if not parts:
                    continue
                
                path = parts[0]
                # Extract UUID from filename
                basename = os.path.basename(path)
                line_id = os.path.splitext(basename)[0]
                
                # Find where the confidence array ends
                bracket_start = line.find("[")
                bracket_end = line.find("]")
                
                if bracket_start != -1 and bracket_end != -1:
                    # Text starts after the closing bracket
                    text_part = line[bracket_end + 1:].strip()
                else:
                    # No confidence scores, text is everything after path
                    text_part = ' '.join(parts[1:])
                
                # The text is space-separated characters with <space> for actual spaces
                results[line_id] = text_part
        
        return results
    
    def _update_metadata_json(self, basename: str, htr_results: Dict[str, str]) -> bool:
        """
        Update the metadata.json file in corrected_lines with new HTR results.
        
        Args:
            basename: The image basename (e.g., "CP 40-559 055-a")
            htr_results: Dictionary mapping line_id to htr_text
        
        Returns:
            True if update was successful, False otherwise
        """
        CORRECTED_LINES_DIR = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines")
        metadata_dir = os.path.join(CORRECTED_LINES_DIR, basename)
        metadata_path = os.path.join(metadata_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.debug(f"No metadata.json found for {basename}, skipping update")
            return False
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading metadata.json for {basename}: {e}")
            return False
        
        # Update htr_text for each line
        lines = metadata.get("lines", [])
        updated_count = 0
        
        for line in lines:
            line_id = line.get("line_id")
            if line_id and line_id in htr_results:
                line["htr_text"] = htr_results[line_id]
                updated_count += 1
        
        # Write back
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error writing metadata.json for {basename}: {e}")
            return False
        
        if updated_count > 0:
            logger.debug(f"Updated {updated_count}/{len(lines)} lines in metadata.json for {basename}")
        
        return True
    
    def process_image_htr(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single image through HTR pipeline and prepare batch request.
        Returns batch request data instead of adding to queue.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Batch request dictionary if image should be processed, None otherwise
        """
        # Check if corrections already exist (check disk instead of checkpoint)
        has_corrections = self._is_image_processed(image_path)
        
        # Only skip if corrections exist (not just HTR)
        if has_corrections and not self.force:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            corrected_dir = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines", basename)
            logger.debug(f"Skipping already processed image (has corrections): {basename} at {corrected_dir}")
            return None
        elif has_corrections and self.force:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            logger.info(f"Force mode: re-processing image with existing corrections: {basename}")
        elif not has_corrections:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            corrected_dir = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines", basename)
            metadata_path = os.path.join(corrected_dir, "metadata.json")
            logger.debug(f"No corrections found for {basename} (checked {metadata_path})")
        
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Step 1: Run HTR tools
            kraken_json, htr_txt, merged_lines = self._run_htr_tools(image_path)
            if not merged_lines:
                logger.warning(f"No lines found in {image_path}")
                return None
            
            with self.stats_lock:
                self.stats["total_lines_processed"] += len(merged_lines)
            
            # Step 2: Get database index data
            parsed = parse_image_filename(image_path)
            if not parsed:
                logger.warning(f"Could not parse roll/rotulus from {image_path}")
                return None
            
            roll, rotulus = parsed
            index_data = self._get_database_index_data(roll, rotulus)
            
            # Step 3: Re-check corrections in case they were created during processing
            basename_check = os.path.splitext(os.path.basename(image_path))[0]
            corrected_dir_check = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines", basename_check)
            metadata_path_check = os.path.join(corrected_dir_check, "metadata.json")
            has_corrections_now = os.path.exists(corrected_dir_check) and os.path.exists(metadata_path_check)
            
            if has_corrections_now:
                logger.info(f"Corrections already exist for {basename_check}, skipping batch request")
                return None
            
            # Step 4: Prepare batch request (don't add to queue yet)
            htr_text_combined = "\n".join([line.get("htr_text", "") for line in merged_lines])
            batch_request = self._prepare_batch_request(
                image_path,
                merged_lines,
                htr_text_combined,
                index_data
            )
            
            if batch_request:
                logger.info(f"Prepared batch request for {image_path} ({len(merged_lines)} lines)")
            
            with self.stats_lock:
                self.stats["images_processed"] += 1
            
            return batch_request
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def process_image(self, image_path: str) -> bool:
        """
        Process a single image through the complete workflow (legacy method that adds to queue).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if processing succeeded, False otherwise
        """
        batch_request = self.process_image_htr(image_path)
        if batch_request:
            # Add to queue for backwards compatibility
            with self.batch_queue_lock:
                self.batch_queue.append(batch_request)
            return True
        return batch_request is None  # True if skipped, False if error
    
    def run(self, batch_size: int = 10, max_workers: int = 4):
        """
        Run the complete bootstrap training workflow on all images.
        
        Args:
            batch_size: Number of images to process in each batch
            max_workers: Maximum number of parallel workers for image processing (default: 4)
        """
        logger.info("Starting bootstrap training workflow...")
        
        # Get all images from input_images directory
        image_files = []
        if os.path.exists(IMAGE_DIR):
            for filename in os.listdir(IMAGE_DIR):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(IMAGE_DIR, filename))
        
        image_files.sort()
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Re-queue images that have HTR but no corrections yet
        # Check for images in htr_work that don't have corrections
        htr_work_dir = os.path.join(BOOTSTRAP_DATA_DIR, "htr_work")
        if os.path.exists(htr_work_dir):
            for basename in os.listdir(htr_work_dir):
                htr_dir = os.path.join(htr_work_dir, basename)
                if not os.path.isdir(htr_dir):
                    continue
                
                # Check if HTR results exist
                htr_txt = os.path.join(htr_dir, "htr.txt")
                kraken_json = os.path.join(htr_dir, "kraken.json")
                
                # Check if corrections already exist
                corrected_dir = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines", basename)
                has_corrections = os.path.exists(corrected_dir) and os.path.exists(
                    os.path.join(corrected_dir, "metadata.json")
                )
                
                # If HTR exists but no corrections, try to re-queue
                if os.path.exists(htr_txt) and os.path.exists(kraken_json) and not has_corrections:
                    # Find the original image path
                    for img_path in image_files:
                        if os.path.splitext(os.path.basename(img_path))[0] == basename:
                            # Check if already processed by checking disk
                            if not self._is_image_processed(img_path):
                                # Try to load HTR data and re-queue
                                logger.info(f"Found HTR results for {basename} but no corrections. Will re-process to queue for corrections...")
                            break
        
        # Process images in chunks of batch_size
        # For each chunk: process all images in parallel through HTR, then batch together for LLM
        logger.info("=== PROCESSING IMAGES IN CHUNKS ===")
        logger.info(f"Processing {len(image_files)} images in chunks of {batch_size}")
        logger.info(f"Each chunk: {max_workers} parallel HTR workers -> batch of {batch_size} for LLM")
        
        # Split images into chunks
        chunks = []
        for i in range(0, len(image_files), batch_size):
            chunk = image_files[i:i + batch_size]
            chunks.append((i // batch_size + 1, chunk))
        
        total_chunks = len(chunks)
        total_processed = 0
        
        # Process each chunk
        for chunk_num, image_chunk in chunks:
            logger.info(f"=== CHUNK {chunk_num}/{total_chunks} ===")
            logger.info(f"Processing {len(image_chunk)} images in parallel through HTR pipeline...")
            
            # Process all images in this chunk in parallel
            batch_requests = []
            
            def process_image_wrapper(image_path: str, idx: int) -> Optional[Dict[str, Any]]:
                """Wrapper to process single image and return batch request."""
                try:
                    logger.info(f"  [{idx}/{len(image_chunk)}] Processing: {os.path.basename(image_path)}")
                    return self.process_image_htr(image_path)
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
                    return None
            
            # Process chunk in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(process_image_wrapper, img_path, idx + 1): idx
                    for idx, img_path in enumerate(image_chunk)
                }
                
                # Collect batch requests as they complete
                for future in as_completed(future_to_idx):
                    batch_request = future.result()
                    if batch_request:
                        batch_requests.append(batch_request)
            
            # Log chunk HTR completion
            logger.info(f"Chunk {chunk_num} HTR complete: {len(batch_requests)}/{len(image_chunk)} images ready for LLM")
            
            # Process batch requests together
            if batch_requests:
                batch_id = f"chunk_{chunk_num}"
                logger.info(f"=== Submitting batch {batch_id} with {len(batch_requests)} images to LLM ===")
                
                results = self._run_batch_corrections_gemini_batch(batch_requests, batch_id)
                
                # Process results and save corrected lines
                saved_count = 0
                for batch_request in batch_requests:
                    key = batch_request['key']
                    img_path = batch_request['image_path']
                    lines = batch_request['lines']
                    
                    if key in results:
                        result = results[key]
                        corrections = result.get('corrections', {})
                        raw_thoughts = result.get('raw_thoughts', '')
                        token_usage = result.get('token_usage', {})
                        
                        if corrections:
                            self._save_corrected_lines(img_path, lines, corrections)
                            saved_count += 1
                            
                            if raw_thoughts:
                                basename = os.path.splitext(os.path.basename(img_path))[0]
                                thoughts_dir = os.path.join(BOOTSTRAP_DATA_DIR, "corrected_lines", basename)
                                os.makedirs(thoughts_dir, exist_ok=True)
                                thoughts_path = os.path.join(thoughts_dir, "gemini_thoughts.log")
                                with open(thoughts_path, 'w', encoding='utf-8') as f:
                                    f.write(raw_thoughts)
                        
                        with self.stats_lock:
                            if self._should_retrain():
                                self._retrain_pylaia_model()
                
                total_processed += len(batch_requests)
                logger.info(f"Chunk {chunk_num} complete: Saved {saved_count}/{len(batch_requests)} corrections. Total processed: {total_processed}/{len(image_files)}")
                
                # Cleanup uploaded files after each chunk
                self.cleanup_cloud_files()
            else:
                logger.info(f"Chunk {chunk_num} complete: No images needed LLM processing (all already corrected or skipped)")
            
            # Save checkpoint after each chunk
            self._save_checkpoint()
            self._save_statistics()
        
        logger.info(f"=== PROCESSING COMPLETE ===")
        logger.info(f"Processed {total_processed} images through LLM")
        logger.info(f"Statistics: {json.dumps(self.stats, indent=2)}")

