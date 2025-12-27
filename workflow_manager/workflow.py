import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import signal
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

import ground_truth
from .paleography import PaleographyMatcher
from .post_correction import (
    BayesianConfig,
    NameDatabase,
    process_image_post_correction,
    process_post_correction_results,
    run_non_batch_post_correction,
    get_corrected_lines_for_stitching,
    get_gemini_flash_client,
    find_latest_pylaia_model,
    fill_missing_ctc_losses,
    DB_PATH as BAYESIAN_DB_PATH,
)
from .prompt_builder import (
    build_county_prompt,
    build_step1_prompt,
    build_step2a_prompt,
    build_step2b_prompt,
    build_step3_prompt,
    build_step4_prompt,
)
from .schemas import (
    get_diplomatic_schema,
    get_final_index_schema,
    get_merged_diplomatic_schema,
)
from .settings import (
    ACTIVE_MODEL_DIR,
    API_MAX_RETRIES,
    API_RETRY_DELAY,
    API_TIMEOUT,
    BASE_DIR,
    COST_INPUT_TOKENS,
    COST_OUTPUT_TOKENS,
    GEMINI_API_KEY,
    IMAGE_DIR,
    KRAKEN_ENV,
    LOG_DIR,
    MODEL_TEXT,
    MODEL_VISION,
    OUTPUT_DIR,
    PYLAIA_ARCH,
    PYLAIA_ENV,
    PYLAIA_MODEL,
    PYLAIA_SYMS,
    SURNAME_DB_PATH,
    THINKING_BUDGET,
    WORK_DIR,
    logger,
)
from .utils import clean_json_string, get_cp40_info, repair_json_string

class WorkflowManager:
    """
    Main workflow manager for CP40 plea roll transcription and extraction.

    Orchestrates the complete pipeline from image input to structured JSON output,
    including HTR processing, AI-powered transcription, entity extraction, and validation.
    Uses Gemini 3 Flash Preview with paid API key in non-batch mode for all LLM operations.

    Pipeline sequence:
    1. Kraken (line segmentation)
    2. PyLaia (HTR)
    3. Post-correction and named entity extraction (Gemini 3 Flash Preview + Bayesian)
    4. Stitching (merge transcriptions)
    5. Expansion (expand abbreviations)
    6. Translation (Latin to English)
    7. Indexing (structured extraction)

    Attributes:
        force: If True, reprocesses existing results. Defaults to False.
        use_images: If True, uploads images to AI models. Defaults to True.
        rerun_from_post_pylaia: If True, rerun from post-correction step onwards.
        client: Google Gemini 3 Flash Preview API client for all operations.
        free_client: Same as client (kept for compatibility).
        uploaded_files_cache: Cache of uploaded file references.
        pm: PaleographyMatcher instance for fuzzy text matching.
        known_surnames: Preloaded list of known surnames from database.
        name_db: NameDatabase instance for Bayesian named entity correction.
        bayesian_config: Configuration for Bayesian correction.
    """

    def __init__(self, force=False, use_images=True, rerun_from_post_pylaia=False):
        """
        Initialize the WorkflowManager.

        Args:
            force: If True, reprocesses all images even if results exist. Defaults to False.
            use_images: If True, uploads images to AI vision models. If False, uses text-only mode.
                Defaults to True.
            rerun_from_post_pylaia: If True, rerun everything from post-correction step onwards,
                keeping existing Kraken and PyLaia results. Defaults to False.

        Raises:
            SystemExit: If GEMINI_API_KEY is not configured in settings.
        """
        self.force = force
        self.use_images = use_images
        self.rerun_from_post_pylaia = rerun_from_post_pylaia
        
        # Initialize Bayesian correction components
        self.bayesian_config = BayesianConfig()
        self.name_db = NameDatabase(BAYESIAN_DB_PATH, self.bayesian_config)
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if not GEMINI_API_KEY: 
            logger.critical("FATAL: GEMINI_API_KEY not set. Set environment variable GEMINI_API_KEY with a paid API key")
            sys.exit("FATAL: GEMINI_API_KEY not set.")
            
        # Initialize Gemini 3 Flash Preview client for all tasks
        try:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.free_client = self.client  # Use same client
            logger.info("Initialized Gemini 3 Flash Preview client with paid API key")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
        self.uploaded_files_cache = {} 
        self.pm=PaleographyMatcher()
        
        # Context caching for prompts (to reduce costs)
        self._cached_content_step1 = None  # Cached content for Step 1 diplomatic transcription
        self._cached_content_county = None  # Cached content for county extraction
        
        # Cache for latest PyLaia model paths (lazy-loaded)
        self._pylaia_model_paths = None
        
        # --- PRELOAD DATABASES ---
        self.known_surnames = self._load_list_from_db(SURNAME_DB_PATH, "SELECT DISTINCT surname FROM TblName")
        # Note: places_data.db loading removed - confidence scoring now uses Bayesian probability from post_correction.json

    def _get_or_create_cached_content_step1(self) -> Optional[str]:
        """
        Get or create cached content for Step 1 diplomatic transcription instructions.
        
        Context caching is disabled - always returns None.
        
        Returns:
            None (caching disabled)
        """
        # Context caching disabled
        return None

    def get_ground_truth_from_db(self, roll_number: str, rotulus_number: str) -> List[Dict]:
        """
        Query the CP40 database for ground truth data matching roll and rotulus numbers.

        Extracts comprehensive case data including parties, occupations, status,
        locations, events, and legal details for validation purposes.

        Args:
            roll_number: The roll number to search for (e.g., "565").
            rotulus_number: The rotulus number to search for (e.g., "481").

        Returns:
            A list of dictionaries containing ground truth case data, or an empty list if no matches found.
            Each dictionary includes parties, events, locations, and case details.
            Duplicate cases from multiple references are filtered out.
        """
        return ground_truth.extract_case_data(roll_number, rotulus_number)

    def _load_list_from_db(self, db_path: str, query: str) -> List[str]:
        """
        Load a list of values from a SQLite database query.

        Helper method to preload reference data (e.g., surnames, place names)
        from SQLite databases for use in validation and confidence scoring.

        Args:
            db_path: Path to the SQLite database file.
            query: SQL SELECT query that returns a single column.

        Returns:
            A list of strings from the query results, with whitespace stripped.
            Returns an empty list if the database file doesn't exist or an error occurs.
        """
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found at {db_path}. Corrections will be skipped.")
            return []
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute(query)
            results = [str(r[0]).strip() for r in c.fetchall() if r[0]]
            conn.close()
            logger.info(f"Loaded {len(results)} entries from {db_path}")
            return results
        except sqlite3.Error as e:
            logger.error(f"Database error reading {db_path}: {e}")
            return []

    def _extract_bayesian_probability(
        self, 
        entity_text: str, 
        entity_type: str, 
        post_correction_results: Dict[str, Any]
    ) -> Optional[float]:
        """
        Extract Bayesian probability for an entity from post_correction.json files.
        
        Args:
            entity_text: The entity text to search for (original or anglicized).
            entity_type: Type of entity: 'surname', 'placename', or 'forename'.
            post_correction_results: Dictionary mapping image_name -> post_correction result dict.
        
        Returns:
            Probability value (0.0-1.0) if found, None otherwise.
        """
        entity_text_lower = entity_text.lower().strip()
        
        for img_name, result in post_correction_results.items():
            if not isinstance(result, dict) or "lines" not in result:
                continue
            
            for line in result.get("lines", []):
                # Check the appropriate entity list
                entities = []
                if entity_type == "surname":
                    entities = line.get("surnames", [])
                elif entity_type == "placename":
                    entities = line.get("placenames", [])
                elif entity_type == "forename":
                    entities = line.get("forenames", [])
                
                for entity in entities:
                    # Check original text
                    orig_text = entity.get("original", "").lower().strip()
                    if orig_text == entity_text_lower:
                        best_candidate = entity.get("best_candidate")
                        if best_candidate and "probability" in best_candidate:
                            return best_candidate["probability"]
                    
                    # Check corrected text
                    corrected_text = entity.get("corrected", "").lower().strip()
                    if corrected_text == entity_text_lower:
                        best_candidate = entity.get("best_candidate")
                        if best_candidate and "probability" in best_candidate:
                            return best_candidate["probability"]
                    
                    # Check best_candidate text
                    best_candidate = entity.get("best_candidate")
                    if best_candidate:
                        candidate_text = best_candidate.get("text", "").lower().strip()
                        if candidate_text == entity_text_lower and "probability" in best_candidate:
                            return best_candidate["probability"]
        
        return None

    def calculate_entity_confidence(
        self,
        entity_str: str,
        htr_raw_combined: str,
        step1_raw_combined: str,
        step2_text: str,
        db_reference: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate confidence score for an extracted entity based on multiple attestations.

        Scores entities on a 0-4 point scale based on where they appear:
        - 1 point if found in HTR output
        - 1 point if found in Step 1 (image-based diplomatic transcription)
        - 1 point if found in Step 2 (consensus diplomatic transcription)
        - 1 point if found in reference database (with fuzzy matching)

        Confidence levels:
        - HIGH: 4 points (all sources agree)
        - MEDIUM: 3 points
        - LOW: 2 points
        - VERY-LOW: 0-1 points

        Args:
            entity_str: The entity string to score (e.g., a name or place).
            htr_raw_combined: Combined HTR text from all lines.
            step1_raw_combined: Combined Step 1 diplomatic transcription text.
            step2_text: Step 2 consensus diplomatic transcription text.
            db_reference: Optional list of reference strings from database for validation.

        Returns:
            A dictionary containing:
            - "term": The original entity string
            - "score": Integer score (0-4)
            - "max_score": Always 4
            - "confidence_level": "HIGH", "MEDIUM", "LOW", or "VERY-LOW"
            - "scoring_breakdown": List of sources that contributed points
            - "stats": Dictionary with "in_db" boolean
        """
        if not entity_str:
            return {"score": 0, "confidence_level": "LOW", "details": "empty input"}

        term = entity_str.strip()
        term_lower = term.lower()
        htr_lower = htr_raw_combined.lower()
        step1_lower = step1_raw_combined.lower()
        step2_lower = step2_text.lower()
        
        points = 0  # Base point for existence
        breakdown = ["Extraction"]

        if term_lower in htr_lower:
            points += 1
            breakdown.append("HTR")
        if term_lower in step1_lower:
            points += 1
            breakdown.append("Img_Diplomatic")
        if term_lower in step2_lower:
            points += 1
            breakdown.append("Consensus_Diplomatic")

        in_db = False
        if db_reference:
            # Check for exact match or very close fuzzy match (weighted dist <= 1.0)
            best_match, dist = self._find_best_match(term, db_reference, max_weighted_distance=1.0)
            if best_match and dist <= 1.0:
                points += 1
                breakdown.append("Database")
                in_db = True
            
        if points == 4: confidence_level = "HIGH"
        elif points >= 3: confidence_level = "MEDIUM"
        elif points >= 2: confidence_level = "LOW"
        else: confidence_level = "VERY-LOW"
            
        return {
            "term": term,
            "score": points,
            "max_score": 4,
            "confidence_level": confidence_level,
            "scoring_breakdown": breakdown,
            "stats": {"in_db": in_db}
        }

    def _find_best_match(
        self, target_word: str, reference_list: List[str], max_weighted_distance: float = 3.0
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching string in a reference list using weighted Levenshtein distance.

        Uses paleographic-aware fuzzy matching optimized for medieval Latin text.
        First checks for exact matches, then finds the closest fuzzy match within
        the specified distance threshold.

        Args:
            target_word: The word to match.
            reference_list: List of candidate strings to search.
            max_weighted_distance: Maximum weighted distance for a valid match. Defaults to 3.0.

        Returns:
            A tuple of (best_match, distance):
            - best_match: The closest matching string, or None if no match within threshold.
            - distance: The weighted Levenshtein distance (999 if no match found).
        """
        if not reference_list or not target_word:
            return None, 999
        
        # 1. Exact match check (Fastest)
        # We assume reference_list contains clean strings
        if target_word in reference_list:
            return target_word, 0

        # 2. Fuzzy match
        best_match = None
        min_dist = float('inf')
        
        target_lower = target_word.lower()
        
        for candidate in reference_list:
            # Optimization: Skip if length difference is already greater than max_distance
            if abs(len(candidate) - len(target_lower)) > max_weighted_distance:
                continue

            dist = self.pm.weighted_levenshtein(target_lower, candidate.lower())
            
            # Update best match if this is closer
            if dist < min_dist:
                min_dist = dist
                best_match = candidate
                
                # Optimization: If we find a distance of 1, it's unlikely we beat it.
                if min_dist == 1:
                    break
        
        # 3. Threshold enforcement
        if min_dist > max_weighted_distance:
            return None, min_dist

        return best_match, min_dist 
    def cleanup_cloud_files(self) -> None:
        """
        Clean up all uploaded files from Google Gemini cloud storage.

        Removes all files that were uploaded for vision model processing.
        Useful for managing cloud storage quotas and costs.
        Uses parallel deletion for faster cleanup.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info("--- STARTING CLOUD STORAGE CLEANUP ---")
        deleted_count = 0
        error_count = 0
        
        # Helper function to delete a single file
        def delete_file(file_name: str) -> bool:
            """Delete a single file. Returns True if successful, False otherwise."""
            try:
                self.client.files.delete(name=file_name)
                return True
            except Exception as e:
                logger.debug(f"Error deleting file {file_name}: {e}")
                return False
        
        try:
            logger.info("Listing files from client...")
            file_iterator = self.client.files.list()
            files_list = list(file_iterator)  # Convert to list to get count
            total_files = len(files_list)
            logger.info(f"Found {total_files} files to delete")
            
            if total_files > 0:
                # Delete files in parallel
                max_workers = min(20, total_files)  # Limit to 20 concurrent deletions
                logger.info(f"Deleting {total_files} files in parallel (max {max_workers} workers)...")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all deletion tasks
                    future_to_file = {
                        executor.submit(delete_file, f.name): f.name
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
                            logger.info(f"Deleted {completed}/{total_files} files...")
        except Exception as e:
            logger.error(f"Error during cloud storage cleanup: {e}")
        
        logger.info(f"--- CLOUD STORAGE CLEANUP COMPLETE: Deleted {deleted_count} files, {error_count} errors ---")

    def _run_command(self, command_str: str, description: str) -> None:
        """
        Execute a shell command and log the results.

        Runs a command in a bash shell and logs success/failure.
        Used for executing HTR tools (Kraken, PyLaia) in their virtual environments.

        Args:
            command_str: The shell command to execute.
            description: Human-readable description for logging purposes.
        """
        logger.debug(f"EXEC CMD [{description}]: {command_str}")
        try:
            result = subprocess.run(
                command_str, 
                shell=True, 
                executable='/bin/bash', 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                logger.error(f"FAILED [{description}] Return Code: {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
            else:
                logger.debug(f"SUCCESS [{description}]")
        except Exception as e:
            logger.exception(f"Exception running command [{description}]")

    def _get_latest_pylaia_model_paths(self) -> Tuple[str, str, str]:
        """
        Get the latest PyLaia model paths from bootstrap_training_data/pylaia_models.
        Copies the latest model (checkpoint, model file, and syms.txt) to the active model directory
        and uses those paths for subsequent work.
        
        Returns:
            A tuple of (checkpoint_path, model_arch_path, syms_path) as strings.
        """
        if self._pylaia_model_paths is None:
            try:
                checkpoint, model_file, syms_file = find_latest_pylaia_model()
                
                # Create active model directory if it doesn't exist
                active_dir = Path(ACTIVE_MODEL_DIR)
                active_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Active model directory created/verified: {active_dir.absolute()}")
                
                # Copy checkpoint to active directory
                checkpoint_dest = active_dir / checkpoint.name
                if not checkpoint_dest.exists() or checkpoint.stat().st_mtime > checkpoint_dest.stat().st_mtime:
                    logger.info(f"Copying checkpoint to active model directory: {checkpoint.name} -> {checkpoint_dest}")
                    shutil.copy2(checkpoint, checkpoint_dest)
                else:
                    logger.debug(f"Checkpoint already up to date in active directory: {checkpoint.name}")
                
                # Copy model file to active directory (handle both file and directory cases)
                model_dest = active_dir / "model"
                model_path = Path(model_file)
                if model_path.is_dir():
                    # If model is a directory, use copytree
                    # Check if we need to update by comparing source directory mtime
                    needs_update = True
                    if model_dest.exists() and model_dest.is_dir():
                        try:
                            source_mtime = model_path.stat().st_mtime
                            dest_mtime = model_dest.stat().st_mtime
                            if source_mtime <= dest_mtime:
                                needs_update = False
                        except (OSError, AttributeError):
                            # If we can't compare, update anyway
                            needs_update = True
                    
                    if needs_update:
                        if model_dest.exists():
                            shutil.rmtree(model_dest)
                        logger.info(f"Copying model directory to active model directory")
                        shutil.copytree(model_path, model_dest)
                    else:
                        logger.debug(f"Model directory already up to date in active directory")
                else:
                    # If model is a file, use copy2
                    if not model_dest.exists() or model_path.stat().st_mtime > model_dest.stat().st_mtime:
                        logger.info(f"Copying model file to active model directory")
                        shutil.copy2(model_path, model_dest)
                    else:
                        logger.debug(f"Model file already up to date in active directory")
                
                # Copy syms.txt to active directory
                syms_dest = active_dir / "syms.txt"
                if not syms_dest.exists() or syms_file.stat().st_mtime > syms_dest.stat().st_mtime:
                    logger.info(f"Copying syms.txt to active model directory")
                    shutil.copy2(syms_file, syms_dest)
                else:
                    logger.debug(f"syms.txt already up to date in active directory")
                
                # Use paths from active directory
                self._pylaia_model_paths = (
                    str(checkpoint_dest),
                    str(model_dest),
                    str(syms_dest)
                )
                logger.info(f"Using latest PyLaia model for HTR from active directory: {checkpoint.parent.name}/{checkpoint.name}")
                logger.info(f"Active model files location: {active_dir.absolute()}")
            except Exception as e:
                logger.warning(f"Failed to find latest PyLaia model: {e}. Falling back to hardcoded model_v10.")
                # Fallback to hardcoded paths from settings
                self._pylaia_model_paths = (
                    PYLAIA_MODEL,
                    PYLAIA_ARCH,
                    PYLAIA_SYMS
                )
        return self._pylaia_model_paths

    def run_htr_tools(self, image_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Run HTR (Handwritten Text Recognition) tools on an image.

        Executes the complete HTR pipeline:
        1. Kraken segmentation (line detection and layout analysis)
        2. Line preprocessing (normalization for PyLaia)
        3. PyLaia recognition (text extraction from normalized lines with word confidence scores)

        Results are cached unless force=True. Intermediate files are saved in
        a subdirectory named after the image basename.

        Args:
            image_path: Path to the input manuscript image.
            output_dir: Directory where HTR results will be saved.

        Returns:
            A tuple of (kraken_json_path, htr_txt_path):
            - kraken_json_path: Path to Kraken segmentation JSON file.
            - htr_txt_path: Path to PyLaia recognition results text file (includes word confidence scores).
        """
        basename = os.path.splitext(os.path.basename(image_path))[0]
        part_dir = os.path.join(output_dir, basename)
        os.makedirs(part_dir, exist_ok=True)
        
        kraken_json = os.path.join(part_dir, "kraken.json")
        lines_dir = os.path.join(part_dir, "lines")
        os.makedirs(lines_dir, exist_ok=True)
        list_txt = os.path.join(part_dir, "img_list.txt")
        htr_res = os.path.join(part_dir, "htr.txt")

        if self.force or not (os.path.exists(htr_res) and os.path.getsize(htr_res) > 0):
            cmd_kraken = f"source {KRAKEN_ENV} && kraken -i '{image_path}' '{kraken_json}' --device cuda:0 segment -bl"
            self._run_command(cmd_kraken, "Kraken Segmentation")

            if not os.path.exists(kraken_json): return kraken_json, htr_res

            preprocess_script = os.path.join(BASE_DIR, "preprocess_lines_greyscale.py")
            cmd_preprocess = f"source {PYLAIA_ENV} && python3 '{preprocess_script}' '{image_path}' '{kraken_json}' '{lines_dir}' '{list_txt}'"
            self._run_command(cmd_preprocess, "Preprocess Lines")

            if os.path.exists(list_txt) and os.path.getsize(list_txt) > 0:
                # Get latest PyLaia model paths
                pylaia_checkpoint, pylaia_arch, pylaia_syms = self._get_latest_pylaia_model_paths()
                cmd_decode = (
                    f"source {PYLAIA_ENV} && "
                    f"pylaia-htr-decode-ctc "
                    f"--trainer.accelerator gpu "
                    f"--trainer.devices 1 "
                    f"--common.checkpoint '{pylaia_checkpoint}' "
                    f"--common.model_filename '{pylaia_arch}' "
                    f"--decode.include_img_ids true "
                    f"--decode.print_word_confidence_score true "
                    f"'{pylaia_syms}' '{list_txt}' > '{htr_res}'"
                )
                self._run_command(cmd_decode, "PyLaia Decode")
            else:
                with open(htr_res, 'w') as f: f.write("")
        return kraken_json, htr_res

    def merge_htr_data(
        self, image_path: str, kraken_json: str, htr_txt: str
    ) -> List[Dict[str, Any]]:
        """
        Merge HTR recognition text with layout geometry data.

        Combines PyLaia recognition results with Kraken segmentation data to create
        structured line data with both text and spatial information. Each line includes:
        - Line ID
        - HTR text
        - Word-level confidence scores (probability of correctness for each word)
        - Bounding box coordinates [ymin, xmin, ymax, xmax]
        - Polygon boundary coordinates

        Lines are sorted vertically by their top coordinate (ymin).

        Args:
            image_path: Path to the source image (used for reference).
            kraken_json: Path to Kraken segmentation JSON file.
            htr_txt: Path to PyLaia recognition results text file (with confidence scores).

        Returns:
            A list of dictionaries, each representing a text line with:
            - "line_id": Line identifier
            - "htr_text": Recognized text
            - "word_confidences": List of dicts with "word" and "confidence" (float 0-1 or None)
            - "bbox": Bounding box [ymin, xmin, ymax, xmax] or None
            - "polygon": List of [x, y] coordinate pairs or None
            Returns an empty list if files cannot be read or parsed.
        """
        try:
            with open(kraken_json, 'r') as f: 
                layout = json.load(f)
                layout_lines = layout.get('lines', [])
        except: return []

        # Map ID to Geometry
        # Import BBOX_LEFT_EXTENSION to extend bounding boxes to the left
        # This ensures bounding boxes passed to LLM reflect the expanded width used by PyLaia
        try:
            from line_preprocessor_greyscale.config import BBOX_LEFT_EXTENSION
        except ImportError:
            try:
                from line_preprocessor.config import BBOX_LEFT_EXTENSION
            except ImportError:
                # Fallback if config is not available
                BBOX_LEFT_EXTENSION = 200
        
        geo_map = {}
        for l in layout_lines:
            boundary = l.get('boundary') # This is the polygon [[x,y], [x,y]]
            bbox = None
            extended_polygon = boundary
            
            if boundary and isinstance(boundary, list) and len(boundary) > 0:
                try:
                    # Extend polygon to the left by BBOX_LEFT_EXTENSION pixels
                    # This ensures bounding boxes match the expanded width used by PyLaia
                    extended_polygon = [[max(0, pt[0] - BBOX_LEFT_EXTENSION), pt[1]] for pt in boundary]
                    
                    xs = [pt[0] for pt in extended_polygon]
                    ys = [pt[1] for pt in extended_polygon]
                    # Store as [ymin, xmin, ymax, xmax] with extended coordinates
                    bbox = [int(min(ys)), int(min(xs)), int(max(ys)), int(max(xs))]
                except Exception:
                    pass
            
            geo_map[l['id']] = {
                "bbox": bbox,
                "polygon": extended_polygon  # Store extended polygon for consistency
            }

        structured = []
        if os.path.exists(htr_txt):
            with open(htr_txt, 'r') as f:
                htr_lines = f.readlines()
                for line in htr_lines:
                    # Parse line format: <filename> ['score1', 'score2', ...] <text>
                    # or fallback to: <filename> <text> (if confidence scores not available)
                    
                    # Try to find confidence scores pattern: ['0.95', '0.87', ...]
                    confidence_match = re.search(r"\[(['\"]?[\d.]+['\"]?[\s,]*)+(['\"]?[\d.]+['\"]?)\]", line)
                    
                    if confidence_match:
                        # Format with confidence scores
                        conf_str = confidence_match.group(0)
                        conf_start = confidence_match.start()
                        conf_end = confidence_match.end()
                        
                        # Extract filename (before confidence scores)
                        filename_part = line[:conf_start].strip()
                        # Extract text (after confidence scores)
                        text_part = line[conf_end:].strip()
                        
                        # Parse confidence scores
                        conf_values = re.findall(r"['\"]?([\d.]+)['\"]?", conf_str)
                        confidence_scores = [float(c) for c in conf_values]
                        
                        # Find filename with extension
                        match = re.search(r'([^\s]+\.(png|jpg|jpeg))', filename_part, re.IGNORECASE)
                        if match:
                            filename_path = match.group(1)
                        else:
                            # Fallback: split on space
                            parts = filename_part.split(' ', 1)
                            if len(parts) > 0:
                                filename_path = parts[0]
                            else:
                                continue
                    else:
                        # Format without confidence scores (backward compatibility)
                        match = re.search(r'(\.png|\.jpg|\.jpeg)\s', line, re.IGNORECASE)
                        if match:
                            split_idx = match.end() - 1
                            filename_path = line[:split_idx].strip()
                            text_part = line[split_idx:].strip()
                        else:
                            parts = line.strip().split(' ', 1)
                            if len(parts) < 2: continue
                            filename_path, text_part = parts[0], parts[1]
                        confidence_scores = []
                    
                    file_id = os.path.splitext(os.path.basename(filename_path))[0]
                    if file_id in geo_map:
                        # Clean text: replace <space> markers
                        text = text_part.replace('<space>', '§').replace(' ', '').replace('§', ' ')
                        
                        # Split text into words and match with confidence scores
                        words = text.split()
                        word_confidences = []
                        
                        if confidence_scores and len(confidence_scores) == len(words):
                            # Perfect match: one confidence per word
                            word_confidences = [
                                {"word": word, "confidence": conf}
                                for word, conf in zip(words, confidence_scores)
                            ]
                        elif confidence_scores:
                            # Mismatch: try to align as best as possible
                            # This can happen if there are special characters or spacing issues
                            for i, word in enumerate(words):
                                if i < len(confidence_scores):
                                    word_confidences.append({
                                        "word": word,
                                        "confidence": confidence_scores[i]
                                    })
                                else:
                                    # No confidence available for this word
                                    word_confidences.append({
                                        "word": word,
                                        "confidence": None
                                    })
                        else:
                            # No confidence scores available
                            word_confidences = [
                                {"word": word, "confidence": None}
                                for word in words
                            ]
                        
                        geo_info = geo_map[file_id]
                        structured.append({
                            "line_id": file_id,
                            "htr_text": text,
                            "word_confidences": word_confidences,
                            "bbox": geo_info['bbox'],  # Extended bounding box reflecting expanded width
                            "polygon": geo_info['polygon']  # Extended polygon matching PyLaia processing
                        })

        # Sort by vertical position (ymin)
        def get_y_min(item): 
            return item['bbox'][0] if item.get('bbox') else 0
            
        structured.sort(key=get_y_min)
        return structured

    def upload_file(self, path: str) -> Optional[Any]:
        """
        Upload an image file to Google Gemini cloud storage with retry logic.

        Uploads are cached to avoid re-uploading the same file. The method waits
        for the file to become active before returning. Includes automatic retry
        for timeout errors.

        Args:
            path: Path to the image file to upload (PNG or JPEG).

        Returns:
            A file reference object if upload succeeds, None if upload fails.
            The file reference can be used in API calls to reference the uploaded image.
        """
        if path in self.uploaded_files_cache:
            return self.uploaded_files_cache[path]
        
        logger.info(f"Uploading {path} to Gemini...")
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        
        # Retry upload if it times out
        last_exception = None
        for attempt in range(API_MAX_RETRIES):
            try:
                f = self.client.files.upload(file=path, config=types.UploadFileConfig(mime_type=mime))
                
                # Wait for file to be active
                while f.state.name == "PROCESSING": 
                    time.sleep(1)
                    f = self.client.files.get(name=f.name)
                
                if f.state.name != "ACTIVE":
                    logger.error(f"File upload failed for {path}: {f.state.name}")
                    return None
                    
                self.uploaded_files_cache[path] = f
                return f
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if it's a retryable error (timeout, write timeout)
                is_timeout = any(keyword in error_msg for keyword in [
                    'timeout', 'timed out', 'deadline exceeded'
                ])
                
                if is_timeout and attempt < API_MAX_RETRIES - 1:
                    wait_time = API_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"File upload timed out (attempt {attempt + 1}/{API_MAX_RETRIES}). "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    if is_timeout:
                        logger.error(f"File upload failed after {API_MAX_RETRIES} attempts due to timeout")
                    else:
                        logger.error(f"File upload failed: {e}")
                    return None
        
        return None

    def _call_api_with_retry(self, client, model_name, contents, config):
        """
        Call the Gemini API with retry logic for timeout/504 errors.
        
        Args:
            client: The Gemini API client instance
            model_name: Name of the model to use
            contents: Content to send
            config: Generation configuration
            
        Returns:
            API response object
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(API_MAX_RETRIES):
            try:
                # Use Gemini API format
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                return response
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if it's a retryable error (timeout, 504, deadline exceeded)
                is_timeout = any(keyword in error_msg for keyword in [
                    '504', 'timeout', 'deadline exceeded', 'timed out'
                ])
                
                if is_timeout and attempt < API_MAX_RETRIES - 1:
                    # Exponential backoff: wait longer each time
                    wait_time = API_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"API call failed with timeout/504 error (attempt {attempt + 1}/{API_MAX_RETRIES}). "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    # Either not a timeout error, or we're out of retries
                    if is_timeout:
                        logger.error(f"API call failed after {API_MAX_RETRIES} attempts due to timeout/504 errors")
                    raise
        
        # If we get here, all retries failed
        raise last_exception
    
    def _truncate_thoughts(self, raw_thoughts: str, batch_key: str = None) -> str:
        """
        Truncate thoughts log to prevent infinite loops and excessive size.
        
        Args:
            raw_thoughts: The raw thoughts text to truncate.
            batch_key: Optional batch key for logging.
        
        Returns:
            Truncated thoughts text with a note if truncation occurred.
        """
        if not raw_thoughts:
            return raw_thoughts
        
        # Truncate thoughts if they exceed reasonable length (500KB)
        MAX_THOUGHTS_SIZE = 500 * 1024  # 500KB
        truncated_thoughts = raw_thoughts
        
        if len(raw_thoughts.encode('utf-8')) > MAX_THOUGHTS_SIZE:
            # Find a good truncation point (end of a line)
            truncated_bytes = raw_thoughts.encode('utf-8')[:MAX_THOUGHTS_SIZE]
            # Try to decode and find last newline
            try:
                truncated_text = truncated_bytes.decode('utf-8', errors='ignore')
                last_newline = truncated_text.rfind('\n')
                if last_newline > MAX_THOUGHTS_SIZE * 0.9:  # If we found a newline in last 10%
                    truncated_thoughts = truncated_text[:last_newline + 1]
                else:
                    truncated_thoughts = truncated_text
            except:
                truncated_thoughts = truncated_bytes.decode('utf-8', errors='ignore')
            
            truncated_thoughts += f"\n\n[THOUGHTS TRUNCATED - Original size: {len(raw_thoughts)} bytes, truncated at: {len(truncated_thoughts.encode('utf-8'))} bytes]\n"
            logger.warning(f"Thoughts log truncated for {batch_key or 'unknown'}: {len(raw_thoughts)} -> {len(truncated_thoughts)} bytes")
        
        # Detect repetitive patterns (same line repeated many times)
        lines = truncated_thoughts.split('\n')
        if len(lines) > 1000:  # If more than 1000 lines
            # Check for repetitive patterns in last 500 lines
            last_500 = lines[-500:]
            line_counts = {}
            for line in last_500:
                line_stripped = line.strip()
                if line_stripped and len(line_stripped) > 20:  # Only count substantial lines
                    line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
            
            # If any line appears more than 50 times, it's likely a loop
            max_repeats = max(line_counts.values()) if line_counts else 0
            if max_repeats > 50:
                # Find where the repetition starts
                repetitive_line = next((line for line, count in line_counts.items() if count > 50), None)
                if repetitive_line:
                    # Find first occurrence of this repetitive pattern
                    first_repeat_idx = None
                    for i, line in enumerate(lines):
                        if line.strip() == repetitive_line:
                            first_repeat_idx = i
                            break
                    
                    if first_repeat_idx and first_repeat_idx < len(lines) - 100:
                        # Truncate before the repetition starts
                        truncated_thoughts = '\n'.join(lines[:first_repeat_idx])
                        truncated_thoughts += f"\n\n[THOUGHTS TRUNCATED - Detected repetitive pattern starting at line {first_repeat_idx + 1}. Same line repeated {max_repeats} times in last 500 lines.]\n"
                        logger.warning(f"Thoughts log truncated due to repetitive pattern for {batch_key or 'unknown'}: line '{repetitive_line[:50]}...' repeated {max_repeats} times")
        
        return truncated_thoughts
    
    def _generate_and_log(
        self, model_name: str, parts: List[Any], config: Any, log_filepath: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate content using AI model with thinking logs enabled.

        Calls the Gemini API with thinking mode enabled, separates thought blocks
        from response text, logs everything to a file, and returns the clean
        response text along with token usage information.
        
        Includes automatic retry logic with exponential backoff for timeout/504 errors.

        Args:
            model_name: Name of the Gemini model to use.
            parts: List of content parts (text and/or file references).
            config: Generation configuration object.
            log_filepath: Path where the full log (thoughts + response) will be saved.

        Returns:
            Tuple of (clean response text, token usage dictionary).
            Token usage dict contains: prompt_tokens, cached_tokens, response_tokens,
            thoughts_tokens, total_tokens, non_cached_input.

        Raises:
            Exception: If API call fails after all retries, the exception is logged and re-raised.
        """
        # Enable Thinking - preserve existing thinking_config if already set
        # Otherwise, set default thinking_config with Minimal thinking level
        if not hasattr(config, 'thinking_config') or config.thinking_config is None:
            config.thinking_config = types.ThinkingConfig(include_thoughts=True, thinking_level="LOW")
        # If thinking_config is already set, preserve it
        
        try:
            # Use cached content for Step 1 diplomatic transcription if available
            cached_content_name = None
            use_cached = False
            if "diplomatic" in str(log_filepath).lower() or "step1" in str(log_filepath).lower():
                cached_content_name = self._get_or_create_cached_content_step1()
                if cached_content_name:
                    # Check if parts contain the full prompt or just variable parts
                    # If it contains "## Input HTR:", we can use cached content
                    full_text = ""
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            full_text += part.text
                    if "## Input HTR:" in full_text and cached_content_name:
                        use_cached = True
            
            # Build request with optional cached content
            if use_cached and cached_content_name:
                # Use cached content - extract only variable parts (HTR JSON + image)
                variable_parts = []
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        # Extract only the variable HTR section
                        if "## Input HTR:" in part.text:
                            # Extract just the HTR JSON part
                            htr_section = part.text.split("## Input HTR:")[-1].strip()
                            variable_parts.append(types.Part.from_text(text=f"## Input HTR:\n\n{htr_section}"))
                    elif hasattr(part, 'file_uri') or (hasattr(part, 'file_data') and part.file_data):
                        variable_parts.append(part)  # Keep image parts
                
                if variable_parts:
                    logger.info(f"Using cached content: {cached_content_name}")
                    # Create new config with cached_content, preserving all original config parameters
                    request_config_dict = {
                        "cached_content": cached_content_name,
                        "thinking_config": config.thinking_config
                    }
                    # Preserve other config parameters
                    for attr in ['response_mime_type', 'response_schema', 'temperature']:
                        if hasattr(config, attr):
                            value = getattr(config, attr)
                            if value is not None:
                                request_config_dict[attr] = value
                    request_config = types.GenerateContentConfig(**request_config_dict)
                    response = self._call_api_with_retry(
                        client=self.client,
                        model_name=model_name,
                        contents=[types.Content(parts=variable_parts)],
                        config=request_config
                    )
                    # Log usage metadata to verify caching is working
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        um = response.usage_metadata
                        prompt_tokens = getattr(um, 'prompt_token_count', 0)
                        cached_tokens = getattr(um, 'cached_content_token_count', 0)
                        total_tokens = getattr(um, 'total_token_count', 0)
                        if cached_tokens > 0:
                            savings_pct = (cached_tokens / (prompt_tokens + cached_tokens)) * 100 if (prompt_tokens + cached_tokens) > 0 else 0
                            logger.info(
                                f"Cache usage: {cached_tokens:,} cached tokens, {prompt_tokens:,} prompt tokens, "
                                f"{total_tokens:,} total tokens ({savings_pct:.1f}% from cache)"
                            )
                        else:
                            logger.warning("Cached content was used but cached_content_token_count is 0 - cache may not be working")
                else:
                    # Fallback to regular call
                    response = self._call_api_with_retry(
                        client=self.client,
                        model_name=model_name,
                        contents=[types.Content(parts=parts)],
                        config=config
                    )
            else:
                # Regular call without cached content
                response = self._call_api_with_retry(
                    client=self.client,
                    model_name=model_name,
                    contents=[types.Content(parts=parts)],
                    config=config
                )
            
            full_log_content = []
            clean_text_content = []

            # Iterate through parts to separate Thoughts from Response
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    # Check for thought parameter (SDK dependent) or infer structure
                    is_thought = getattr(part, 'thought', False)
                    
                    if is_thought:
                        full_log_content.append(f"--- THOUGHT BLOCK ---\n{part.text}\n")
                    else:
                        full_log_content.append(f"--- RESPONSE ---\n{part.text}\n")
                        clean_text_content.append(part.text)
            
            # Save the log (with truncation to prevent infinite loops)
            log_content = "\n".join(full_log_content)
            # Use truncation method if available, otherwise write directly
            if hasattr(self, '_truncate_thoughts'):
                truncated_log = self._truncate_thoughts(log_content, log_filepath)
            else:
                # Fallback: simple truncation if method doesn't exist
                MAX_SIZE = 500 * 1024  # 500KB
                if len(log_content.encode('utf-8')) > MAX_SIZE:
                    truncated_log = log_content.encode('utf-8')[:MAX_SIZE].decode('utf-8', errors='ignore')
                    truncated_log += f"\n\n[THOUGHTS TRUNCATED - Original size: {len(log_content)} bytes]\n"
                else:
                    truncated_log = log_content
            with open(log_filepath, "w", encoding="utf-8") as f:
                f.write(truncated_log)
            
            # Extract token usage information
            token_usage = {
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "response_tokens": 0,
                "thoughts_tokens": 0,
                "total_tokens": 0,
                "non_cached_input": 0
            }
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                um = response.usage_metadata
                # Safely extract token counts, handling None values
                def safe_int(value):
                    """Convert value to int, defaulting to 0 if None or invalid."""
                    if value is None:
                        return 0
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return 0
                
                token_usage["prompt_tokens"] = safe_int(getattr(um, 'prompt_token_count', 0))
                token_usage["cached_tokens"] = safe_int(getattr(um, 'cached_content_token_count', 0))
                token_usage["response_tokens"] = safe_int(getattr(um, 'candidates_token_count', 0))
                token_usage["thoughts_tokens"] = safe_int(getattr(um, 'thoughts_token_count', 0))
                token_usage["total_tokens"] = safe_int(getattr(um, 'total_token_count', 0))
                token_usage["non_cached_input"] = token_usage["prompt_tokens"] - token_usage["cached_tokens"]
                
            return "".join(clean_text_content), token_usage

        except Exception as e:
            logger.error(f"LLM Generation failed for {log_filepath}: {e}")
            raise e
    # --- PROMPT GENERATORS ---
    
    def prompt_step1a_county_extraction(self, image_path: str) -> List[Any]:
        """
        Build prompt for county extraction from marginal annotations.

        Creates a prompt that instructs the AI to identify the county name
        from the left margin of the plea roll image.

        Args:
            image_path: Path to the manuscript image.

        Returns:
            List of prompt parts (text + optional image reference).
        """
        return build_county_prompt(image_path, self.upload_file, self.use_images)


    def prompt_step1_diplomatic(self, lines: List[Dict], image_path: str) -> List[Any]:
        """
        Build prompt for diplomatic transcription (Step 1).

        Creates a comprehensive prompt for AI-assisted paleographic transcription
        that preserves medieval orthography, abbreviations, and special characters.

        Args:
            lines: List of line dictionaries with HTR text and geometry.
            image_path: Path to the manuscript image.

        Returns:
            List of prompt parts (text + optional image reference).
        """
        return build_step1_prompt(lines, image_path, self.upload_file, self.use_images)

    def prompt_step3_translation(self, latin_text: str) -> List[Any]:
        """
        Build prompt for translating diplomatic Latin to English (Step 3).

        Args:
            latin_text: The diplomatic Latin text to translate.

        Returns:
            List of prompt parts (text only).
        """
        return build_step3_prompt(latin_text)

    def _get_batch_cache_path(self, batch_key: str, output_dir: str = None) -> str:
        """
        Get the path to the batch ID cache file for a given batch key.
        
        Args:
            batch_key: Unique identifier for the batch request.
            output_dir: Optional output directory. If not provided, uses WORK_DIR.
        
        Returns:
            Path to the batch cache JSON file.
        """
        safe_key = re.sub(r'[^a-zA-Z0-9]', '_', batch_key)
        cache_dir = output_dir if output_dir else WORK_DIR
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"batch_cache_{safe_key}.json")
    
    def _get_cached_batch_id(self, batch_key: str, output_dir: str = None) -> Optional[str]:
        """
        Retrieve a cached batch job ID if it exists.
        
        Args:
            batch_key: Unique identifier for the batch request.
            output_dir: Optional output directory. If not provided, uses WORK_DIR.
        
        Returns:
            Batch job name (ID) if found, None otherwise.
        """
        cache_path = self._get_batch_cache_path(batch_key, output_dir)
        if os.path.exists(cache_path) and not self.force:
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                    job_name = cache_data.get('job_name')
                    if job_name:
                        # Verify the batch job still exists and is not completed
                        try:
                            batch_job = self.client.batches.get(name=job_name)
                            completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
                            def get_state_name(batch_job):
                                if hasattr(batch_job.state, 'name'):
                                    return batch_job.state.name
                                elif isinstance(batch_job.state, str):
                                    return batch_job.state
                                else:
                                    return str(batch_job.state)
                            current_state = get_state_name(batch_job)
                            if current_state not in completed_states:
                                logger.info(f"[{batch_key}] Found cached batch ID: {job_name}, resuming polling")
                                return job_name
                            elif current_state == 'JOB_STATE_SUCCEEDED':
                                logger.info(f"[{batch_key}] Cached batch {job_name} is already completed ({current_state}), retrieving and processing results")
                                return job_name
                            else:
                                logger.info(f"[{batch_key}] Cached batch {job_name} is in completed state ({current_state}), creating new batch")
                        except Exception as e:
                            logger.warning(f"[{batch_key}] Cached batch {job_name} no longer exists: {e}")
                return None
            except Exception as e:
                logger.warning(f"[{batch_key}] Failed to read batch cache: {e}")
        return None
    
    def _save_batch_id(self, batch_key: str, job_name: str, output_dir: str = None) -> None:
        """
        Save a batch job ID to cache.
        
        Args:
            batch_key: Unique identifier for the batch request.
            job_name: The batch job name (ID) to cache.
            output_dir: Optional output directory. If not provided, uses WORK_DIR.
        """
        cache_path = self._get_batch_cache_path(batch_key, output_dir)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "job_name": job_name,
                    "batch_key": batch_key,
                    "created_at": time.time()
                }, f, indent=2)
            logger.debug(f"[{batch_key}] Saved batch ID to cache: {job_name}")
        except Exception as e:
            logger.warning(f"[{batch_key}] Failed to save batch cache: {e}")

    def _generate_batch_single(
        self, model_name: str, parts: List[Any], config: Any, log_filepath: str, batch_key: str, output_dir: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate content using batch API for a single request.
        
        Wraps a single request in a batch API call. Useful for Steps 2a, 2b, 3, and 4.
        
        Args:
            model_name: Name of the Gemini model to use.
            parts: List of content parts (text and/or file references).
            config: Generation configuration object.
            log_filepath: Path where the full log (thoughts + response) will be saved.
            batch_key: Unique key for this batch request.
        
        Returns:
            Tuple of (clean response text, token usage dictionary).
        """
        import tempfile
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp())
        jsonl_file = temp_dir / "batch_request.jsonl"
        
        try:
            # Convert parts to dict format
            parts_dict = []
            for part in parts:
                if hasattr(part, 'text') and part.text:
                    parts_dict.append({"text": part.text})
                elif hasattr(part, 'file_data') and part.file_data:
                    if hasattr(part.file_data, 'file_uri'):
                        parts_dict.append({
                            "file_data": {
                                "mime_type": part.file_data.mime_type if hasattr(part.file_data, 'mime_type') else "image/png",
                                "file_uri": part.file_data.file_uri
                            }
                        })
            
            # Build generation config dict
            # NOTE: response_schema is NOT included for batch API - it causes INVALID_ARGUMENT errors
            # The batch API doesn't support dict-format schemas in JSONL files
            # We'll request JSON format and parse from text response instead
            gen_config = {
                "temperature": getattr(config, 'temperature', 0.0),
                "max_output_tokens": getattr(config, 'max_output_tokens', 8192) or 32768,  # Use larger default for batch
            }
            
            if hasattr(config, 'response_mime_type') and config.response_mime_type:
                gen_config["response_mime_type"] = config.response_mime_type
            else:
                # Default to JSON if response_schema was requested (even though we can't include schema)
                if hasattr(config, 'response_schema') and config.response_schema:
                    gen_config["response_mime_type"] = "application/json"
            
            # Don't include response_schema - batch API doesn't support it
            # The prompt should instruct the model to return the correct JSON format
            
            if hasattr(config, 'thinking_config') and config.thinking_config:
                gen_config["thinking_config"] = {
                    "include_thoughts": config.thinking_config.include_thoughts,
                    "thinking_level": config.thinking_config.thinking_level
                }
            
            # Build request object
            request_obj = {
                "key": batch_key,
                "request": {
                    "contents": [{"parts": parts_dict}],
                    "generation_config": gen_config
                }
            }
            
            # Write to JSONL
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(request_obj) + "\n")
            
            # Check for cached batch ID first
            cached_job_name = self._get_cached_batch_id(batch_key, output_dir)
            batch_job = None
            
            if cached_job_name:
                try:
                    batch_job = self.client.batches.get(name=cached_job_name)
                    logger.info(f"[{batch_key}] Resuming cached batch: {cached_job_name}")
                except Exception as e:
                    logger.warning(f"[{batch_key}] Failed to resume cached batch, creating new: {e}")
                    batch_job = None
            
            # Upload and create batch if not resuming
            if not batch_job:
                try:
                    uploaded_file = self.client.files.upload(
                        file=str(jsonl_file),
                        config=types.UploadFileConfig(mime_type='application/jsonl')
                    )
                except Exception as e:
                    # Log the JSONL content for debugging
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        jsonl_content = f.read()
                    logger.error(f"Failed to upload batch JSONL file. Content (first 2000 chars): {jsonl_content[:2000]}")
                    raise
                
                try:
                    batch_job = self.client.batches.create(
                        model=model_name,
                        src=uploaded_file.name,
                        config={"display_name": f"batch_{batch_key}"}
                    )
                    # Save batch ID to cache
                    self._save_batch_id(batch_key, batch_job.name, output_dir)
                except Exception as e:
                    # Log the request structure for debugging
                    logger.error(f"Failed to create batch job. Request structure: {json.dumps(request_obj, indent=2)[:2000]}")
                    logger.error(f"Generation config: {json.dumps(gen_config, indent=2)}")
                    raise
            
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
            max_polls = 1200
            
            while current_state not in completed_states:
                time.sleep(30)
                poll_count += 1
                batch_job = self.client.batches.get(name=batch_job.name)
                current_state = get_state_name(batch_job)
                logger.info(f"Batch status: {current_state} (poll {poll_count}/{max_polls})")
                
                if poll_count >= max_polls:
                    raise TimeoutError(f"Batch job timed out after {max_polls} polls")
            
            if current_state != 'JOB_STATE_SUCCEEDED':
                # Try to get error details from the batch job
                error_msg = f"Batch job failed with state: {current_state}"
                if hasattr(batch_job, 'error') and batch_job.error:
                    error_details = str(batch_job.error)
                    error_msg += f". Error details: {error_details}"
                raise RuntimeError(error_msg)
            
            # Download results
            results_file = temp_dir / "batch_results.jsonl"
            
            if hasattr(batch_job, 'dest') and batch_job.dest and hasattr(batch_job.dest, 'file_name'):
                result_file_name = batch_job.dest.file_name
                file_content_bytes = self.client.files.download(file=result_file_name)
                
                with open(results_file, 'wb') as f:
                    if isinstance(file_content_bytes, bytes):
                        f.write(file_content_bytes)
                    else:
                        for chunk in file_content_bytes:
                            f.write(chunk)
            else:
                raise RuntimeError("No result file found in batch job")
            
            # Parse result
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    result_obj = json.loads(line)
                    
                    if result_obj.get("status") == "SUCCEEDED" or "response" in result_obj:
                        response_obj = result_obj.get("response", {})
                        
                        # Extract response text
                        response_text = ""
                        raw_thoughts = ""
                        
                        if isinstance(response_obj, dict) and "candidates" in response_obj and response_obj["candidates"]:
                            candidate = response_obj["candidates"][0]
                            if isinstance(candidate, dict) and "content" in candidate:
                                content = candidate["content"]
                                if isinstance(content, dict) and "parts" in content and content["parts"]:
                                    all_parts_text = []
                                    for part in content["parts"]:
                                        if isinstance(part, dict) and "text" in part:
                                            all_parts_text.append(part["text"])
                                            raw_thoughts += part["text"]
                                    
                                    if all_parts_text:
                                        response_text = all_parts_text[-1] if len(all_parts_text) > 1 else all_parts_text[0]
                        
                        # Extract token usage
                        usage_metadata = result_obj.get("usage_metadata") or response_obj.get("usageMetadata", {})
                        token_usage = {
                            "prompt_tokens": int(usage_metadata.get("promptTokenCount") or usage_metadata.get("prompt_token_count", 0) or 0),
                            "response_tokens": int(usage_metadata.get("candidatesTokenCount") or usage_metadata.get("candidates_token_count", 0) or 0),
                            "thoughts_tokens": int(usage_metadata.get("thoughtsTokenCount") or usage_metadata.get("thoughts_token_count", 0) or 0),
                            "total_tokens": int(usage_metadata.get("totalTokenCount") or usage_metadata.get("total_token_count", 0) or 0),
                            "cached_tokens": int(usage_metadata.get("cachedContentTokenCount") or usage_metadata.get("cached_content_token_count", 0) or 0),
                            "non_cached_input": 0
                        }
                        
                        # Log thoughts (with truncation to prevent infinite loops)
                        if raw_thoughts and log_filepath:
                            try:
                                # Use truncation method if available, otherwise write directly
                                if hasattr(self, '_truncate_thoughts'):
                                    truncated_thoughts = self._truncate_thoughts(raw_thoughts, batch_key)
                                else:
                                    # Fallback: simple truncation if method doesn't exist
                                    MAX_SIZE = 500 * 1024  # 500KB
                                    if len(raw_thoughts.encode('utf-8')) > MAX_SIZE:
                                        truncated_thoughts = raw_thoughts.encode('utf-8')[:MAX_SIZE].decode('utf-8', errors='ignore')
                                        truncated_thoughts += f"\n\n[THOUGHTS TRUNCATED - Original size: {len(raw_thoughts)} bytes]\n"
                                    else:
                                        truncated_thoughts = raw_thoughts
                                with open(log_filepath, 'w', encoding='utf-8') as f:
                                    f.write(truncated_thoughts)
                            except Exception as e:
                                logger.warning(f"Failed to write thoughts log: {e}")
                        
                        # If response_mime_type was JSON but no schema was used, try to parse JSON from text
                        if gen_config.get("response_mime_type") == "application/json" and response_text:
                            # The response should be JSON, but it might be in the text field
                            # Try to extract and parse it
                            try:
                                # Response might already be JSON if response_mime_type worked
                                parsed = json.loads(response_text)
                                # If it parsed successfully, return the text (caller will parse if needed)
                                return response_text, token_usage
                            except json.JSONDecodeError:
                                # Try to extract JSON from the text using clean_json_string
                                cleaned = clean_json_string(response_text)
                                if cleaned:
                                    try:
                                        json.loads(cleaned)  # Validate it's valid JSON
                                        return cleaned, token_usage
                                    except json.JSONDecodeError:
                                        pass
                                # If JSON parsing fails, return the raw text and let caller handle it
                                logger.warning(f"Could not parse JSON from batch response, returning raw text")
                                return response_text, token_usage
                        
                        return response_text, token_usage
                    else:
                        error_info = result_obj.get("error", {})
                        error_message = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                        error_status = result_obj.get("status", "UNKNOWN")
                        logger.error(f"Batch request failed with status {error_status}: {error_message}")
                        logger.error(f"Full result object: {json.dumps(result_obj, indent=2)[:1000]}")
                        raise RuntimeError(f"Batch request failed: {error_message}")
            
            raise RuntimeError("No result found in batch output")
            
        finally:
            # Cleanup
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

    def prompt_step2a_merge(self, diplomatic_results: Dict[str, Any], group_lines_by_image: Dict[str, List[Dict[str, Any]]] = None) -> List[Any]:
        """
        Build prompt for merging diplomatic transcriptions (Step 2a).

        Creates a prompt to stitch together overlapping text fragments from
        multiple images and extract surnames and place names.

        Args:
            diplomatic_results: Dictionary mapping image names to their Step 1 JSON results.
            group_lines_by_image: Optional dictionary mapping image names to HTR line data with bounding boxes.

        Returns:
            List of prompt parts (text only).
        """
        return build_step2a_prompt(diplomatic_results, group_lines_by_image)

    def prompt_step2b_expansion(self, merged_diplomatic_text: str) -> List[Any]:
        """
        Build prompt for expanding abbreviations (Step 2b).

        Creates a prompt to expand medieval abbreviations in diplomatic text
        to standard legal Latin.

        Args:
            merged_diplomatic_text: The merged diplomatic text from Step 2a.

        Returns:
            List of prompt parts (text only).
        """
        return build_step2b_prompt(merged_diplomatic_text)

    def prompt_step4_indexing(
        self,
        english_text: str,
        latin_text: str = None,
        date_info: Dict[str, Any] = None,
        county_info: Dict[str, Any] = None,
    ) -> List[Any]:
        """
        Build prompt for structured entity extraction (Step 4).

        Creates a prompt to extract structured case data including parties,
        events, locations, dates, and legal details.

        Args:
            english_text: The English translation text.
            latin_text: Optional diplomatic Latin text for reference.
            date_info: Optional dictionary with date information.
            county_info: Optional dictionary with county information.

        Returns:
            List of prompt parts (text only).
        """
        return build_step4_prompt(english_text, latin_text, date_info, county_info)

    def validate_extraction(
        self, extracted_data: Dict, english_text: str, latin_text: str = None
    ) -> Dict[str, Any]:
        """
        Validate extracted case data for common errors and inconsistencies.

        Performs post-processing checks including:
        - Missing primary parties (plaintiff/defendant)
        - Missing attorney extraction when mentioned in text
        - Names in text not found in extraction
        - Writ type mismatches
        - County source validation

        Args:
            extracted_data: The structured extraction result from Step 4.
            english_text: The English translation text for cross-reference.
            latin_text: Optional Latin text for additional validation.

        Returns:
            A dictionary containing:
            - "validation_issues": List of issue dictionaries with severity and message
            - "issue_count": Total number of issues found
            - "critical_count": Number of critical severity issues
        """
        issues = []

        # Check for mandatory TblPleadings and TblPostea in Cases array
        cases = extracted_data.get('Cases', extracted_data.get('cases', []))
        for case in cases:
            # Check TblPleadings is present and not empty
            pleadings = case.get('TblPleadings', [])
            if not pleadings or len(pleadings) == 0:
                issues.append({
                    "severity": "critical",
                    "type": "missing_pleadings",
                    "message": "TblPleadings is MANDATORY and must contain at least one entry"
                })
            else:
                # Validate each pleading is one sentence
                for idx, pleading in enumerate(pleadings):
                    pleading_text = pleading.get('PleadingText', '')
                    if not pleading_text:
                        issues.append({
                            "severity": "critical",
                            "type": "empty_pleading_text",
                            "message": f"Pleading entry {idx + 1} has empty PleadingText"
                        })
                    # Check if it's roughly one sentence (simple heuristic: should have one period/exclamation/question mark)
                    sentence_endings = pleading_text.count('.') + pleading_text.count('!') + pleading_text.count('?')
                    if sentence_endings == 0 and len(pleading_text) > 50:
                        issues.append({
                            "severity": "medium",
                            "type": "pleading_not_one_sentence",
                            "message": f"Pleading entry {idx + 1} may not be a single sentence (no sentence-ending punctuation found)"
                        })
            
            # Check TblPostea is present and not empty
            postea = case.get('TblPostea', [])
            if not postea or len(postea) == 0:
                issues.append({
                    "severity": "critical",
                    "type": "missing_postea",
                    "message": "TblPostea is MANDATORY and must contain at least one entry"
                })
            else:
                # Validate each postea entry
                for idx, postea_entry in enumerate(postea):
                    postea_text = postea_entry.get('PosteaText', '')
                    if not postea_text:
                        issues.append({
                            "severity": "critical",
                            "type": "empty_postea_text",
                            "message": f"Postea entry {idx + 1} has empty PosteaText"
                        })
                    # Check if it's roughly one sentence
                    sentence_endings = postea_text.count('.') + postea_text.count('!') + postea_text.count('?')
                    if sentence_endings == 0 and len(postea_text) > 50:
                        issues.append({
                            "severity": "medium",
                            "type": "postea_not_one_sentence",
                            "message": f"Postea entry {idx + 1} may not be a single sentence (no sentence-ending punctuation found)"
                        })

        for case in cases:
            parties = case.get('parties', [])
            
            # 1. Check for missing primary parties
            roles_present = set()
            for p in parties:
                roles_present.update(p.get('roles', []))
            
            if 'plaintiff' not in roles_present:
                issues.append({
                    "severity": "critical",
                    "type": "missing_plaintiff",
                    "message": "No plaintiff identified in party list"
                })
            
            if 'defendant' not in roles_present:
                issues.append({
                    "severity": "critical", 
                    "type": "missing_defendant",
                    "message": "No defendant identified in party list"
                })
            
            # 2. Check for attorney extraction
            attorney_keywords = ['attornatum', 'attorney', 'per', 'attorn']
            text_lower = english_text.lower()
            has_attorney_mention = any(kw in text_lower for kw in attorney_keywords)
            has_attorney_party = any('attorney' in r for p in parties for r in p.get('roles', []))
            
            if has_attorney_mention and not has_attorney_party:
                issues.append({
                    "severity": "high",
                    "type": "missing_attorney",
                    "message": "Text mentions attorney but none extracted"
                })
            
            # 3. Cross-reference names in text vs extracted
            # Extract all capitalized name-like patterns from text
            potential_names = set(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', english_text))
            extracted_names = set()
            for p in parties:
                name = p.get('name', {})
                if name.get('originalString'):
                    extracted_names.add(name['originalString'])
                fn = name.get('firstName', '')
                ln = name.get('lastName', '')
                if fn and ln:
                    extracted_names.add(f"{fn} {ln}")
            
            # Find names in text not in extraction (simple heuristic)
            for pn in potential_names:
                if not any(self._fuzzy_match(pn, en) for en in extracted_names):
                    # Check if it's actually a person name (not a place)
                    place_indicators = ['parish', 'ward', 'county', 'street', 'london', 'westminster']
                    if not any(pi in english_text.lower()[max(0, english_text.find(pn)-50):english_text.find(pn)+50].lower() 
                            for pi in place_indicators):
                        issues.append({
                            "severity": "medium",
                            "type": "potentially_missed_name",
                            "message": f"Name '{pn}' appears in text but may not be in extraction"
                        })
            
            # 4. Validate writ type
            writ_type = case.get('caseDetails', {}).get('writ', {}).get('type', '')
            if 'goods' in english_text.lower() or 'chattels' in english_text.lower():
                if writ_type == 'debt':
                    issues.append({
                        "severity": "medium",
                        "type": "writ_type_review",
                        "message": "Writ classified as Debt but text mentions goods/chattels - verify not Detinue"
                    })
            
            # 5. County validation
            county = case.get('caseIdentifier', {}).get('county', '')
            county_source = case.get('caseIdentifier', {}).get('countySource', '')
            if county_source != 'marginal_annotation':
                issues.append({
                    "severity": "high",
                    "type": "county_source",
                    "message": f"County '{county}' not from marginal annotation - verify accuracy"
                })
        
        return {
            "validation_issues": issues,
            "issue_count": len(issues),
            "critical_count": len([i for i in issues if i['severity'] == 'critical'])
        }

    def _fuzzy_match(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """
        Perform simple fuzzy string matching for names.

        Uses SequenceMatcher to compute similarity ratio between two names.

        Args:
            name1: First name to compare.
            name2: Second name to compare.
            threshold: Minimum similarity ratio (0.0-1.0) to consider a match. Defaults to 0.8.

        Returns:
            True if the similarity ratio meets or exceeds the threshold, False otherwise.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio() >= threshold

    def run_batch_job(self, batch_requests: List[Dict], batch_id: str) -> Dict[str, Any]:
        """
        Submit Step 1 diplomatic transcription requests as a batch job.

        Creates a batch job for processing multiple images in parallel. Supports
        resuming interrupted batches by saving state files. Each group gets a unique
        state file to prevent collisions when processing multiple cases.

        Args:
            batch_requests: List of request dictionaries, each containing:
                - "key": Unique identifier (format: "{group_id}::{image_name}")
                - "parts": List of prompt parts (text and/or image references)
            batch_id: Unique identifier for this batch (typically the case group ID).

        Returns:
            Dictionary mapping request keys to their results. Each result contains:
            - "data": Parsed JSON response data
            - "raw_thoughts": Full raw response text including thoughts
            Returns empty dict if batch creation fails or no requests provided.
        """
        if not batch_requests:
            return {}


        # Use consistent cache format with other batch calls
        cached_job_name = self._get_cached_batch_id(batch_id, WORK_DIR)
        batch_job = None

        # 1. ATTEMPT RESUME (Specific to this group)
        if cached_job_name:
            try:
                batch_job = self.client.batches.get(name=cached_job_name)
                logger.info(f"[{batch_id}] Found cached batch ID. Resuming Job: {cached_job_name}")
            except Exception as e:
                logger.warning(f"[{batch_id}] Failed to resume cached batch (creating new): {e}")
                batch_job = None

        # 2. CREATE NEW
        if not batch_job:
            timestamp = int(time.time())
            # Use safe_id in the jsonl filename too
            safe_id = re.sub(r'[^a-zA-Z0-9]', '_', batch_id)
            jsonl_filename = os.path.join(WORK_DIR, f"batch_{safe_id}_{timestamp}.jsonl")
            
            logger.info(f"[{batch_id}] Creating Batch JSONL with {len(batch_requests)} requests...")
            
            # Get cached content name once for all batch requests
            cached_content_name = self._get_or_create_cached_content_step1()
            if cached_content_name:
                logger.info(f"Using cached content for batch job: {cached_content_name}")
            
            with open(jsonl_filename, "w", encoding="utf-8") as f:
                for item in batch_requests:
                    raw_parts = []
                    
                    # If using cached content, extract only variable parts (HTR JSON + image)
                    if cached_content_name:
                        for p in item['parts']:
                            if hasattr(p, 'text') and p.text:
                                # Extract only the HTR section (variable part)
                                if "## Input HTR:" in p.text:
                                    htr_section = p.text.split("## Input HTR:")[-1].strip()
                                    raw_parts.append({"text": f"## Input HTR:\n\n{htr_section}"})
                            elif hasattr(p, 'file_data') and p.file_data:
                                # Keep image parts
                                if hasattr(p.file_data, 'file_uri'):
                                    raw_parts.append({
                                        "file_data": {
                                            "file_uri": p.file_data.file_uri, 
                                            "mime_type": p.file_data.mime_type
                                        }
                                    })
                    else:
                        # No cache - include all parts
                        for p in item['parts']:
                            if hasattr(p, 'model_dump'):
                                raw_parts.append(p.model_dump(exclude_none=True))
                            elif hasattr(p, 'dict'):
                                raw_parts.append(p.dict(exclude_none=True))
                            else:
                                if p.text:
                                    raw_parts.append({"text": p.text})
                                elif p.file_data:
                                    raw_parts.append({
                                        "file_data": {
                                            "file_uri": p.file_data.file_uri, 
                                            "mime_type": p.file_data.mime_type
                                        }
                                    })
                    
                    t_config = types.ThinkingConfig(include_thoughts=True, thinking_level="LOW")
                    
                    # Build generation config
                    gen_config = {
                        "response_mime_type": "application/json",
                        "response_schema": get_diplomatic_schema(),
                        "temperature": 0.0,
                        "thinkingConfig": {"includeThoughts": True, "thinkingLevel": t_config.thinking_level}
                    }
                    
                    # Build request object - cached_content goes at the request level, not in generation_config
                    req_obj = {
                        "contents": [{"parts": raw_parts}],
                        "generation_config": gen_config
                    }
                    
                    # Add cached_content at the request level (not in generation_config)
                    if cached_content_name:
                        req_obj["cached_content"] = cached_content_name
                    
                    line = {
                        "key": item['key'],
                        "request": req_obj
                    }
                    f.write(json.dumps(line) + "\n")

            logger.info(f"[{batch_id}] Uploading JSONL...")
            batch_input_file = self.client.files.upload(
                file=jsonl_filename, 
                config=types.UploadFileConfig(mime_type='application/jsonl')
            )
            
            logger.info(f"[{batch_id}] Submitting Batch Job...")
            batch_job = self.client.batches.create(
                model=MODEL_TEXT,  # Use Gemini 3 Flash Preview for Step 1 batch jobs (if batch mode is used)
                src=batch_input_file.name,
            )
            logger.info(f"[{batch_id}] Job Created: {batch_job.name}")

            # Save batch ID to cache (using consistent format)
            self._save_batch_id(batch_id, batch_job.name, WORK_DIR)

        # 3. POLL STATUS
        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
        while batch_job.state.name not in completed_states:
            logger.info(f"[{batch_id}] Status: {batch_job.state.name}. Waiting 60s...")
            time.sleep(60)
            batch_job = self.client.batches.get(name=batch_job.name)

        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            logger.error(f"[{batch_id}] Batch Job Failed: {batch_job.state.name}")
            return {}

        # 4. RETRIEVE RESULTS
        logger.info(f"[{batch_id}] Success. Downloading...")
        results_map = {}
        total_cached_tokens = 0
        total_prompt_tokens = 0
        total_response_tokens = 0
        total_thoughts_tokens = 0
        total_tokens = 0
        
        if batch_job.dest and batch_job.dest.file_name:
            content = self.client.files.download(file=batch_job.dest.file_name).decode('utf-8')
            for line in content.splitlines():
                if not line.strip(): continue
                try:
                    res = json.loads(line)
                    key = res.get('key')
                    
                    # Log usage metadata if available (check multiple possible locations)
                    um = None
                    if 'response' in res:
                        # Check if usageMetadata (camelCase) is in response
                        if 'usageMetadata' in res['response']:
                            um = res['response']['usageMetadata']
                        # Check if usage_metadata (snake_case) is in response
                        elif 'usage_metadata' in res['response']:
                            um = res['response']['usage_metadata']
                        # Also check if it's at the top level (camelCase)
                        elif 'usageMetadata' in res:
                            um = res['usageMetadata']
                        # Also check if it's at the top level (snake_case)
                        elif 'usage_metadata' in res:
                            um = res['usage_metadata']
                    # Also check top level of res (camelCase first, then snake_case)
                    elif 'usageMetadata' in res:
                        um = res['usageMetadata']
                    elif 'usage_metadata' in res:
                        um = res['usage_metadata']
                    
                    # Store token usage for this item
                    item_token_usage = {
                        "prompt_tokens": 0,
                        "cached_tokens": 0,
                        "response_tokens": 0,
                        "thoughts_tokens": 0,
                        "total_tokens": 0,
                        "non_cached_input": 0
                    }
                    
                    if um:
                        # Helper function to safely extract int values, handling None
                        def safe_int(value):
                            """Convert value to int, defaulting to 0 if None or invalid."""
                            if value is None:
                                return 0
                            try:
                                return int(value)
                            except (ValueError, TypeError):
                                return 0
                        
                        # Handle both camelCase and snake_case field names
                        cached_tokens = safe_int(um.get('cached_content_token_count') or um.get('cachedContentTokenCount', 0))
                        prompt_tokens = safe_int(um.get('prompt_token_count') or um.get('promptTokenCount', 0))
                        response_tokens = safe_int(um.get('candidates_token_count') or um.get('candidatesTokenCount', 0))
                        total = safe_int(um.get('total_token_count') or um.get('totalTokenCount', 0))
                        thoughts_tokens = safe_int(um.get('thoughts_token_count') or um.get('thoughtsTokenCount', 0))
                        
                        # Calculate non-cached input tokens (what you're actually billed for as new input)
                        non_cached_input = prompt_tokens - cached_tokens
                        
                        item_token_usage = {
                            "prompt_tokens": prompt_tokens,
                            "cached_tokens": cached_tokens,
                            "response_tokens": response_tokens,
                            "thoughts_tokens": thoughts_tokens,
                            "total_tokens": total,
                            "non_cached_input": non_cached_input
                        }
                        
                        total_cached_tokens += cached_tokens
                        total_prompt_tokens += prompt_tokens
                        total_response_tokens += response_tokens
                        total_thoughts_tokens += thoughts_tokens
                        total_tokens += total
                        
                        if cached_tokens > 0:
                            savings_pct = (cached_tokens / prompt_tokens) * 100 if prompt_tokens > 0 else 0
                            logger.info(
                                f"[{batch_id}] {key}: {cached_tokens:,} cached (billed at cache rate), "
                                f"{non_cached_input:,} new input (billed at input rate), "
                                f"{response_tokens:,} output, {thoughts_tokens:,} thoughts, "
                                f"{total:,} total ({savings_pct:.1f}% from cache)"
                            )
                    else:
                        # Log the structure for debugging if cache was used but no metadata found
                        if self._cached_content_step1 and 'response' in res:
                            logger.debug(f"[{batch_id}] {key}: No usage_metadata found in response. Response keys: {list(res.get('response', {}).keys())}")
                    
                    if 'response' in res:
                        try:
                            cand = res['response'].get('candidates', [{}])[0]
                            parts = cand.get('content', {}).get('parts', [])
                            full_raw_text = "".join([p.get('text', '') for p in parts])
                            cleaned_json_str = clean_json_string(full_raw_text)
                            if cleaned_json_str:
                                results_map[key] = {
                                    "data": json.loads(cleaned_json_str),
                                    "raw_thoughts": full_raw_text,
                                    "token_usage": item_token_usage
                                }
                        except Exception: pass
                except json.JSONDecodeError: pass
        
        # 5. LOG CACHE USAGE SUMMARY
        batch_cache_name = self._cached_content_step1  # Get cache name if it exists
        if batch_cache_name and total_cached_tokens > 0:
            savings_pct = (total_cached_tokens / (total_prompt_tokens + total_cached_tokens)) * 100 if (total_prompt_tokens + total_cached_tokens) > 0 else 0
            logger.info(
                f"[{batch_id}] Cache usage summary: {total_cached_tokens:,} cached tokens, "
                f"{total_prompt_tokens:,} prompt tokens, {total_tokens:,} total tokens "
                f"({savings_pct:.1f}% from cache)"
            )
        elif batch_cache_name and total_tokens > 0:
            logger.warning(f"[{batch_id}] Cached content was used but no cached tokens found in results")
        
        # Store aggregate token usage in results_map
        results_map["_batch_token_usage"] = {
            "prompt_tokens": total_prompt_tokens,
            "cached_tokens": total_cached_tokens,
            "response_tokens": total_response_tokens,
            "thoughts_tokens": total_thoughts_tokens,
            "total_tokens": total_tokens,
            "non_cached_input": total_prompt_tokens - total_cached_tokens
        }
        
        # 6. CLEANUP (remove cache file after successful completion)
        if results_map:
            cache_path = self._get_batch_cache_path(batch_id, WORK_DIR)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    logger.debug(f"[{batch_id}] Removed batch cache file after successful completion")
                except Exception as e:
                    logger.warning(f"[{batch_id}] Failed to remove batch cache: {e}")
        
        return results_map

    def _regenerate_master_record_from_final_index(self, gid: str, out_dir: str) -> bool:
        """
        Regenerate a minimal master_record.json from final_index.json if it exists.
        
        Args:
            gid: Group ID
            out_dir: Output directory for the group
            
        Returns:
            True if master_record.json was successfully regenerated, False otherwise
        """
        final_index_path = os.path.join(out_dir, "final_index.json")
        master_record_path = os.path.join(out_dir, "master_record.json")
        
        if not os.path.exists(final_index_path):
            logger.debug(f"[{gid}] final_index.json not found. Cannot regenerate master_record.json.")
            return False
        
        try:
            # Load final_index.json
            with open(final_index_path, "r", encoding="utf-8") as f:
                final_data = json.load(f)
            
            # Extract roll_number and rotulus_number from group_id
            roll_match = re.search(r'(?:CP\s*40|Roll|no)[-._\s]+(\d+)', gid, re.IGNORECASE)
            rot_match = re.search(r'(?:f|rot|m)[-._\s]*(\d+[a-z]?)', gid, re.IGNORECASE) 
            if not rot_match: 
                rot_match = re.search(r'[-._\s](\d+[a-z]?)$', gid)
            
            roll_num = roll_match.group(1) if roll_match else None
            rot_num = rot_match.group(1) if rot_match else None
            
            # Remove trailing letter from rotulus number if present (e.g., "305d" -> "305")
            if rot_num:
                rot_num = re.sub(r'[a-z]+$', '', rot_num, flags=re.IGNORECASE)
            
            # Extract metadata from final_index.json
            # Handle different possible structures
            ref_data = None
            for key in ["TblReference", "case_metadata", "archival_metadata"]:
                candidate = final_data.get(key)
                if candidate and isinstance(candidate, dict) and len(candidate) > 0:
                    ref_data = candidate
                    break
            
            if not ref_data:
                ref_data = {}
            
            county = ref_data.get("County") or ref_data.get("county", "UNKNOWN")
            county_source = ref_data.get("CountySource") or ref_data.get("county_source", "not_extracted")
            county_original_latin = ref_data.get("CountyOriginalLatin") or ref_data.get("county_original_latin", "")
            term = ref_data.get("Term") or ref_data.get("term", "")
            calendar_year = ref_data.get("CalendarYear") or ref_data.get("dateyear") or ref_data.get("calendar_year")
            regnal_year = ref_data.get("RegnalYear") or ref_data.get("regnal_year", "")
            
            # Get date context from roll number
            date_info = get_cp40_info(roll_num) if roll_num else None
            if not date_info and calendar_year and term:
                # Fallback: construct date_info from final_index.json if available
                date_info = {
                    "Roll": int(roll_num) if roll_num else None,
                    "Calendar Year": calendar_year,
                    "Term": term,
                    "Regnal Year": regnal_year or ""
                }
            
            # Construct minimal master_record.json
            minimal_master_record = {
                "case_metadata": {
                    "group_id": gid,
                    "roll_number": roll_num or "unknown",
                    "rotulus_number": rot_num or "unknown",
                    "county": county,
                    "county_source": county_source,
                    "county_confidence": "high" if county and county != "UNKNOWN" else "none",
                    "county_original_latin": county_original_latin,
                    "processed_at": time.ctime(),
                    "date_context": date_info or {}
                },
                "ground_truth_from_db": [None],
                "source_material": [],
                "text_content": {
                    "latin_reconstructed": "",
                    "english_translation": ""
                },
                "extracted_entities": {
                    "surnames": [],
                    "place_names": [],
                    "marginal_county": {"original": county_original_latin, "anglicized": county}
                },
                "legal_index": final_data,
                "validation": {},
                "token_usage": {},
                "estimated_cost": {}
            }
            
            # Save the regenerated master_record.json
            with open(master_record_path, "w", encoding="utf-8") as f:
                json.dump(minimal_master_record, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{gid}] Regenerated master_record.json from final_index.json")
            return True
            
        except Exception as e:
            logger.error(f"[{gid}] Failed to regenerate master_record.json from final_index.json: {e}", exc_info=True)
            return False

    def _run_subprocess_robust(self, cmd: List[str], cwd: Optional[str] = None, timeout: int = 120, 
                                description: str = "subprocess", env: Optional[Dict[str, str]] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Run a subprocess with robust timeout and cleanup handling.
        
        Uses Popen with proper process group management to ensure child processes
        are killed on timeout. This prevents blocking of other workers.
        
        Args:
            cmd: Command and arguments as a list
            cwd: Working directory (optional)
            timeout: Timeout in seconds
            description: Description for logging
            
        Returns:
            Tuple of (success: bool, stdout: Optional[str], stderr: Optional[str])
        """
        process = None
        try:
            # Create process with new process group (Unix) or creation flags (Windows)
            creation_flags = 0
            preexec_fn = None
            if platform.system() == 'Windows':
                # On Windows, use CREATE_NEW_PROCESS_GROUP to allow killing child processes
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                # On Unix, use preexec_fn to create new process group
                preexec_fn = os.setpgrp
            
            # Merge provided env with current environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env=process_env,
                creationflags=creation_flags if platform.system() == 'Windows' else 0,
                preexec_fn=os.setpgrp if platform.system() != 'Windows' else None
            )
            
            # Wait for process with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return (process.returncode == 0, stdout, stderr)
            except subprocess.TimeoutExpired:
                # Kill the process group
                logger.warning(f"{description} timed out after {timeout} seconds. Killing process group...")
                try:
                    if platform.system() == 'Windows':
                        # On Windows, try to kill the process and its children
                        # Use taskkill to kill the process tree
                        try:
                            subprocess.run(
                                ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                capture_output=True,
                                timeout=5
                            )
                        except:
                            # Fallback to direct kill
                            process.kill()
                    else:
                        # On Unix, kill the entire process group
                        try:
                            pgid = os.getpgid(process.pid)
                            os.killpg(pgid, signal.SIGTERM)
                            # Wait a bit for graceful termination
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                # Force kill if still running
                                os.killpg(pgid, signal.SIGKILL)
                                process.wait()
                        except (ProcessLookupError, OSError):
                            # Process may have already terminated, try direct kill
                            try:
                                process.kill()
                                process.wait(timeout=2)
                            except:
                                pass
                except (ProcessLookupError, OSError, subprocess.TimeoutExpired) as e:
                    # Process may have already terminated
                    logger.debug(f"Error killing process: {e}")
                    try:
                        process.kill()
                    except:
                        pass
                
                return (False, None, f"Process timed out after {timeout} seconds")
                
        except FileNotFoundError:
            return (False, None, f"Command not found: {cmd[0]}")
        except Exception as e:
            logger.error(f"Error running {description}: {e}", exc_info=True)
            return (False, None, str(e))
        finally:
            # Ensure process is cleaned up
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                    except:
                        pass

    def _ensure_pdf_report(self, gid: str, out_dir: str) -> bool:
        """
        Check if comparison report PDF exists for a group, and generate it if missing.
        
        Args:
            gid: Group ID
            out_dir: Output directory for the group
            
        Returns:
            True if PDF exists or was successfully generated, False otherwise
        """
        master_record_path = os.path.join(out_dir, "master_record.json")
        
        # Check if master_record.json exists
        if not os.path.exists(master_record_path):
            logger.debug(f"[{gid}] master_record.json not found. Cannot generate PDF.")
            # Try to regenerate from final_index.json if it exists
            final_index_path = os.path.join(out_dir, "final_index.json")
            if os.path.exists(final_index_path):
                logger.info(f"[{gid}] master_record.json not found but final_index.json exists. Regenerating master_record.json...")
                if self._regenerate_master_record_from_final_index(gid, out_dir):
                    logger.info(f"[{gid}] Successfully regenerated master_record.json from final_index.json")
                else:
                    logger.warning(f"[{gid}] Failed to regenerate master_record.json. Cannot generate PDF.")
                    return False
            else:
                logger.debug(f"[{gid}] final_index.json also not found. Cannot generate PDF.")
                return False
        
        # Determine expected PDF filename(s) from master_record.json metadata
        try:
            with open(master_record_path, "r", encoding="utf-8") as f:
                master_data = json.load(f)
            meta = master_data.get("case_metadata", {})
            roll_number = meta.get("roll_number", "unknown")
            rotulus_number = meta.get("rotulus_number", "unknown")
            
            # Check for multiple ground truth cases or AI cases
            ground_truth_data = master_data.get("ground_truth_from_db", [])
            gt_cases = []
            for item in ground_truth_data:
                if item is None:
                    continue
                elif isinstance(item, list):
                    gt_cases.extend([c for c in item if isinstance(c, dict)])
                elif isinstance(item, dict):
                    gt_cases.append(item)
            
            ai_cases = master_data.get("legal_index", {}).get("Cases", [])
            # Filter out empty cases - check for non-empty dicts with actual content
            ai_cases = [
                c for c in ai_cases 
                if c and isinstance(c, dict) and (c.get("TblCase") or c.get("Agents") or len(c) > 0)
            ]
            
            # Filter out empty GT cases too
            gt_cases = [
                c for c in gt_cases 
                if c and isinstance(c, dict) and (c.get("TblCase") or c.get("Agents") or len(c) > 0)
            ]
            
            # Only treat as "multiple cases" if there are ACTUALLY 2+ valid cases
            has_multiple = (len(gt_cases) >= 2) or (len(ai_cases) >= 2)
            
            if has_multiple:
                # Multiple cases - check for match files
                base_pdf_pattern = f"comparison_report_CP40-{roll_number}_{rotulus_number}_match*.pdf"
                import glob
                existing_pdfs = glob.glob(os.path.join(out_dir, base_pdf_pattern))
                if existing_pdfs:
                    logger.debug(f"[{gid}] Comparison report PDFs already exist: {len(existing_pdfs)} file(s)")
                    return True
                # Continue to generate - will create multiple files
                pdf_file_path = None  # Will be determined after generation
            else:
                # Single case - use original logic
                pdf_filename = f"comparison_report_CP40-{roll_number}_{rotulus_number}.pdf"
                pdf_file_path = os.path.join(out_dir, pdf_filename)
                if os.path.exists(pdf_file_path):
                    logger.debug(f"[{gid}] Comparison report PDF already exists: {pdf_file_path}")
                    return True
        except Exception as e:
            logger.warning(f"[{gid}] Could not read master_record.json to determine PDF filename: {e}")
            return False
        
        # PDF is missing, try to generate it
        logger.info(f"[{gid}] Comparison report PDF not found. Attempting to generate: {pdf_file_path}")
        try:
            # Import report generator directly (will execute in current process, so logging is visible)
            try:
                from report_generator.report import generate_latex_report
            except ImportError as e:
                logger.error(f"[{gid}] Failed to import report generator: {e}")
                return False
            
            # Note: SKIP_REPORT_IMAGE_PROCESSING is no longer set by default
            # Images will be extracted for the report. Set SKIP_REPORT_IMAGE_PROCESSING="true"
            # in your environment if you want to skip image processing for faster generation.
            original_skip_flag = os.environ.get("SKIP_REPORT_IMAGE_PROCESSING")
            # Don't force it to "true" - allow images to be extracted
            # os.environ["SKIP_REPORT_IMAGE_PROCESSING"] = "true"  # Removed to enable image extraction
            
            try:
                # Change to output directory to match subprocess behavior
                original_cwd = os.getcwd()
                os.chdir(out_dir)
                
                # Step 1: Generate the .tex file (call directly, not via subprocess)
                logger.info(f"[{gid}] Generating LaTeX report (direct call, logging will be visible)")
                
                # Read master_record.json (we're already in out_dir, so relative path works)
                master_record_path = "master_record.json"
                with open(master_record_path, "r", encoding="utf-8") as f:
                    master_data = json.load(f)
                
                # Generate LaTeX report directly
                generate_latex_report(master_data, filename=None, api_key=None)
                
            except Exception as e:
                error_msg = f"Report generator failed: {e}"
                logger.error(f"[{gid}] {error_msg}", exc_info=True)
                import traceback
                logger.error(f"[{gid}] Traceback: {traceback.format_exc()}")
                return False
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                # Restore original environment variable
                if original_skip_flag is None:
                    os.environ.pop("SKIP_REPORT_IMAGE_PROCESSING", None)
                else:
                    os.environ["SKIP_REPORT_IMAGE_PROCESSING"] = original_skip_flag

            # Step 2: Compile the .tex file(s) into .pdf using xelatex
            # Check for multiple tex files (match files) or single file
            import glob
            base_tex_pattern = f"comparison_report_CP40-{roll_number}_{rotulus_number}_match*.tex"
            tex_files = glob.glob(os.path.join(out_dir, base_tex_pattern))
            
            # Also check for single file
            single_tex_file = os.path.join(out_dir, f"comparison_report_CP40-{roll_number}_{rotulus_number}.tex")
            if os.path.exists(single_tex_file) and single_tex_file not in tex_files:
                tex_files.append(single_tex_file)
            
            if not tex_files:
                logger.warning(f"[{gid}] No .tex files found after report generation")
                return False
            
            # Pre-check: Verify xelatex is available (only once)
            try:
                check_result = subprocess.run(
                    ["xelatex", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if check_result.returncode != 0:
                    logger.error(f"[{gid}] xelatex --version failed. xelatex may not be properly installed.")
                    return False
                logger.debug(f"[{gid}] xelatex found: {check_result.stdout.split(chr(10))[0] if check_result.stdout else 'version check passed'}")
            except FileNotFoundError:
                logger.error(f"[{gid}] 'xelatex' command not found. Please install TeX Live or MiKTeX.")
                logger.error(f"[{gid}] On Ubuntu/Debian: sudo apt-get install texlive-xetex")
                return False
            except Exception as e:
                logger.warning(f"[{gid}] Could not verify xelatex installation: {e}")
            
            # Compile each tex file
            success_count = 0
            for tex_file_path in tex_files:
                tex_filename = os.path.basename(tex_file_path)
                logger.info(f"[{gid}] Compiling {tex_filename} to PDF...")
                
                cmd_xelatex_args = [
                    "xelatex", 
                    "-interaction=nonstopmode", 
                    "-output-directory", out_dir, 
                    tex_file_path
                ]

                try:
                    # Pass 1
                    logger.info(f"[{gid}] Running: Compile PDF (Pass 1) for {tex_filename}")
                    success1, stdout1, stderr1 = self._run_subprocess_robust(
                        cmd_xelatex_args,
                        cwd=out_dir,
                        timeout=120,
                        description=f"[{gid}] Compile PDF (Pass 1) for {tex_filename}"
                    )
                    
                    # Log xelatex output for debugging (even if it "succeeded")
                    if stdout1:
                        # Check for common errors in output
                        if "Error" in stdout1 or "Fatal" in stdout1 or "!" in stdout1:
                            logger.warning(f"[{gid}] xelatex Pass 1 had errors/warnings for {tex_filename}:\n{stdout1[-2000:]}")
                        else:
                            logger.debug(f"[{gid}] xelatex Pass 1 output for {tex_filename} (last 500 chars):\n{stdout1[-500:]}")
                    if stderr1:
                        logger.warning(f"[{gid}] xelatex Pass 1 stderr for {tex_filename}:\n{stderr1}")
                    
                    if not success1:
                        logger.error(f"[{gid}] xelatex Pass 1 failed for {tex_filename}. Stderr: {stderr1}")
                        continue  # Try next file instead of returning False
                    
                    # Pass 2 (Standard practice for LaTeX cross-references)
                    logger.info(f"[{gid}] Running: Compile PDF (Pass 2) for {tex_filename}")
                    success2, stdout2, stderr2 = self._run_subprocess_robust(
                        cmd_xelatex_args,
                        cwd=out_dir,
                        timeout=120,
                        description=f"[{gid}] Compile PDF (Pass 2) for {tex_filename}"
                    )
                    
                    # Log xelatex output for debugging
                    if stdout2:
                        if "Error" in stdout2 or "Fatal" in stdout2 or "!" in stdout2:
                            logger.warning(f"[{gid}] xelatex Pass 2 had errors/warnings for {tex_filename}:\n{stdout2[-2000:]}")
                        else:
                            logger.debug(f"[{gid}] xelatex Pass 2 output for {tex_filename} (last 500 chars):\n{stdout2[-500:]}")
                    if stderr2:
                        logger.warning(f"[{gid}] xelatex Pass 2 stderr for {tex_filename}:\n{stderr2}")
                    
                    if not success2:
                        logger.error(f"[{gid}] xelatex Pass 2 failed for {tex_filename}. Stderr: {stderr2}")
                        continue  # Try next file instead of returning False

                    # Check if PDF was actually created
                    pdf_file_path = os.path.splitext(tex_file_path)[0] + ".pdf"
                    if os.path.exists(pdf_file_path):
                        logger.info(f"[{gid}] Successfully created PDF report: {pdf_file_path}")
                        success_count += 1
                        
                        # Clean up LaTeX auxiliary files
                        base_name = os.path.splitext(tex_filename)[0]
                        aux_extensions = ['.aux', '.log', '.out', '.toc', '.fdb_latexmk', '.fls', '.synctex.gz']
                        cleaned_count = 0
                        for ext in aux_extensions:
                            aux_file = os.path.join(out_dir, base_name + ext)
                            if os.path.exists(aux_file):
                                try:
                                    os.remove(aux_file)
                                    cleaned_count += 1
                                except OSError as e:
                                    logger.warning(f"[{gid}] Could not remove {aux_file}: {e}")
                        
                        if cleaned_count > 0:
                            logger.debug(f"[{gid}] Cleaned up {cleaned_count} LaTeX auxiliary file(s) for {tex_filename}")
                    else:
                        logger.warning(f"[{gid}] PDF file was not created for {tex_filename}")
                except Exception as e:
                    logger.error(f"[{gid}] Error compiling {tex_filename}: {e}", exc_info=True)
                    continue  # Try next file
            
            # Return True if at least one PDF was successfully created
            if success_count > 0:
                logger.info(f"[{gid}] Successfully compiled {success_count}/{len(tex_files)} PDF report(s)")
                return True
            else:
                logger.error(f"[{gid}] Failed to compile any PDF reports")
                return False

        except Exception as e:
            logger.error(f"[{gid}] An exception occurred during PDF report generation: {e}", exc_info=True)
            return False

    def _process_group(self, gid: str, paths: List[str]) -> None:
        """
        Process a single group through the complete workflow pipeline.
        
        This method contains all the group processing logic that was previously
        in the execute method's for loop. It processes one group at a time.
        
        Args:
            gid: Group ID
            paths: List of image file paths for this group
        """
        out_dir = os.path.join(OUTPUT_DIR, gid.replace(" ", "_"))
        os.makedirs(out_dir, exist_ok=True)
        
        # Check skip condition
        final_json_path = os.path.join(out_dir, "final_index.json")
        if os.path.exists(final_json_path) and not self.force and not self.rerun_from_post_pylaia:
            logger.info(f"Skipping Group {gid} (Already Complete).")
            return
        
        if self.rerun_from_post_pylaia:
            logger.info(f"RERUN MODE: Will reprocess from post-correction step onwards")

        logger.info(f"=== PROCESSING GROUP: {gid} ===")
        
        # Track line images that belong to this worker's group for cleanup
        # This ensures each worker only deletes its own line files, not those of other workers
        worker_line_images = set()  # Set of absolute paths to line image files
        
        try:
            # --- PHASE 1: PREP & HTR PROCESSING ---
            group_lines_by_image = {}
            post_correction_results = {}  # Store post-correction results per image
            diplomatic_results = {}  # For backward compatibility with downstream steps

            for p in paths:
                img_name = os.path.basename(p)
                post_corr_json_path = os.path.join(out_dir, f"{img_name}_post_correction.json")

                # 1. Run HTR Tools (Kraken + PyLaia)
                k, h = self.run_htr_tools(p, out_dir)
                lines = self.merge_htr_data(p, k, h)
                group_lines_by_image[img_name] = lines
                
                # Track line images for this image
                basename = os.path.splitext(img_name)[0]
                lines_dir = os.path.join(out_dir, basename, "lines")
                if os.path.exists(lines_dir):
                    # Collect all line image files in this directory
                    for line in lines:
                        line_id = line.get("line_id")
                        if line_id:
                            # Try different extensions
                            for ext in ['.png', '.jpg', '.jpeg']:
                                line_image_path = os.path.join(lines_dir, f"{line_id}{ext}")
                                if os.path.exists(line_image_path):
                                    worker_line_images.add(os.path.abspath(line_image_path))
                                    break
                
                if not lines:
                    logger.warning(f"[{gid}] No HTR lines found for {img_name}. Skipping image.")
                    continue

            # --- PHASE 1.5: POST-CORRECTION AND NAMED ENTITY EXTRACTION (NON-BATCH MODE) ---
            logger.info(f"[{gid}] Running post-correction and named entity extraction in non-batch mode...")
            
            # Collect images that need processing
            should_rerun = self.force or self.rerun_from_post_pylaia
            image_lines_map = {}  # image_path -> {'lines': [...], 'image_name': '...'}
            images_to_process = []
            
            for p in paths:
                img_name = os.path.basename(p)
                post_corr_json_path = os.path.join(out_dir, f"{img_name}_post_correction.json")
                lines = group_lines_by_image.get(img_name, [])
                
                if not lines:
                    continue
                
                # Check if we should load existing results or process new
                if os.path.exists(post_corr_json_path) and not should_rerun:
                    logger.info(f"[{gid}] Loading existing post-correction for {img_name}...")
                    try:
                        with open(post_corr_json_path, 'r', encoding='utf-8') as f:
                            loaded_result = json.load(f)
                            
                            # Validate and normalize the loaded result
                            if isinstance(loaded_result, list):
                                # Old format: result is a list of lines
                                logger.warning(f"[{gid}] Post-correction JSON for {img_name} is in old list format. Converting...")
                                loaded_result = {
                                    "image_name": img_name,
                                    "lines": loaded_result
                                }
                            elif not isinstance(loaded_result, dict):
                                logger.error(f"[{gid}] Invalid post-correction JSON format for {img_name}: {type(loaded_result)}")
                                raise ValueError(f"Invalid format: {type(loaded_result)}")
                            
                            # Ensure it has required keys
                            if "lines" not in loaded_result:
                                logger.error(f"[{gid}] Post-correction JSON for {img_name} missing 'lines' key")
                                raise ValueError("Missing 'lines' key")
                            
                            # Check if we need to fill in missing CTC loss values
                            # This happens when files were created before the model was available
                            try:
                                updated_result, was_updated = fill_missing_ctc_losses(
                                    loaded_result,
                                    lines,
                                    self.name_db,
                                    self.bayesian_config,
                                    out_dir
                                )
                                
                                # If CTC losses were updated, save the file back
                                if was_updated:
                                    logger.info(f"[{gid}] Updated CTC loss values for {img_name}, saving...")
                                    with open(post_corr_json_path, 'w', encoding='utf-8') as f:
                                        json.dump(updated_result, f, indent=2, ensure_ascii=False)
                                
                                post_correction_results[img_name] = updated_result
                            except Exception as e:
                                logger.warning(f"[{gid}] Could not fill missing CTC losses for {img_name}: {e}. Using loaded result as-is.")
                                post_correction_results[img_name] = loaded_result
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"[{gid}] Corrupt or invalid post-correction JSON for {img_name}: {e}. Re-processing.")
                        images_to_process.append(p)
                        image_lines_map[p] = {'lines': lines, 'image_name': img_name}
                else:
                    # Need to process this image
                    if img_name not in post_correction_results:
                        images_to_process.append(p)
                        image_lines_map[p] = {'lines': lines, 'image_name': img_name}
            
            # Initialize token usage for step 1
            token_usage_step1 = None
            
            # Try to load token usage from saved file if all results were loaded from disk
            if not images_to_process:
                token_usage_file = os.path.join(out_dir, "step1_token_usage.json")
                if os.path.exists(token_usage_file):
                    try:
                        with open(token_usage_file, 'r', encoding='utf-8') as f:
                            token_usage_step1 = json.load(f)
                            logger.info(f"[{gid}] Loaded Step 1 token usage from saved file: {token_usage_step1}")
                    except Exception as e:
                        logger.warning(f"[{gid}] Failed to load Step 1 token usage from file: {e}")
            
            # Process images in batch mode if we have any to process
            if images_to_process:
                logger.info(f"[{gid}] Processing {len(images_to_process)} images with Gemini 3 Flash Preview (batch API)...")
                
                # Get Gemini client for post-correction
                gemini_client = get_gemini_flash_client()
                
                # Run batch post-correction
                from .post_correction import run_batch_post_correction
                batch_response = run_batch_post_correction(
                    image_lines_map,
                    batch_id=gid,
                    client=gemini_client,
                    out_dir=out_dir
                )
                
                # Extract results and token usage from response (aggregated across all images in the group)
                batch_results = batch_response.get("results", {}) if isinstance(batch_response, dict) else batch_response
                token_usage_step1 = batch_response.get("token_usage") if isinstance(batch_response, dict) else None
                
                # Save token usage to file for future runs
                if token_usage_step1 is not None:
                    token_usage_file = os.path.join(out_dir, "step1_token_usage.json")
                    try:
                        with open(token_usage_file, 'w', encoding='utf-8') as f:
                            json.dump(token_usage_step1, f, indent=2, ensure_ascii=False)
                        logger.info(f"[{gid}] Saved Step 1 token usage to {token_usage_file}")
                    except Exception as e:
                        logger.warning(f"[{gid}] Failed to save Step 1 token usage to file: {e}")
                
                # Process results and apply Bayesian correction
                if batch_results:
                    processed_results = process_post_correction_results(
                        batch_results,
                        image_lines_map,
                        self.name_db,
                        self.bayesian_config,
                        out_dir
                    )
                    
                    # Store results and save to files
                    for img_name, result in processed_results.items():
                        post_correction_results[img_name] = result
                        post_corr_json_path = os.path.join(out_dir, f"{img_name}_post_correction.json")
                        with open(post_corr_json_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    logger.error(f"[{gid}] Post-correction failed. Using fallback.")
                    # Fallback: use raw HTR text for images that failed
                    for p in images_to_process:
                        img_name = image_lines_map[p]['image_name']
                        lines = image_lines_map[p]['lines']
                        post_correction_results[img_name] = {
                            "image_name": img_name,
                            "lines": [
                                {
                                    "line_id": f"L{idx:02d}",
                                    "original_htr_text": line.get("htr_text", ""),
                                    "corrected_text": line.get("htr_text", ""),
                                    "bbox": line.get("bbox"),
                                    "forenames": [],
                                    "surnames": [],
                                    "placenames": []
                                }
                                for idx, line in enumerate(lines, 1)
                            ]
                        }
            
            # Convert post-correction results to diplomatic format for downstream steps
            for p in paths:
                img_name = os.path.basename(p)
                if img_name in post_correction_results:
                    result = post_correction_results[img_name]
                    
                    # Handle case where result might be a list (from old format) or dict
                    if isinstance(result, list):
                        # Old format: result is directly a list of lines
                        logger.warning(f"[{gid}] Post-correction result for {img_name} is in old list format. Converting...")
                        result = {
                            "image_name": img_name,
                            "lines": result
                        }
                        post_correction_results[img_name] = result  # Update to new format
                    
                    # Ensure result is a dict with "lines" key
                    if not isinstance(result, dict):
                        logger.error(f"[{gid}] Invalid post-correction result format for {img_name}: {type(result)}")
                        continue
                    
                    if "lines" not in result:
                        logger.error(f"[{gid}] Post-correction result for {img_name} missing 'lines' key")
                        continue
                    
                    corrected_lines = get_corrected_lines_for_stitching(result)
                    
                    # Validate corrected_lines is a list of dicts
                    if not isinstance(corrected_lines, list):
                        logger.error(f"[{gid}] get_corrected_lines_for_stitching returned non-list for {img_name}: {type(corrected_lines)}")
                        continue
                    
                    # Validate each line is a dict
                    valid_lines = []
                    for idx, line in enumerate(corrected_lines):
                        if not isinstance(line, dict):
                            logger.error(f"[{gid}] Line {idx} from get_corrected_lines_for_stitching is not a dict for {img_name}: {type(line)}")
                            logger.error(f"  Line value: {line}")
                            continue
                        # Ensure it has 'transcription' key
                        if "transcription" not in line:
                            logger.warning(f"[{gid}] Line {idx} missing 'transcription' key, adding empty string")
                            line["transcription"] = ""
                        valid_lines.append(line)
                    
                    if not valid_lines:
                        logger.error(f"[{gid}] No valid lines after processing {img_name}")
                        continue
                    
                    diplomatic_results[img_name] = {"lines": valid_lines}
                    
                    # Also save as step1.json for compatibility
                    step1_json_path = os.path.join(out_dir, f"{img_name}_step1.json")
                    with open(step1_json_path, 'w', encoding='utf-8') as f:
                        json.dump({"lines": corrected_lines}, f, indent=2, ensure_ascii=False)
            
            # Step 1 token usage is collected from batch post-correction if available
            # If still None, try to load from master_record.json as a fallback
            # Also load token usage for all other steps (2a, 2b, 3, 4) from master_record.json if available
            master_record_path = os.path.join(out_dir, "master_record.json")
            token_usage_from_master = {}
            if os.path.exists(master_record_path):
                try:
                    with open(master_record_path, 'r', encoding='utf-8') as f:
                        master_data = json.load(f)
                        token_usage_from_master = master_data.get("token_usage", {})
                        if token_usage_from_master:
                            logger.info(f"[{gid}] Loaded token usage data from master_record.json for all steps")
                except Exception as e:
                    logger.debug(f"[{gid}] Could not load token usage from master_record.json: {e}")
            
            # Helper function to load token usage from multiple sources
            def load_token_usage_from_sources(step_key: str, step_specific_file: str = None) -> Optional[Dict[str, Any]]:
                """Try to load token usage from multiple sources in order of preference.
                
                Args:
                    step_key: The key used in master_record.json (e.g., "step1_diplomatic_transcription")
                    step_specific_file: Optional path to a step-specific token usage file
                
                Returns:
                    Token usage dict if found, None otherwise
                """
                # Try 1: master_record.json (already loaded)
                if token_usage_from_master:
                    usage = token_usage_from_master.get(step_key)
                    if usage:
                        logger.debug(f"[{gid}] Found {step_key} token usage in master_record.json")
                        return usage
                
                # Try 2: Step-specific token usage file (like step1_token_usage.json)
                if step_specific_file and os.path.exists(step_specific_file):
                    try:
                        with open(step_specific_file, 'r', encoding='utf-8') as f:
                            usage = json.load(f)
                            if usage:
                                logger.info(f"[{gid}] Loaded {step_key} token usage from {step_specific_file}")
                                return usage
                    except Exception as e:
                        logger.debug(f"[{gid}] Could not load token usage from {step_specific_file}: {e}")
                
                # Try 3: Check if there's a token usage file with standard naming pattern
                # (e.g., step2a_token_usage.json, step2b_token_usage.json, etc.)
                step_num = step_key.split('_')[0]  # Extract "step1", "step2a", etc.
                standard_token_file = os.path.join(out_dir, f"{step_num}_token_usage.json")
                if os.path.exists(standard_token_file):
                    try:
                        with open(standard_token_file, 'r', encoding='utf-8') as f:
                            usage = json.load(f)
                            if usage:
                                logger.info(f"[{gid}] Loaded {step_key} token usage from {standard_token_file}")
                                return usage
                    except Exception as e:
                        logger.debug(f"[{gid}] Could not load token usage from {standard_token_file}: {e}")
                
                return None
            
            # Use Step 1 token usage from multiple sources if not already available
            if token_usage_step1 is None:
                step1_token_file = os.path.join(out_dir, "step1_token_usage.json")
                token_usage_step1 = load_token_usage_from_sources("step1_diplomatic_transcription", step1_token_file)
                if token_usage_step1:
                    logger.info(f"[{gid}] Loaded Step 1 token usage from available sources")
            
            if token_usage_step1 is None:
                logger.info(f"[{gid}] Post-correction complete. Step 1 token usage not available from batch job, saved file, or master record.")
            else:
                logger.info(f"[{gid}] Post-correction complete. Step 1 token usage collected: {token_usage_step1}")

            if not diplomatic_results:
                logger.error(f"No diplomatic results available for {gid} (Batch failed or empty).")
                return

            # --- PHASE 3.5: COUNTY EXTRACTION ---
            # County info will be extracted from Step 2a marginal_county field
            county_info = None

            # --- PHASE 4: TEXT PROCESSING CHAIN ---
            # 1. PREPARE DATA FOR SCORING (Fixing Undefined Variables)
            all_htr_text = ""
            all_step1_text = ""
            
            # Aggregate raw text for confidence scoring
            sorted_imgs = sorted(diplomatic_results.keys())
            for img_name in sorted_imgs:
                # Get HTR Text
                htr_lines = group_lines_by_image.get(img_name, [])
                for l in htr_lines:
                    all_htr_text += l.get('htr_text', '') + " "
                
                # Get Step 1 Text
                step1_data = diplomatic_results.get(img_name, {})
                for l in step1_data.get('lines', []):
                    all_step1_text += l.get('transcription', '') + " "

            # 2. EXECUTE STEP 2a: MERGE & EXTRACT (The Missing Link)
            step2a_path = os.path.join(out_dir, "step2a_merged.json")
            data_2a = {}
            step2a_loaded_from_disk = False
            token_usage_2a = None
            
            # Force rerun if in post-pylaia rerun mode
            should_skip_2a = os.path.exists(step2a_path) and not self.force and not self.rerun_from_post_pylaia

            if should_skip_2a:
                logger.info(f"[{gid}] Skipping Step 2a LLM (Loading from {step2a_path})...")
                try:
                    with open(step2a_path, 'r', encoding='utf-8') as f:
                        data_2a = json.load(f)
                        # Handle case where data_2a is a list instead of dict
                        if isinstance(data_2a, list):
                            logger.warning(f"[{gid}] data_2a loaded as list, extracting first element")
                            if len(data_2a) > 0 and isinstance(data_2a[0], dict):
                                data_2a = data_2a[0]
                            else:
                                logger.error(f"[{gid}] data_2a list is empty or first element is not a dict. Re-running.")
                                data_2a = {}
                        elif not isinstance(data_2a, dict):
                            logger.error(f"[{gid}] data_2a is not a dict (type: {type(data_2a)}). Re-running.")
                            data_2a = {}
                        step2a_loaded_from_disk = True
                    # Try to load token usage from multiple sources
                    token_usage_2a = load_token_usage_from_sources("step2a_merge_and_extract")
                    if token_usage_2a:
                        logger.info(f"[{gid}] Loaded Step 2a token usage from available sources")
                except Exception as e:
                    logger.error(f"[{gid}] Failed to load Step 2a file: {e}. Re-running.")

            if not data_2a:
                logger.info(f"[{gid}] Running Step 2a: Merging and Entity Extraction...")
                
                # Validate diplomatic_results before building prompt
                logger.debug(f"[{gid}] Validating diplomatic_results before Step 2a...")
                for img_name, data in diplomatic_results.items():
                    if not isinstance(data, dict):
                        logger.error(f"[{gid}] diplomatic_results[{img_name}] is not a dict: {type(data)}")
                        logger.error(f"  Value (first 500 chars): {str(data)[:500]}")
                    elif "lines" not in data:
                        logger.error(f"[{gid}] diplomatic_results[{img_name}] missing 'lines' key")
                    else:
                        lines = data.get("lines")
                        if not isinstance(lines, list):
                            logger.error(f"[{gid}] diplomatic_results[{img_name}]['lines'] is not a list: {type(lines)}")
                        else:
                            # Check each line in the list
                            for line_idx, line in enumerate(lines[:5]):  # Check first 5 lines
                                if not isinstance(line, dict):
                                    logger.error(f"[{gid}] diplomatic_results[{img_name}]['lines'][{line_idx}] is not a dict: {type(line)}")
                                    logger.error(f"  Line value (first 200 chars): {str(line)[:200]}")
                
                # Validate group_lines_by_image
                if group_lines_by_image:
                    logger.debug(f"[{gid}] Validating group_lines_by_image...")
                    for img_name, htr_lines in group_lines_by_image.items():
                        if not isinstance(htr_lines, list):
                            logger.error(f"[{gid}] group_lines_by_image[{img_name}] is not a list: {type(htr_lines)}")
                        else:
                            for line_idx, htr_line in enumerate(htr_lines[:3]):  # Check first 3 lines
                                if not isinstance(htr_line, dict):
                                    logger.error(f"[{gid}] group_lines_by_image[{img_name}][{line_idx}] is not a dict: {type(htr_line)}")
                
                try:
                    parts_2a = self.prompt_step2a_merge(diplomatic_results, group_lines_by_image)
                except Exception as e:
                    logger.error(f"[{gid}] Error building Step 2a prompt: {e}", exc_info=True)
                    logger.error(f"[{gid}] diplomatic_results keys: {list(diplomatic_results.keys())}")
                    for img_name, data in diplomatic_results.items():
                        logger.error(f"[{gid}]   {img_name}: type={type(data)}, keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    raise
                
                raw_json_2a, token_usage_2a = self._generate_batch_single(
                    model_name=MODEL_TEXT,
                    parts=parts_2a,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", 
                        response_schema=get_merged_diplomatic_schema(),
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level="LOW")
                    ),
                    log_filepath=os.path.join(out_dir, "step2a_thoughts.log"),
                    batch_key=f"{gid}_step2a",
                    output_dir=out_dir
                )
                
                # Save raw response for debugging
                raw_response_path = os.path.join(out_dir, "step2a_raw_response.json")
                try:
                    with open(raw_response_path, 'w', encoding='utf-8') as f:
                        f.write(raw_json_2a)
                    logger.debug(f"[{gid}] Saved raw Step 2a response to {raw_response_path}")
                except Exception as e:
                    logger.warning(f"[{gid}] Failed to save raw response: {e}")
                
                cleaned_json = clean_json_string(raw_json_2a)
                
                # Try to parse JSON, with repair attempt if it fails
                try:
                    data_2a = json.loads(cleaned_json)
                except json.JSONDecodeError as e:
                    # Save cleaned (but malformed) JSON for debugging
                    cleaned_json_path = os.path.join(out_dir, "step2a_cleaned_json_debug.json")
                    try:
                        with open(cleaned_json_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_json)
                        logger.warning(
                            f"[{gid}] JSON parsing failed at line {e.lineno}, column {e.colno} (char {e.pos}): {e.msg}. "
                            f"Cleaned JSON saved to {cleaned_json_path}. Attempting to repair..."
                        )
                    except Exception as save_error:
                        logger.warning(f"[{gid}] Failed to save cleaned JSON: {save_error}")
                    
                    # Try to repair the JSON
                    repaired_json = repair_json_string(cleaned_json)
                    try:
                        data_2a = json.loads(repaired_json)
                        logger.info(f"[{gid}] Successfully repaired and parsed JSON after initial failure")
                    except json.JSONDecodeError as repair_error:
                        # Save repaired JSON for debugging
                        repaired_json_path = os.path.join(out_dir, "step2a_repaired_json_debug.json")
                        try:
                            with open(repaired_json_path, 'w', encoding='utf-8') as f:
                                f.write(repaired_json)
                            logger.error(
                                f"[{gid}] JSON repair failed at line {repair_error.lineno}, column {repair_error.colno} "
                                f"(char {repair_error.pos}): {repair_error.msg}. "
                                f"Repaired JSON saved to {repaired_json_path}, raw response saved to {raw_response_path}"
                            )
                        except Exception as save_error2:
                            logger.error(f"[{gid}] Failed to save repaired JSON: {save_error2}")
                        
                        raise json.JSONDecodeError(
                            f"JSON parsing failed even after repair attempt. Original error: {e.msg} "
                            f"(line {e.lineno}, col {e.colno}, char {e.pos}). "
                            f"Repair error: {repair_error.msg} (line {repair_error.lineno}, col {repair_error.colno}, "
                            f"char {repair_error.pos}). Raw response saved to {raw_response_path}",
                            repaired_json,
                            repair_error.pos
                        )
                
                # Handle case where data_2a is a list instead of dict
                if isinstance(data_2a, list):
                    logger.warning(f"[{gid}] data_2a parsed as list (length: {len(data_2a)})")
                    if len(data_2a) > 0:
                        first_elem = data_2a[0]
                        if isinstance(first_elem, dict):
                            logger.info(f"[{gid}] Extracting first dict element from list")
                            data_2a = first_elem
                        else:
                            # List contains non-dict items (e.g., strings)
                            logger.error(f"[{gid}] data_2a is a list but elements are not dicts")
                            logger.error(f"[{gid}] First element type: {type(first_elem)}, value: {first_elem}")
                            logger.error(f"[{gid}] Full list (first 20 items): {data_2a[:20] if len(data_2a) > 20 else data_2a}")
                            logger.error(f"[{gid}] This suggests the LLM returned just an array (possibly place_names) instead of the expected object structure")
                            logger.error(f"[{gid}] Raw response saved to: {raw_response_path}")
                            raise ValueError(
                                f"data_2a is a list of {type(first_elem).__name__} instead of dict. "
                                f"Expected a dict with keys: merged_text, surnames, place_names, marginal_county. "
                                f"Got: {data_2a[:10] if len(data_2a) > 10 else data_2a}... "
                                f"(Raw response saved to {raw_response_path})"
                            )
                    else:
                        logger.error(f"[{gid}] data_2a is an empty list")
                        logger.error(f"[{gid}] Raw response saved to: {raw_response_path}")
                        raise ValueError(f"data_2a is an empty list. Raw response saved to {raw_response_path}")
                elif not isinstance(data_2a, dict):
                    logger.error(f"[{gid}] data_2a is not a dict (type: {type(data_2a)})")
                    logger.error(f"[{gid}] Value: {str(data_2a)[:500]}")
                    logger.error(f"[{gid}] Raw response saved to: {raw_response_path}")
                    raise ValueError(
                        f"data_2a is not a dict: {type(data_2a)}. "
                        f"Expected a dict with keys: merged_text, surnames, place_names, marginal_county. "
                        f"Raw response saved to {raw_response_path}"
                    )

            merged_text = data_2a.get("merged_text", "")
            extracted_surnames = data_2a.get("surnames", [])
            extracted_places = data_2a.get("place_names", [])
            marginal_county = data_2a.get("marginal_county", {"original": "", "anglicized": ""})
            
            # Normalize marginal_county: sometimes LLM returns it as a list instead of a dict
            if isinstance(marginal_county, list):
                logger.warning(f"[{gid}] marginal_county is a list, converting to dict (taking first element)")
                if len(marginal_county) > 0 and isinstance(marginal_county[0], dict):
                    marginal_county = marginal_county[0]
                else:
                    logger.warning(f"[{gid}] marginal_county list is empty or first element is not a dict, using default")
                    marginal_county = {"original": "", "anglicized": ""}
            elif not isinstance(marginal_county, dict):
                logger.warning(f"[{gid}] marginal_county is not a dict or list (type: {type(marginal_county)}), using default")
                marginal_county = {"original": "", "anglicized": ""}
            
            # --- CONSTRUCT COUNTY INFO FROM MARGINAL COUNTY ---
            county_info = {
                "county": marginal_county.get("anglicized", "UNKNOWN"),
                "county_original_latin": marginal_county.get("original", ""),
                "source": "marginal_annotation" if marginal_county.get("anglicized") else "not_found",
                "confidence": "high" if marginal_county.get("anglicized") else "none"
            }
            logger.info(f"[{gid}] Extracted county from margin: {county_info['county']} (Latin: {county_info['county_original_latin']})")

            # --- BAYESIAN PROBABILITY SCORING ---
            # Extract probabilities from post_correction.json files for confidence scoring.
            # We use Bayesian probability instead of the old multi-source confidence scoring.
            
            # 1. Process Surnames
            scored_surnames = []
            corrected_surnames_list = [] 

            for sn in extracted_surnames:
                # Extract Bayesian probability from post_correction results
                probability = self._extract_bayesian_probability(sn, "surname", post_correction_results)
                
                score_data = {
                    "term": sn,
                    "probability": probability if probability is not None else None
                }
                
                final_surname_val = sn
                scored_surnames.append(score_data)
                corrected_surnames_list.append(final_surname_val)

            # 2. Process Place Names
            scored_places = []
            corrected_places_list = [] 

            for pl in extracted_places:
                orig = pl.get('original', '')
                anglicized = pl.get('anglicized', '')
                
                # Extract Bayesian probability from post_correction results
                # Try original first, then anglicized
                probability = self._extract_bayesian_probability(orig, "placename", post_correction_results)
                if probability is None and anglicized:
                    probability = self._extract_bayesian_probability(anglicized, "placename", post_correction_results)
                
                score_data = {
                    "term": orig,
                    "anglicized": anglicized,
                    "probability": probability if probability is not None else None
                }
                
                scored_places.append(score_data)
                corrected_places_list.append(pl)

            # --- UPDATE DATA_2A AND SAVE (If new or changed) ---
            data_2a['merged_text'] = merged_text
            data_2a['surnames'] = corrected_surnames_list
            data_2a['place_names'] = corrected_places_list

            # Only write to disk if we didn't just load it, or if you want to ensure corrections are saved
            if not step2a_loaded_from_disk:
                with open(step2a_path, "w", encoding="utf-8") as f:
                    json.dump(data_2a, f, indent=2)

            # Step 2b: Expand
            latin_text = ""
            step2b_path = os.path.join(out_dir, "step2b_latin_expanded.txt")
            token_usage_2b = None
            
            should_skip_2b = os.path.exists(step2b_path) and not self.force and not self.rerun_from_post_pylaia
            if should_skip_2b:
                logger.info(f"[{gid}] Skipping Step 2b LLM (Loading from disk)...")
                with open(step2b_path, 'r', encoding='utf-8') as f:
                    latin_text = f.read()
                # Try to load token usage from multiple sources
                token_usage_2b = load_token_usage_from_sources("step2b_expand_abbreviations")
                if token_usage_2b:
                    logger.info(f"[{gid}] Loaded Step 2b token usage from available sources")
            else:
                try:
                    parts = self.prompt_step2b_expansion(merged_text)
                    latin_text, token_usage_2b = self._generate_batch_single(
                        model_name=MODEL_TEXT,
                        parts=parts,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level="minimal")
                        ),
                        log_filepath=os.path.join(out_dir, "step2b_thoughts.log"),
                        batch_key=f"{gid}_step2b",
                        output_dir=out_dir
                    )
                    with open(step2b_path, "w", encoding="utf-8") as f: f.write(latin_text)
                except Exception as e:
                    logger.error(f"Step 2b failed for {gid} (using merged text): {e}")
                    latin_text = merged_text

            # Step 3: Translate
            english_text = ""
            step3_path = os.path.join(out_dir, "step3_english.txt")
            token_usage_3 = None
            
            should_skip_3 = os.path.exists(step3_path) and not self.force and not self.rerun_from_post_pylaia
            if should_skip_3:
                logger.info(f"[{gid}] Skipping Step 3 LLM (Loading from disk)...")
                with open(step3_path, 'r', encoding='utf-8') as f:
                    english_text = f.read()
                # Try to load token usage from multiple sources
                token_usage_3 = load_token_usage_from_sources("step3_translation")
                if token_usage_3:
                    logger.info(f"[{gid}] Loaded Step 3 token usage from available sources")
            else:
                parts = self.prompt_step3_translation(latin_text)
                english_text, token_usage_3 = self._generate_batch_single(
                    model_name=MODEL_TEXT,
                    parts=parts,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level="minimal")
                    ),
                    log_filepath=os.path.join(out_dir, "step3_thoughts.log"),
                    batch_key=f"{gid}_step3",
                    output_dir=out_dir
                )
                with open(step3_path, "w", encoding="utf-8") as f: f.write(english_text)

            # Step 4: Indexing & Master Record
            
            # Metadata extraction (Always needed for Context, even if we skip LLM)
            roll_match = re.search(r'(?:CP\s*40|Roll|no)[-._\s]+(\d+)', gid, re.IGNORECASE)
            rot_match = re.search(r'(?:f|rot|m)[-._\s]*(\d+[a-z]?)', gid, re.IGNORECASE) 
            if not rot_match: rot_match = re.search(r'[-._\s](\d+[a-z]?)$', gid)
            
            roll_num = roll_match.group(1) if roll_match else None
            rot_num = rot_match.group(1) if rot_match else None
            date_info = get_cp40_info(roll_num) if roll_num else None

            # --- FETCH GROUND TRUTH (Always needed) ---
            ground_truth_cases = []
            if roll_num and rot_num:
                logger.info(f"[{gid}] Attempting to fetch DB Ground Truth for Roll {roll_num}, Rot {rot_num}")
                ground_truth_cases = self.get_ground_truth_from_db(roll_num, rot_num)
                if ground_truth_cases:
                    logger.info(f"[{gid}] Found {len(ground_truth_cases)} matching ground truth cases in DB.")
                else:
                    logger.info(f"[{gid}] No ground truth found in DB.")

            # Check if Final JSON exists
            final_data = {}
            token_usage_4 = None
            should_skip_4 = os.path.exists(final_json_path) and not self.force and not self.rerun_from_post_pylaia
            if should_skip_4:
                logger.info(f"[{gid}] Skipping Step 4 LLM (Loading final_index.json)...")
                try:
                    with open(final_json_path, 'r', encoding='utf-8') as f:
                        final_data = json.load(f)
                    # Try to load token usage from multiple sources
                    token_usage_4 = load_token_usage_from_sources("step4_indexing")
                    if token_usage_4:
                        logger.info(f"[{gid}] Loaded Step 4 token usage from available sources")
                except Exception as e:
                     logger.error(f"[{gid}] Failed to load final index: {e}. Re-running.")

            if not final_data:
                # Ensure county_info has a value (fallback if step 2a didn't extract it)
                if not county_info:
                    county_info = {
                        "county": "UNKNOWN",
                        "source": "not_extracted",
                        "confidence": "none",
                        "county_original_latin": ""
                    }
                    logger.warning(f"[{gid}] County info not available from Step 2a, using UNKNOWN")
                
                parts = self.prompt_step4_indexing(
                    english_text, 
                    latin_text=latin_text,
                    date_info=date_info,
                    county_info=county_info
                )
                raw_json, token_usage_4 = self._generate_batch_single(
                    model_name=MODEL_TEXT,
                    parts=parts,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=get_final_index_schema(),
                        temperature=0.0,
                        max_output_tokens=64000,
                        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level="MEDIUM")
                    ),
                    log_filepath=os.path.join(out_dir, "step4_thoughts.log"),
                    batch_key=f"{gid}_step4",
                    output_dir=out_dir
                )
                try:
                    cleaned_json = clean_json_string(raw_json)
                    final_data = json.loads(cleaned_json)
                except json.JSONDecodeError as e:
                    # Save raw and cleaned JSON for debugging
                    debug_json_path = os.path.join(out_dir, "step4_raw_json_debug.json")
                    debug_cleaned_path = os.path.join(out_dir, "step4_cleaned_json_debug.json")
                    try:
                        with open(debug_json_path, 'w', encoding='utf-8') as f:
                            f.write(raw_json)
                        cleaned_json = clean_json_string(raw_json)
                        with open(debug_cleaned_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_json)
                        logger.error(
                            f"[{gid}] JSON parsing failed at line {e.lineno}, column {e.colno} (char {e.pos}): {e.msg}. "
                            f"Raw JSON saved to {debug_json_path}, cleaned JSON saved to {debug_cleaned_path}"
                        )
                    except Exception as save_error:
                        logger.error(f"[{gid}] Failed to save debug JSON files: {save_error}")
                    
                    # Try to extract JSON from step4_thoughts.log
                    thoughts_log_path = os.path.join(out_dir, "step4_thoughts.log")
                    json_extracted = False
                    if os.path.exists(thoughts_log_path):
                        try:
                            logger.info(f"[{gid}] Attempting to extract JSON from {thoughts_log_path}")
                            with open(thoughts_log_path, 'r', encoding='utf-8') as f:
                                log_content = f.read()
                            
                            # Find JSON in the log (look for array or object starting with [ or {)
                            # JSON typically appears after the thinking text
                            json_start = -1
                            for start_char in ['[', '{']:
                                pos = log_content.rfind(start_char)
                                if pos > json_start:
                                    json_start = pos
                            
                            if json_start >= 0:
                                # Extract from the first [ or { to the end, then find matching closing bracket
                                json_candidate = log_content[json_start:]
                                # Try to find the end of the JSON by finding matching brackets
                                bracket_count = 0
                                brace_count = 0
                                json_end = -1
                                in_string = False
                                escape_next = False
                                
                                for i, char in enumerate(json_candidate):
                                    if escape_next:
                                        escape_next = False
                                        continue
                                    if char == '\\':
                                        escape_next = True
                                        continue
                                    if char == '"' and not escape_next:
                                        in_string = not in_string
                                        continue
                                    if not in_string:
                                        if char == '[':
                                            bracket_count += 1
                                        elif char == ']':
                                            bracket_count -= 1
                                            if bracket_count == 0 and json_candidate[0] == '[':
                                                json_end = i + 1
                                                break
                                        elif char == '{':
                                            brace_count += 1
                                        elif char == '}':
                                            brace_count -= 1
                                            if brace_count == 0 and json_candidate[0] == '{':
                                                json_end = i + 1
                                                break
                                
                                # If we found a complete JSON, use it
                                if json_end > 0:
                                    json_from_log = json_candidate[:json_end].strip()
                                    # Clean and parse
                                    cleaned_log_json = clean_json_string(json_from_log)
                                    final_data = json.loads(cleaned_log_json)
                                    # If it's an array, extract the first element (or handle as needed)
                                    if isinstance(final_data, list) and len(final_data) > 0:
                                        final_data = final_data[0]
                                    logger.info(f"[{gid}] Successfully extracted JSON from thoughts log")
                                    json_extracted = True
                                else:
                                    # JSON might be incomplete - try to fix it by closing brackets/braces
                                    logger.info(f"[{gid}] JSON appears incomplete in thoughts log, attempting to fix...")
                                    json_from_log = json_candidate.strip()
                                    
                                    # Remove trailing incomplete values (like unclosed strings)
                                    # Find the last complete key-value pair or array element
                                    json_fixed = json_from_log.rstrip()
                                    
                                    # Remove trailing comma
                                    json_fixed = json_fixed.rstrip(',').rstrip()
                                    
                                    # If we're in the middle of a string value, remove the incomplete value
                                    # Look for patterns like: "key": "incomplete or "key": incomplete
                                    # Find the last complete closing quote, brace, or bracket
                                    last_complete_pos = -1
                                    in_string = False
                                    escape_next = False
                                    
                                    for i in range(len(json_fixed) - 1, -1, -1):
                                        char = json_fixed[i]
                                        if escape_next:
                                            escape_next = False
                                            continue
                                        if char == '\\':
                                            escape_next = True
                                            continue
                                        if char == '"' and not escape_next:
                                            in_string = not in_string
                                            if not in_string:  # Found a closing quote
                                                # Check if this looks like the end of a complete value
                                                if i < len(json_fixed) - 1:
                                                    after = json_fixed[i+1:].strip()
                                                    if after in ['', ',', '}', ']']:
                                                        last_complete_pos = i + 1
                                                        break
                                        elif not in_string:
                                            if char in ['}', ']']:
                                                last_complete_pos = i + 1
                                                break
                                    
                                    # If we found a complete position, truncate there
                                    if last_complete_pos > 0:
                                        json_fixed = json_fixed[:last_complete_pos].rstrip().rstrip(',')
                                    
                                    # Count unclosed brackets and braces
                                    bracket_count = 0
                                    brace_count = 0
                                    in_string = False
                                    escape_next = False
                                    
                                    for char in json_fixed:
                                        if escape_next:
                                            escape_next = False
                                            continue
                                        if char == '\\':
                                            escape_next = True
                                            continue
                                        if char == '"' and not escape_next:
                                            in_string = not in_string
                                            continue
                                        if not in_string:
                                            if char == '[':
                                                bracket_count += 1
                                            elif char == ']':
                                                bracket_count -= 1
                                            elif char == '{':
                                                brace_count += 1
                                            elif char == '}':
                                                brace_count -= 1
                                    
                                    # Add missing closing brackets/braces (close innermost first)
                                    if json_fixed.startswith('['):
                                        json_fixed += '}' * brace_count
                                        json_fixed += ']' * bracket_count
                                    elif json_fixed.startswith('{'):
                                        json_fixed += ']' * bracket_count
                                        json_fixed += '}' * brace_count
                                    
                                    try:
                                        cleaned_log_json = clean_json_string(json_fixed)
                                        final_data = json.loads(cleaned_log_json)
                                        # If it's an array, extract the first element
                                        if isinstance(final_data, list) and len(final_data) > 0:
                                            final_data = final_data[0]
                                        logger.info(f"[{gid}] Successfully extracted and fixed incomplete JSON from thoughts log")
                                        json_extracted = True
                                    except json.JSONDecodeError as fix_error:
                                        logger.warning(f"[{gid}] Could not fix incomplete JSON: {fix_error}")
                            else:
                                logger.warning(f"[{gid}] No JSON found in thoughts log")
                        except Exception as log_extract_error:
                            logger.warning(f"[{gid}] Failed to extract JSON from thoughts log: {log_extract_error}")
                    
                    # If still not extracted, try the cleaned_json_debug.json file as a fallback
                    if not json_extracted:
                        debug_cleaned_path = os.path.join(out_dir, "step4_cleaned_json_debug.json")
                        if os.path.exists(debug_cleaned_path):
                            try:
                                logger.info(f"[{gid}] Attempting to extract JSON from {debug_cleaned_path}")
                                with open(debug_cleaned_path, 'r', encoding='utf-8') as f:
                                    debug_content = f.read().strip()
                                
                                # Try to fix incomplete JSON by adding missing closing braces
                                bracket_count = 0
                                brace_count = 0
                                in_string = False
                                escape_next = False
                                
                                for char in debug_content:
                                    if escape_next:
                                        escape_next = False
                                        continue
                                    if char == '\\':
                                        escape_next = True
                                        continue
                                    if char == '"' and not escape_next:
                                        in_string = not in_string
                                        continue
                                    if not in_string:
                                        if char == '[':
                                            bracket_count += 1
                                        elif char == ']':
                                            bracket_count -= 1
                                        elif char == '{':
                                            brace_count += 1
                                        elif char == '}':
                                            brace_count -= 1
                                
                                # Fix incomplete JSON
                                debug_fixed = debug_content.rstrip(',').rstrip()
                                if debug_fixed.startswith('{'):
                                    debug_fixed += '}' * brace_count
                                    debug_fixed += ']' * bracket_count
                                elif debug_fixed.startswith('['):
                                    debug_fixed += '}' * brace_count
                                    debug_fixed += ']' * bracket_count
                                
                                try:
                                    cleaned_debug_json = clean_json_string(debug_fixed)
                                    final_data = json.loads(cleaned_debug_json)
                                    logger.info(f"[{gid}] Successfully extracted JSON from cleaned debug file")
                                    json_extracted = True
                                except json.JSONDecodeError as debug_error:
                                    logger.warning(f"[{gid}] Could not parse cleaned debug JSON: {debug_error}")
                            except Exception as debug_extract_error:
                                logger.warning(f"[{gid}] Failed to extract from debug file: {debug_extract_error}")
                    
                    if not json_extracted:
                        raise
            
            # --- PHASE 5: VALIDATION ---
            validation_results = self.validate_extraction(
                final_data, 
                english_text, 
                latin_text
            )
            
            if validation_results['critical_count'] > 0:
                logger.warning(f"Validation found {validation_results['critical_count']} critical issues")

            # Construct Master Record
            master_images_data = []
            for img_path in paths:
                img_filename = os.path.basename(img_path)
                htr_lines = group_lines_by_image.get(img_filename, [])
                diplomatic_data = diplomatic_results.get(img_filename, {})
                diplomatic_lines_list = diplomatic_data.get("lines", [])
                diplo_map = {l.get("id"): l.get("transcription", "") for l in diplomatic_lines_list}

                consolidated_lines = []
                for idx, htr_item in enumerate(htr_lines, 1):
                    simple_id = f"L{idx:02d}"
                    consolidated_lines.append({
                        "line_id": simple_id,
                        "original_file_id": htr_item.get("line_id"),
                        "kraken_polygon": htr_item.get("polygon"),
                        "kraken_bbox": htr_item.get("bbox"),
                        "text_htr": htr_item.get("htr_text"),
                        "word_confidences": htr_item.get("word_confidences", []),
                        "text_diplomatic": diplo_map.get(simple_id, "")
                    })
                master_images_data.append({"filename": img_filename, "lines": consolidated_lines})

            # Collect all token usage information
            token_usage_data = {}
            if token_usage_step1 is not None:
                token_usage_data["step1_diplomatic_transcription"] = token_usage_step1
            else:
                logger.warning(f"[{gid}] Step 1 token usage is None - batch job may not have been run or token usage not available")
            if token_usage_2a:
                token_usage_data["step2a_merge_and_extract"] = token_usage_2a
            if token_usage_2b:
                token_usage_data["step2b_expand_abbreviations"] = token_usage_2b
            if token_usage_3:
                token_usage_data["step3_translation"] = token_usage_3
            if token_usage_4:
                token_usage_data["step4_indexing"] = token_usage_4
            
            # Calculate totals
            total_prompt = sum(usage.get("prompt_tokens", 0) for usage in token_usage_data.values())
            total_cached = sum(usage.get("cached_tokens", 0) for usage in token_usage_data.values())
            total_response = sum(usage.get("response_tokens", 0) for usage in token_usage_data.values())
            total_thoughts = sum(usage.get("thoughts_tokens", 0) for usage in token_usage_data.values())
            total_all = sum(usage.get("total_tokens", 0) for usage in token_usage_data.values())
            total_non_cached = sum(usage.get("non_cached_input", 0) for usage in token_usage_data.values())
            
            token_usage_data["_totals"] = {
                "prompt_tokens": total_prompt,
                "cached_tokens": total_cached,
                "response_tokens": total_response,
                "thoughts_tokens": total_thoughts,
                "total_tokens": total_all,
                "non_cached_input": total_non_cached
            }
            
            # Calculate estimated costs
            def calculate_step_cost(usage: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
                """Calculate cost for a single step.
                
                Gemini 3 Flash pricing: $0.25 per million input tokens, $1.50 per million output tokens (including thinking)
                
                Returns:
                    Tuple of (total_cost, breakdown_dict)
                """
                if not usage:
                    return 0.0, {
                        "input_cost": 0.0,
                        "output_cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0
                    }
                
                # Input tokens = prompt_tokens + cached_tokens
                input_tokens = usage.get("prompt_tokens", 0) + usage.get("cached_tokens", 0)
                # Output tokens = response_tokens + thoughts_tokens
                output_tokens = usage.get("response_tokens", 0) + usage.get("thoughts_tokens", 0)
                
                input_cost = (input_tokens / 1_000_000) * COST_INPUT_TOKENS
                output_cost = (output_tokens / 1_000_000) * COST_OUTPUT_TOKENS
                total_cost = input_cost + output_cost
                
                breakdown = {
                    "input_cost": round(input_cost, 6),
                    "output_cost": round(output_cost, 6),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
                return total_cost, breakdown
            
            # Calculate costs for each step
            cost_breakdown = {}
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            
            for step_name, usage in token_usage_data.items():
                if step_name == "_totals":
                    continue  # Skip totals, we'll calculate separately
                step_cost, step_breakdown = calculate_step_cost(usage)
                cost_breakdown[step_name] = {
                    "cost_usd": round(step_cost, 6),
                    "breakdown": step_breakdown
                }
                total_cost += step_cost
                total_input_tokens += step_breakdown["input_tokens"]
                total_output_tokens += step_breakdown["output_tokens"]
            
            # Calculate total cost breakdown
            total_input_cost = (total_input_tokens / 1_000_000) * COST_INPUT_TOKENS
            total_output_cost = (total_output_tokens / 1_000_000) * COST_OUTPUT_TOKENS
            
            cost_breakdown["_total_cost_usd"] = round(total_cost, 6)
            cost_breakdown["_total_breakdown"] = {
                "input_cost": round(total_input_cost, 6),
                "output_cost": round(total_output_cost, 6),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            }
            
            # Ensure county_info exists for master record
            if not county_info:
                county_info = {
                    "county": "UNKNOWN",
                    "source": "not_extracted",
                    "confidence": "none",
                    "county_original_latin": ""
                }
            
            comprehensive_record = {
                "case_metadata": {
                    "group_id": gid,
                    "roll_number": roll_num,
                    "rotulus_number": rot_num,
                    "county": county_info.get("county", "UNKNOWN"),
                    "county_source": county_info.get("source", "not_extracted"),
                    "county_confidence": county_info.get("confidence", "none"),
                    "county_original_latin": county_info.get("county_original_latin", ""),
                    "processed_at": time.ctime(),
                    "date_context": date_info
                },
                "ground_truth_from_db": [ground_truth_cases], 
                "source_material": master_images_data,
                "text_content": {
                    "latin_reconstructed": latin_text,
                    "english_translation": english_text
                },
                "extracted_entities": {
                    "surnames": scored_surnames,
                    "place_names": scored_places,
                    "marginal_county": marginal_county if marginal_county and marginal_county.get("anglicized") else {"original": "", "anglicized": ""}
                },
                "legal_index": final_data,
                "validation": validation_results,
                "token_usage": token_usage_data,
                "estimated_cost": cost_breakdown
            }
            # Save Finals
            with open(os.path.join(out_dir, "master_record.json"), "w", encoding="utf-8") as f:
                json.dump(comprehensive_record, f, indent=2, ensure_ascii=False)
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)

            logger.info(f"SUCCESS: Group {gid} fully processed.")

            # --- PHASE 6: PDF REPORT GENERATION ---
            logger.info(f"[{gid}] Checking for PDF validation report...")
            try:
                self._ensure_pdf_report(gid, out_dir)
            except Exception as e:
                logger.warning(f"[{gid}] PDF generation failed (non-blocking): {e}. Continuing...")

        except Exception as e:
            logger.error(f"Error during chain processing for {gid}: {e}", exc_info=True)
        
        finally:
            # --- CLEANUP ---
            # Delete line images that belong to this worker's group
            # Only delete files tracked by this worker to avoid interfering with other concurrent workers
            if worker_line_images:
                deleted_count = 0
                error_count = 0
                for line_image_path in worker_line_images:
                    try:
                        if os.path.exists(line_image_path):
                            os.remove(line_image_path)
                            deleted_count += 1
                    except OSError as e:
                        logger.warning(f"[{gid}] Could not delete line image {line_image_path}: {e}")
                        error_count += 1
                
                if deleted_count > 0:
                    logger.info(f"[{gid}] Cleaned up {deleted_count} line image file(s) (worker-specific cleanup)")
                if error_count > 0:
                    logger.warning(f"[{gid}] Failed to delete {error_count} line image file(s)")
            
            # Delete cloud files immediately to make room for next group
            self.cleanup_cloud_files()
            self.uploaded_files_cache = {}

    def execute(self, groups: Dict[str, List[str]]) -> None:
        """
        Execute the complete workflow pipeline for all image groups.

        Processes each case group through the full pipeline:
        1. Kraken (line segmentation)
        2. PyLaia (HTR recognition)
        3. Post-correction and named entity extraction (LLM + Bayesian)
        4. Stitching (Step 2a: merge transcriptions)
        5. Expansion (Step 2b: expand abbreviations)
        6. Translation (Step 3: Latin to English)
        7. Indexing (Step 4: structured extraction)
        8. Validation and confidence scoring
        9. Final index generation

        Skips groups that already have final_index.json unless force=True.
        If rerun_from_post_pylaia=True, keeps existing Kraken/PyLaia results but
        reruns everything from post-correction onwards.
        Processes groups in parallel with up to 5 groups pending at a time.

        Args:
            groups: Dictionary mapping group IDs to lists of image file paths.
                Typically produced by ImageGrouper.scan().
        """
        if not groups:
            logger.error("No groups found.")
            return

        logger.info(f"=== STARTING PARALLEL PROCESSING ({len(groups)} Groups, max 5 concurrent) ===")

        # Filter out groups that should be skipped before processing
        groups_to_process = {}
        skipped_groups = []
        for gid, paths in groups.items():
            out_dir = os.path.join(OUTPUT_DIR, gid.replace(" ", "_"))
            final_json_path = os.path.join(out_dir, "final_index.json")
            if os.path.exists(final_json_path) and not self.force and not self.rerun_from_post_pylaia:
                logger.info(f"Skipping Group {gid} (Already Complete).")
                skipped_groups.append((gid, out_dir))
                continue
            groups_to_process[gid] = paths
        
        # Check and generate PDF reports for skipped groups if missing
        if skipped_groups:
            logger.info(f"Checking PDF reports for {len(skipped_groups)} already-complete groups...")
            for gid, out_dir in skipped_groups:
                try:
                    self._ensure_pdf_report(gid, out_dir)
                except Exception as e:
                    logger.warning(f"[{gid}] PDF generation failed (non-blocking): {e}. Continuing...")

        if not groups_to_process:
            logger.info("No groups to process (all already complete).")
            return

        logger.info(f"Processing {len(groups_to_process)} groups in parallel (max 5 concurrent)...")

        # Process groups in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all group processing tasks
            future_to_gid = {
                executor.submit(self._process_group, gid, paths): gid
                for gid, paths in groups_to_process.items()
            }
            
            # Process completed tasks as they finish
            completed = 0
            total = len(future_to_gid)
            for future in as_completed(future_to_gid):
                gid = future_to_gid[future]
                completed += 1
                try:
                    future.result()  # This will raise any exception that occurred
                    logger.info(f"Completed group {gid} ({completed}/{total})")
                except Exception as e:
                    logger.error(f"Group {gid} failed with error: {e}", exc_info=True)

        logger.info(f"=== PARALLEL PROCESSING COMPLETE ({completed}/{total} groups processed) ===")
