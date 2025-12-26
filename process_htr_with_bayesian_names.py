#!/usr/bin/env python3
"""
Process HTR text from htr_work subfolders, extract names using Gemini 2.5 Flash,
and use Bayesian approach with Pylaia loss to select best candidates from cp40_records.db.

=================================================================================
BAYESIAN FRAMEWORK
=================================================================================

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

=================================================================================
KEY DESIGN DECISIONS
=================================================================================

1. CONTEXT: Use raw HTR text (not Gemini-corrected) as context for Pylaia,
   since it's closer to what the visual model expects.

2. ENTITY POSITIONING: Use fuzzy string matching (rapidfuzz) to locate entities
   in text, rather than relying on Gemini's character indices.

3. LOSS CALCULATION: Calculate CTC loss for just the entity region, not the
   full line, to avoid context effects dominating.

4. NORMALIZATION: Normalize CTC loss by entity character count to make
   comparisons fair across different-length candidates.

5. SMOOTHING: Use Laplace smoothing with configurable α to handle unseen names.

6. ORIGINAL BONUS: Give a small bonus to the original extraction (what Gemini saw)
   since it represents the model's best visual interpretation.
"""

import hashlib
import os
import json
import pickle
import sqlite3
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from math import log, exp
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    from laia.utils import SymbolsTable
    from google import genai
    from google.genai import types
    from symspellpy import SymSpell, Verbosity
    from rapidfuzz import fuzz, process as fuzz_process
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Install with: pip install symspellpy rapidfuzz google-genai")
    raise

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BayesianConfig:
    """Configuration for Bayesian candidate selection."""
    # Weights for combining likelihood and prior
    likelihood_weight: float = 1.0  # Weight for log P(image|candidate)
    prior_weight: float = 1.0       # Weight for log P(candidate)
    
    # Laplace smoothing parameter for prior
    smoothing_alpha: float = 1.0
    
    # Bonus for original extraction (in log space, added to score)
    # This reflects that Gemini's extraction has some validity
    original_bonus: float = 0.5
    
    # Penalty per edit distance from original (in log space, subtracted from score)
    # This favors candidates closer to what was actually written
    distance_penalty: float = 1.5
    
    # Minimum similarity for fuzzy position matching
    fuzzy_match_threshold: float = 80.0
    
    # Maximum edit distance for SymSpell candidate generation
    max_edit_distance: int = 2
    
    # Length ratio bounds for candidate filtering
    min_length_ratio: float = 0.7
    max_length_ratio: float = 1.4


# Paths
BASE_DIR = Path(__file__).parent
HTR_WORK_DIR = BASE_DIR / "bootstrap_training_data" / "htr_work"
PYLAIA_MODELS_DIR = BASE_DIR / "bootstrap_training_data" / "pylaia_models"
DB_PATH = BASE_DIR / "cp40_records.db"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Track which key is currently working (cached across calls)
_current_api_key: Optional[str] = None
_current_client: Optional[Any] = None


def get_gemini_client() -> Any:
    """
    Get a working Gemini 3.0 Flash client using paid API key.
    Caches the working client for subsequent calls.
    """
    global _current_api_key, _current_client
    
    if _current_client is not None:
        return _current_client
    
    # Use paid key from environment (required for batch mode)
    paid_key = os.environ.get('GEMINI_API_KEY')
    if not paid_key:
        raise RuntimeError("GEMINI_API_KEY environment variable must be set for Gemini 3.0 Flash batch mode")
    
    try:
        client = genai.Client(api_key=paid_key)
        # Test the client with a minimal request
        response = client.models.generate_content(
            model="gemini-3.0-flash",
            contents=[types.Part.from_text(text="test")],
            config=types.GenerateContentConfig(max_output_tokens=1)
        )
        # If we get here, the key works
        print(f"  ✓ Using paid API key for Gemini 3.0 Flash")
        _current_api_key = paid_key
        _current_client = client
        return client
    except Exception as e:
        print(f"  ✗ Paid key failed: {str(e)[:50]}...")
        raise RuntimeError(f"Failed to initialize Gemini 3.0 Flash client: {e}")


def clean_htr_text(text: str) -> str:
    """
    Clean raw HTR output by removing inter-character spaces
    and replacing <space> tokens with actual spaces.
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
    
    Args:
        entity: The entity text to find
        text: The full text to search in
        threshold: Minimum similarity score (0-100)
    
    Returns:
        (start_idx, end_idx) or None if not found
    """
    if not entity or not text:
        return None
    
    # Try exact match first (fastest)
    idx = text.find(entity)
    if idx != -1:
        return (idx, idx + len(entity))
    
    # Try case-insensitive exact match
    lower_text = text.lower()
    lower_entity = entity.lower()
    idx = lower_text.find(lower_entity)
    if idx != -1:
        return (idx, idx + len(entity))
    
    # Fuzzy matching: slide a window and find best match
    entity_len = len(entity)
    best_score = 0
    best_pos = None
    
    # Check each word and nearby regions
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
# PYLAIA MODEL FUNCTIONS
# =============================================================================

def find_latest_pylaia_model() -> Tuple[Path, Path, Path]:
    """Find the latest Pylaia model from pylaia_models directory."""
    if not PYLAIA_MODELS_DIR.exists():
        raise FileNotFoundError(f"Pylaia models directory not found: {PYLAIA_MODELS_DIR}")
    
    # Find all model directories
    model_dirs = []
    for item in PYLAIA_MODELS_DIR.iterdir():
        if item.is_dir() and item.name.startswith("model_v"):
            try:
                version = int(item.name.replace("model_v", ""))
                model_dirs.append((version, item))
            except ValueError:
                continue
    
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {PYLAIA_MODELS_DIR}")
    
    latest_version, latest_dir = max(model_dirs, key=lambda x: x[0])
    
    # Find checkpoint
    experiment_dir = latest_dir / "experiment"
    checkpoint_patterns = ["*-lowest_va_cer.ckpt", "*-last.ckpt"]
    
    checkpoint = None
    search_dirs = [experiment_dir, latest_dir] if experiment_dir.exists() else [latest_dir]
    
    for search_dir in search_dirs:
        for pattern in checkpoint_patterns:
            files = list(search_dir.glob(pattern))
            if files:
                checkpoint = max(files, key=lambda p: p.stat().st_mtime)
                break
        if checkpoint:
            break
    
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint found in {latest_dir}")
    
    model_file = latest_dir / "model"
    syms_file = latest_dir / "syms.txt"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not syms_file.exists():
        raise FileNotFoundError(f"Syms file not found: {syms_file}")
    
    print(f"Using model: model_v{latest_version}")
    print(f"  Checkpoint: {checkpoint.name}")
    
    return checkpoint, model_file, syms_file


def load_pylaia_model(checkpoint_path: Path, model_arch_path: Path, symbols_path: Path):
    """Load Pylaia model using laia's ModelLoader."""
    print(f"Loading symbols from: {symbols_path}")
    syms = SymbolsTable(str(symbols_path))
    
    try:
        from laia.common.loader import ModelLoader
        
        train_path = str(model_arch_path.parent)
        model_filename = model_arch_path.name
        
        loader = ModelLoader(train_path, filename=model_filename, device="cpu")
        
        # Try to prepare checkpoint - different laia versions have different signatures
        checkpoint = None
        try:
            # First try with just checkpoint path (newer laia versions)
            checkpoint = loader.prepare_checkpoint(str(checkpoint_path))
        except (TypeError, AttributeError) as e1:
            try:
                # Try with 3 arguments (older laia versions)
                checkpoint = loader.prepare_checkpoint(
                    str(checkpoint_path), experiment_dirpath=None, monitor=None
                )
            except (TypeError, AttributeError) as e2:
                # Fall back to using checkpoint path directly
                checkpoint = str(checkpoint_path)
        
        if checkpoint is None:
            checkpoint = str(checkpoint_path)
        
        model = loader.load_by(checkpoint)
        print("✓ Model loaded using laia ModelLoader")
        
    except Exception as e:
        print(f"  ModelLoader failed: {e}, trying fallback...")
        import pytorch_lightning as pl
        model = pl.LightningModule.load_from_checkpoint(
            str(checkpoint_path), map_location='cpu', strict=False
        )
        print("✓ Model loaded using PyTorch Lightning")
    
    model.eval()
    
    # Extract actual model from Lightning wrapper
    for attr in ['model', 'net', 'crnn']:
        if hasattr(model, attr):
            actual_model = getattr(model, attr)
            actual_model.eval()
            return actual_model, syms
    
    return model, syms


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Preprocess image for Pylaia model (grayscale, resize to 128px height)."""
    img = Image.open(image_path).convert('L')
    
    width, height = img.size
    new_height = 128
    new_width = int(width * (new_height / height))
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)


def text_to_indices(text: str, syms: SymbolsTable) -> Optional[List[int]]:
    """
    Convert text to symbol indices, handling common variations.
    Returns None if any character cannot be mapped.
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
    
    Returns:
        Log probabilities tensor of shape (Time, Batch, Classes)
    """
    with torch.no_grad():
        output = model(image_tensor)
    
    # Ensure (Time, Batch, Classes) format
    if output.dim() == 3 and output.size(0) == image_tensor.size(0):
        output = output.transpose(0, 1)
    
    return F.log_softmax(output, dim=2)


def calculate_ctc_loss_from_logprobs(
    log_probs: torch.Tensor, 
    syms: SymbolsTable, 
    text: str
) -> float:
    """
    Calculate CTC loss using pre-computed log probabilities.
    
    This is the fast path - use when you have cached log_probs from get_model_log_probs().
    
    Returns:
        Raw CTC loss (negative log-likelihood), or float('inf') if impossible.
    """
    # Get indices for text
    indices = text_to_indices(text, syms)
    if indices is None:
        return float('inf')
    
    # Check feasibility (CTC requires T >= S)
    if log_probs.size(0) < len(indices):
        return float('inf')
    
    # Calculate CTC loss
    target = torch.tensor(indices, dtype=torch.long)
    input_lengths = torch.tensor([log_probs.size(0)], dtype=torch.long)
    target_lengths = torch.tensor([len(indices)], dtype=torch.long)
    
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=False)
    
    try:
        loss = ctc_loss_fn(log_probs, target, input_lengths, target_lengths)
        loss_value = loss.item()
        
        if not np.isfinite(loss_value):
            return float('inf')
        
        return loss_value
    except RuntimeError:
        return float('inf')


def calculate_ctc_loss(model, syms: SymbolsTable, image_tensor: torch.Tensor, text: str) -> float:
    """
    Calculate CTC loss for text given image.
    
    NOTE: This runs a full forward pass. For multiple candidates on the same image,
    use get_model_log_probs() once, then calculate_ctc_loss_from_logprobs() for each.
    
    Returns:
        Raw CTC loss (negative log-likelihood), or float('inf') if impossible.
    """
    log_probs = get_model_log_probs(model, image_tensor)
    return calculate_ctc_loss_from_logprobs(log_probs, syms, text)


# =============================================================================
# DATABASE QUERY CLASS
# =============================================================================

class DatabaseQuery:
    """Helper class for querying cp40_records.db with SymSpell fuzzy matching."""
    
    def __init__(self, db_path: Path, config: BayesianConfig):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.config = config
        self.db_path = db_path
        
        self._load_data()
        self._create_symspell_dicts()
    
    def _load_data(self):
        """Load all names with frequencies from database."""
        # Forenames and Latin forms
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='forenames'"
        )
        if cursor.fetchone():
            cursor = self.conn.execute("SELECT id, english_name, frequency FROM forenames")
            self.forenames = {
                row['english_name']: {'id': row['id'], 'frequency': row['frequency']}
                for row in cursor.fetchall()
            }
            
            cursor = self.conn.execute("""
                SELECT fl.latin_abbreviated, fl.case_name, f.english_name, f.frequency
                FROM forename_latin_forms fl
                JOIN forenames f ON fl.forename_id = f.id
            """)
            self.forename_latin_forms = defaultdict(list)
            for row in cursor.fetchall():
                self.forename_latin_forms[row['latin_abbreviated']].append({
                    'english_name': row['english_name'],
                    'case_name': row['case_name'],
                    'frequency': row['frequency']
                })
        else:
            self.forenames = {}
            self.forename_latin_forms = {}
        
        # Surnames
        cursor = self.conn.execute("""
            SELECT s.surname, COUNT(ps.person_id) as frequency
            FROM surnames s
            LEFT JOIN person_surnames ps ON s.id = ps.surname_id
            GROUP BY s.surname
        """)
        self.surnames = {row['surname']: row['frequency'] for row in cursor.fetchall()}
        
        # Placenames
        cursor = self.conn.execute("""
            SELECT p.name, COUNT(ep.entry_id) as frequency
            FROM places p
            LEFT JOIN entry_places ep ON p.id = ep.place_id
            GROUP BY p.name
        """)
        self.placenames = {row['name']: row['frequency'] for row in cursor.fetchall()}
        
        # Calculate totals for prior computation
        self.total_forename_freq = sum(info['frequency'] for info in self.forenames.values()) or 1
        self.total_surname_freq = sum(self.surnames.values()) or 1
        self.total_placename_freq = sum(self.placenames.values()) or 1
    
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
                    print(f"Cache outdated (DB modified), rebuilding dictionaries")
                    return None
            
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            
            # Validate cache structure
            required_keys = ['forename_symspell', 'surname_symspell', 'placename_symspell']
            if all(key in cached for key in required_keys):
                print(f"Loaded SymSpell dictionaries from cache: {cache_path.name}")
                return cached
            else:
                print("Cache file has invalid structure, rebuilding")
                return None
        except Exception as e:
            print(f"Failed to load cache: {e}, rebuilding dictionaries")
            return None
    
    def _save_symspell_cache(self, cache_path: Path, dicts: Dict[str, SymSpell]) -> None:
        """Save SymSpell dictionaries to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(dicts, f)
            print(f"Saved SymSpell dictionaries to cache: {cache_path.name}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _create_symspell_dicts(self):
        """Create SymSpell dictionaries for fast fuzzy lookup, using cache if available."""
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        cached_dicts = self._load_symspell_cache(cache_path)
        if cached_dicts:
            self.forename_symspell = cached_dicts['forename_symspell']
            self.surname_symspell = cached_dicts['surname_symspell']
            self.placename_symspell = cached_dicts['placename_symspell']
            return
        
        # Cache miss or invalid - create dictionaries
        max_dist = self.config.max_edit_distance
        
        # Forename Latin forms
        self.forename_symspell = SymSpell(max_dictionary_edit_distance=max_dist)
        for latin_form, forms in self.forename_latin_forms.items():
            max_freq = max((f['frequency'] for f in forms), default=1)
            self.forename_symspell.create_dictionary_entry(latin_form, max_freq)
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
        
        # Save to cache
        dicts_to_cache = {
            'forename_symspell': self.forename_symspell,
            'surname_symspell': self.surname_symspell,
            'placename_symspell': self.placename_symspell
        }
        self._save_symspell_cache(cache_path, dicts_to_cache)
    
    def _get_effective_distance(self, text_len: int) -> int:
        """
        Get effective edit distance based on text length.
        Short names get stricter matching to avoid spurious matches.
        """
        if text_len <= 3:
            return 0  # Exact match only
        elif text_len <= 5:
            return 1
        else:
            return self.config.max_edit_distance
    
    def _get_latin_stems(self, latin_text: str) -> List[str]:
        """
        Generate possible stems by stripping common Latin case endings.
        This helps match abbreviated forms (Rob'm) with full forms (Rob'tum).
        
        Latin case endings:
        - Nominative: -us, -um, -a, -is, -es
        - Genitive: -i, -ae, -is, -ium
        - Dative: -o, -ae, -i, -ibus
        - Accusative: -um, -am, -em, -os, -as, -es
        - Ablative: -o, -a, -e, -ibus
        """
        stems = [latin_text]  # Always include original
        
        # Common Latin endings to strip (ordered by length, longest first)
        endings = [
            'ibus', 'orum', 'arum',  # Plural endings
            'tum', 'tus', 'tae', 'tam', 'tis', 'dum', 'dus',  # Full endings
            'um', 'us', 'ae', 'am', 'os', 'as', 'es', 'is',  # 2-char endings
            'o', 'a', 'e', 'i', 'm', 's',  # 1-char endings
        ]
        
        for ending in endings:
            if latin_text.endswith(ending) and len(latin_text) > len(ending) + 2:
                stem = latin_text[:-len(ending)]
                if stem not in stems:
                    stems.append(stem)
                # Also try with apostrophe (e.g., "Rob'" from "Rob'tum")
                if "'" in stem:
                    stems.append(stem)
        
        return stems
    
    def _passes_length_filter(self, original_len: int, candidate_len: int) -> bool:
        """Check if candidate passes length ratio filter."""
        if original_len == 0:
            return False
        ratio = candidate_len / original_len
        return self.config.min_length_ratio <= ratio <= self.config.max_length_ratio
    
    def find_forename_candidates(self, text: str, declension: Optional[str] = None) -> List[Dict]:
        """
        Find forename candidates using SymSpell with dynamic distance.
        
        Uses stem-based lookup to handle Latin ending variations:
        e.g., "Rob'tum" should find "Rob'm" (both are accusative forms of Robert)
        """
        candidates = []
        effective_dist = self._get_effective_distance(len(text))
        
        # Exact match in Latin forms
        if text in self.forename_latin_forms:
            for form in self.forename_latin_forms[text]:
                if declension is None or form['case_name'] == declension:
                    candidates.append({
                        'text': text,
                        'english_name': form['english_name'],
                        'declension': form['case_name'],
                        'distance': 0,
                        'frequency': form['frequency']
                    })
        
        # SymSpell lookup on original text
        suggestions = self.forename_symspell.lookup(
            text, verbosity=Verbosity.ALL,
            max_edit_distance=effective_dist, include_unknown=True
        )
        
        for suggestion in suggestions:
            if not self._passes_length_filter(len(text), len(suggestion.term)):
                continue
            
            latin_form = suggestion.term
            if latin_form in self.forename_latin_forms:
                for form in self.forename_latin_forms[latin_form]:
                    if declension is None or form['case_name'] == declension:
                        candidates.append({
                            'text': latin_form,
                            'english_name': form['english_name'],
                            'declension': form['case_name'],
                            'distance': suggestion.distance,
                            'frequency': form['frequency']
                        })
            elif latin_form in self.forenames:
                candidates.append({
                    'text': latin_form,
                    'english_name': latin_form,
                    'declension': None,
                    'distance': suggestion.distance,
                    'frequency': self.forenames[latin_form]['frequency']
                })
        
        # STEM-BASED LOOKUP: Handle Latin ending variations
        # e.g., "Rob'tum" -> stem "Rob'" -> find "Rob'm", "Rob't", etc.
        stems = self._get_latin_stems(text)
        for stem in stems[1:]:  # Skip first (original text, already searched)
            # Search for Latin forms that start with this stem
            stem_suggestions = self.forename_symspell.lookup(
                stem, verbosity=Verbosity.ALL,
                max_edit_distance=1,  # Lower distance for stems
                include_unknown=True
            )
            
            for suggestion in stem_suggestions:
                latin_form = suggestion.term
                # Check if this Latin form shares the same stem
                if not latin_form.startswith(stem.rstrip("'")):
                    continue
                
                if latin_form in self.forename_latin_forms:
                    for form in self.forename_latin_forms[latin_form]:
                        if declension is None or form['case_name'] == declension:
                            # Use the original text's edit distance, not stem's
                            actual_distance = self._levenshtein(text, latin_form)
                            if actual_distance <= self.config.max_edit_distance:
                                candidates.append({
                                    'text': latin_form,
                                    'english_name': form['english_name'],
                                    'declension': form['case_name'],
                                    'distance': actual_distance,
                                    'frequency': form['frequency']
                                })
        
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            key = (c['text'], c.get('english_name'), c.get('declension'))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        return sorted(unique, key=lambda x: (x['distance'], -x['frequency']))
    
    def _levenshtein(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
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
    
    def find_surname_candidates(self, text: str) -> List[Dict]:
        """Find surname candidates using SymSpell with dynamic distance."""
        candidates = []
        effective_dist = self._get_effective_distance(len(text))
        
        # Exact match
        if text in self.surnames:
            candidates.append({
                'text': text,
                'distance': 0,
                'frequency': self.surnames[text]
            })
        
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
        
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c['text'] not in seen:
                seen.add(c['text'])
                unique.append(c)
        
        return sorted(unique, key=lambda x: (x['distance'], -x['frequency']))
    
    def find_placename_candidates(self, text: str) -> List[Dict]:
        """Find placename candidates using SymSpell with dynamic distance."""
        candidates = []
        effective_dist = self._get_effective_distance(len(text))
        
        # Exact match
        if text in self.placenames:
            candidates.append({
                'text': text,
                'distance': 0,
                'frequency': self.placenames[text]
            })
        
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
        
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c['text'] not in seen:
                seen.add(c['text'])
                unique.append(c)
        
        return sorted(unique, key=lambda x: (x['distance'], -x['frequency']))
    
    def close(self):
        self.conn.close()


# =============================================================================
# BAYESIAN CANDIDATE SELECTION
# =============================================================================

@dataclass
class ScoredCandidate:
    """A candidate with its Bayesian score components."""
    text: str
    frequency: int
    distance: int
    english_name: Optional[str] = None
    declension: Optional[str] = None
    
    # Scores
    raw_ctc_loss: float = float('inf')
    normalized_log_likelihood: float = float('-inf')
    log_prior: float = float('-inf')
    total_score: float = float('-inf')
    
    is_original: bool = False


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
    
    def __init__(self, model, syms: SymbolsTable, config: BayesianConfig):
        self.model = model
        self.syms = syms
        self.config = config
    
    def compute_log_prior(self, frequency: int, total_frequency: int, vocab_size: int) -> float:
        """
        Compute Laplace-smoothed log prior probability.
        
        P(candidate) = (freq + α) / (total + α × V)
        """
        alpha = self.config.smoothing_alpha
        numerator = frequency + alpha
        denominator = total_frequency + alpha * vocab_size
        return log(numerator / denominator)
    
    def compute_log_likelihood(self, log_probs: torch.Tensor, text: str, context: str,
                                entity_position: Optional[Tuple[int, int]],
                                original_length: int) -> Tuple[float, float]:
        """
        Compute normalized log-likelihood from CTC loss using cached log_probs.
        
        Strategy: Calculate CTC loss for the full line with entity inserted,
        then normalize by ORIGINAL entity length for fair comparison.
        
        Args:
            log_probs: Pre-computed log probabilities from get_model_log_probs()
            text: The candidate text to score
            context: The full line context
            entity_position: (start, end) position of entity in context
            original_length: Length of the original extracted text (used for normalization)
        
        Returns:
            (raw_loss, normalized_log_likelihood)
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
        
        # Normalize by ORIGINAL entity length (not candidate length)
        # This ensures fair comparison: all candidates use the same denominator
        if original_length > 0:
            # -loss gives log probability; normalize by original length
            normalized = -raw_loss / original_length
        else:
            normalized = float('-inf')
        
        return raw_loss, normalized
    
    def select_best(
        self,
        candidates: List[Dict],
        original_text: str,
        image_tensor: torch.Tensor,
        log_probs: torch.Tensor,
        context: str,  # Raw HTR text
        entity_position: Optional[Tuple[int, int]],
        total_frequency: int,
        vocab_size: int,
        entity_type: str = "entity"
    ) -> Tuple[Optional[ScoredCandidate], List[ScoredCandidate]]:
        """
        Select the best candidate using Bayesian scoring.
        
        Args:
            candidates: List of candidate dicts from database query
            original_text: The original extracted text (from Gemini)
            image_tensor: Pre-loaded image tensor
            log_probs: Pre-computed log probabilities from model forward pass
            context: Raw HTR text (used as context for CTC)
            entity_position: (start, end) position in context
            total_frequency: Total frequency for prior computation
            vocab_size: Vocabulary size for Laplace smoothing
            entity_type: "forename", "surname", or "placename"
        
        Returns:
            (best_candidate, all_scored_candidates)
        """
        if not candidates:
            return None, []
        
        # Ensure original is in candidates
        original_in_list = any(c['text'] == original_text for c in candidates)
        if not original_in_list:
            # Add original with frequency 0 (will get smoothing bonus)
            candidates = [{'text': original_text, 'frequency': 0, 'distance': 0}] + candidates
        
        # Score all candidates (using cached log_probs - no repeated forward passes!)
        scored = []
        best_score = float('-inf')
        best_candidate = None
        original_length = len(original_text)  # Used for fair normalization
        
        for cand in candidates:
            text = cand['text']
            freq = cand.get('frequency', 0)
            
            # Compute log prior
            log_prior = self.compute_log_prior(freq, total_frequency, vocab_size)
            
            # Compute log likelihood using cached log_probs
            # Note: normalize by original_length so all candidates use same denominator
            raw_loss, norm_likelihood = self.compute_log_likelihood(
                log_probs, text, context, entity_position, original_length
            )
            
            # Combined score
            score = (
                self.config.likelihood_weight * norm_likelihood +
                self.config.prior_weight * log_prior
            )
            
            # Bonus for original extraction
            is_original = (text == original_text)
            if is_original:
                score += self.config.original_bonus
            
            # Penalty for edit distance from original
            # This favors candidates closer to what was actually written
            distance = cand.get('distance', 0)
            score -= distance * self.config.distance_penalty
            
            scored_cand = ScoredCandidate(
                text=text,
                frequency=freq,
                distance=cand.get('distance', 0),
                english_name=cand.get('english_name'),
                declension=cand.get('declension'),
                raw_ctc_loss=raw_loss,
                normalized_log_likelihood=norm_likelihood,
                log_prior=log_prior,
                total_score=score,
                is_original=is_original
            )
            scored.append(scored_cand)
            
            if score > best_score:
                best_score = score
                best_candidate = scored_cand
        
        return best_candidate, sorted(scored, key=lambda x: -x.total_score)


# =============================================================================
# GEMINI API
# =============================================================================

def reset_gemini_client():
    """Reset the cached Gemini client to force trying new keys on next call."""
    global _current_api_key, _current_client
    _current_api_key = None
    _current_client = None


def call_gemini_for_extraction_batch(client, all_lines_data: List[List[Dict]], batch_id: str = None) -> List[List[Dict]]:
    """
    Call Gemini 3.0 Flash in batch mode to correct HTR text and extract entities.
    
    Args:
        client: Gemini client
        all_lines_data: List of lists, where each inner list contains line data dicts for one image
        batch_id: Optional batch identifier for state management
    
    Returns:
        List of lists, where each inner list contains results for one image
    """
    import tempfile
    from pathlib import Path
    
    if not all_lines_data:
        return []
    
    # Create temporary directory for batch files
    temp_dir = Path(tempfile.mkdtemp())
    jsonl_file = temp_dir / "batch_requests.jsonl"
    
    # Prepare batch requests in JSONL format
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for idx, lines_data in enumerate(all_lines_data):
            lines_json = json.dumps(lines_data, indent=2, ensure_ascii=False)
            
            prompt = f"""You are an expert 15th-century English Court of Common Pleas paleographer.

Perform a **Diplomatic Transcription** of the HTR text below.

## Input Lines:
{lines_json}

## Tasks for EACH line:

1. **Diplomatic Correction**: Correct clear OCR/HTR errors only.
   - PRESERVE all scribal abbreviations exactly (e.g., keep "Rob'tus", NOT "Robertus")
   - PRESERVE apostrophes marking abbreviations (use straight apostrophe ')
   - DO NOT modernize or standardize spellings

2. **Extract Entities** (copy EXACTLY from corrected text):
   - **Forenames**: Latin abbreviated names (e.g., "Joh'es", "Ric'us"). Include declension.
   - **Surnames**: Family names (e.g., "Clanyng", "Ogle")
   - **Placenames**: Place names (e.g., "London'", "Holborn'")

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

Return ONLY valid JSON, no other text."""

            request_obj = {
                "custom_id": f"image_{idx}",
                "params": {
                    "model": "gemini-3.0-flash",
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": {
                        "temperature": 0.2,
                        "response_mime_type": "application/json"
                    }
                }
            }
            f.write(json.dumps(request_obj) + "\n")
    
    try:
        # Upload JSONL file
        print(f"  Uploading batch file with {len(all_lines_data)} requests...")
        uploaded_file = client.files.upload(path=str(jsonl_file))
        
        # Create batch job
        print(f"  Creating batch job...")
        batch_job = client.batches.create(
            model="gemini-3.0-flash",
            src=uploaded_file.name,
            config={"display_name": batch_id or f"htr_extraction_{int(time.time())}"}
        )
        
        print(f"  Batch job created: {batch_job.name}")
        print(f"  Status: {batch_job.state}")
        
        # Poll for completion
        while batch_job.state in ["PENDING", "RUNNING"]:
            time.sleep(30)
            batch_job = client.batches.get(name=batch_job.name)
            print(f"  Batch status: {batch_job.state}")
        
        if batch_job.state != "SUCCEEDED":
            print(f"  Batch job failed with state: {batch_job.state}")
            # Fallback: return empty extractions
            return [[{"key": line["key"], "corrected_text": line["htr_text"], 
                     "forenames": [], "surnames": [], "placenames": []} 
                    for line in lines_data] for lines_data in all_lines_data]
        
        # Download results
        print(f"  Downloading results...")
        results_file = temp_dir / "batch_results.jsonl"
        with open(results_file, 'wb') as f:
            for chunk in batch_job.download():
                f.write(chunk)
        
        # Parse results
        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                result_obj = json.loads(line)
                if result_obj.get("status") == "SUCCEEDED":
                    response_text = result_obj.get("response", {}).get("text", "")
                    try:
                        parsed = json.loads(response_text)
                        results.append(parsed if isinstance(parsed, list) else [parsed])
                    except json.JSONDecodeError:
                        print(f"    Warning: Failed to parse JSON for {result_obj.get('custom_id')}")
                        results.append([])
                else:
                    print(f"    Warning: Request {result_obj.get('custom_id')} failed: {result_obj.get('status')}")
                    results.append([])
        
        # Ensure we have results for all requests
        while len(results) < len(all_lines_data):
            results.append([])
        
        return results[:len(all_lines_data)]
        
    except Exception as e:
        print(f"    Gemini batch API error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return empty extractions
        return [[{"key": line["key"], "corrected_text": line["htr_text"], 
                 "forenames": [], "surnames": [], "placenames": []} 
                for line in lines_data] for lines_data in all_lines_data]
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


def call_gemini_for_extraction(client, lines_data: List[Dict], retry_on_error: bool = True) -> List[Dict]:
    """
    Legacy function for single-image processing. 
    For batch processing, use call_gemini_for_extraction_batch instead.
    """
    # For single requests, we can still use the regular API
    lines_json = json.dumps(lines_data, indent=2, ensure_ascii=False)
    
    prompt = f"""You are an expert 15th-century English Court of Common Pleas paleographer.

Perform a **Diplomatic Transcription** of the HTR text below.

## Input Lines:
{lines_json}

## Tasks for EACH line:

1. **Diplomatic Correction**: Correct clear OCR/HTR errors only.
   - PRESERVE all scribal abbreviations exactly (e.g., keep "Rob'tus", NOT "Robertus")
   - PRESERVE apostrophes marking abbreviations (use straight apostrophe ')
   - DO NOT modernize or standardize spellings

2. **Extract Entities** (copy EXACTLY from corrected text):
   - **Forenames**: Latin abbreviated names (e.g., "Joh'es", "Ric'us"). Include declension.
   - **Surnames**: Family names (e.g., "Clanyng", "Ogle")
   - **Placenames**: Place names (e.g., "London'", "Holborn'")

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

Return ONLY valid JSON, no other text."""

    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json"
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-3.0-flash",
            contents=[types.Part.from_text(text=prompt)],
            config=config,
        )
        
        if response and hasattr(response, 'text') and response.text:
            result = json.loads(response.text)
            return result if isinstance(result, list) else [result]
    except Exception as e:
        print(f"    Gemini API error: {e}")
    
    # Fallback: return empty extractions
    return [{"key": line["key"], "corrected_text": line["htr_text"], 
             "forenames": [], "surnames": [], "placenames": []} 
            for line in lines_data]


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_entity(
    entity_text: str,
    entity_type: str,
    candidates: List[Dict],
    selector: BayesianSelector,
    image_tensor: torch.Tensor,
    log_probs: torch.Tensor,
    raw_htr_text: str,  # Use raw HTR as context
    total_freq: int,
    vocab_size: int,
    declension: Optional[str] = None
) -> Dict:
    """
    Process a single entity (forename, surname, or placename).
    
    Args:
        entity_text: The extracted entity text
        entity_type: "forename", "surname", or "placename"
        candidates: List of candidate dicts from database
        selector: BayesianSelector instance
        image_tensor: Pre-loaded and preprocessed image tensor
        log_probs: Pre-computed log probabilities (forward pass done once per line)
        raw_htr_text: Raw HTR text as context
        total_freq: Total frequency for prior computation
        vocab_size: Vocabulary size
        declension: Optional declension for forenames
    
    Returns a result dict with candidates and best selection.
    """
    # Find entity position in raw HTR using fuzzy matching
    position = find_entity_in_text(entity_text, raw_htr_text)
    
    # Run Bayesian selection (using cached log_probs - fast!)
    best, scored = selector.select_best(
        candidates=candidates,
        original_text=entity_text,
        image_tensor=image_tensor,
        log_probs=log_probs,
        context=raw_htr_text,
        entity_position=position,
        total_frequency=total_freq,
        vocab_size=vocab_size,
        entity_type=entity_type
    )
    
    # Build result
    result = {
        'original': entity_text,
        'position_in_htr': list(position) if position else None,
        'candidates': [
            {
                'text': s.text,
                'frequency': s.frequency,
                'distance': s.distance,
                'english_name': s.english_name,
                'declension': s.declension,
                'raw_ctc_loss': s.raw_ctc_loss if s.raw_ctc_loss != float('inf') else None,
                'normalized_log_likelihood': s.normalized_log_likelihood if s.normalized_log_likelihood != float('-inf') else None,
                'log_prior': s.log_prior,
                'total_score': s.total_score,
                'is_original': s.is_original
            }
            for s in scored[:10]  # Top 10 candidates
        ]
    }
    
    if declension:
        result['declension'] = declension
    
    if best:
        result['best_candidate'] = {
            'text': best.text,
            'english_name': best.english_name,
            'declension': best.declension,
            'frequency': best.frequency,
            'distance': best.distance,
            'normalized_log_likelihood': best.normalized_log_likelihood if best.normalized_log_likelihood != float('-inf') else None,
            'log_prior': best.log_prior,
            'total_score': best.total_score
        }
    
    return result


def process_subfolder(
    subfolder_path: Path,
    model,
    syms: SymbolsTable,
    db_query: DatabaseQuery,
    config: BayesianConfig
) -> Dict[str, Any]:
    """Process a single HTR work subfolder."""
    print(f"\nProcessing: {subfolder_path.name}")
    
    htr_file = subfolder_path / "htr.txt"
    kraken_file = subfolder_path / "kraken.json"
    
    if not htr_file.exists() or not kraken_file.exists():
        print(f"  Warning: Missing htr.txt or kraken.json")
        return {}
    
    # Parse htr.txt
    lines_data = []
    with open(htr_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ']' not in line:
                continue
            
            idx = line.index(']')
            if '[' not in line[:idx]:
                continue
            
            path_end = line.index('[')
            line_path = line[:path_end].strip()
            htr_text_raw = line[idx+1:].strip()
            
            lines_data.append({
                'line_id': Path(line_path).stem,
                'line_path': line_path,
                'htr_text_raw': htr_text_raw,
                'htr_text_cleaned': clean_htr_text(htr_text_raw)
            })
    
    # Load kraken.json for bounding boxes
    with open(kraken_file, 'r', encoding='utf-8') as f:
        kraken_data = json.load(f)
    
    line_bbox_map = {}
    for line_info in kraken_data.get('lines', []):
        line_id = line_info.get('id')
        boundary = line_info.get('boundary', [])
        if line_id and boundary:
            xs = [p[0] for p in boundary]
            ys = [p[1] for p in boundary]
            if xs and ys:
                line_bbox_map[line_id] = [min(ys), min(xs), max(ys), max(xs)]
    
    # Prepare Gemini input
    gemini_input = [
        {'key': f"L{i+1:02d}", 'bbox': line_bbox_map.get(ld['line_id']), 'htr_text': ld['htr_text_cleaned']}
        for i, ld in enumerate(lines_data)
    ]
    
    print(f"  Calling Gemini 3.0 Flash for {len(gemini_input)} lines...")
    # Use batch mode for better efficiency
    client = get_gemini_client()
    # For single subfolder, wrap in list for batch API
    gemini_results_list = call_gemini_for_extraction_batch(client, [gemini_input], batch_id=f"subfolder_{subfolder_path.name}")
    gemini_results = gemini_results_list[0] if gemini_results_list else []
    gemini_map = {r.get('key'): r for r in gemini_results}
    
    # Create selector
    selector = BayesianSelector(model, syms, config)
    
    # Process each line
    results = {'subfolder': subfolder_path.name, 'config': vars(config), 'lines': []}
    
    for i, line_data in enumerate(lines_data):
        line_id = line_data['line_id']
        raw_htr = line_data['htr_text_cleaned']  # Use cleaned raw HTR as context
        key = f"L{i+1:02d}"
        
        gemini_result = gemini_map.get(key, {
            'corrected_text': raw_htr, 'forenames': [], 'surnames': [], 'placenames': []
        })
        
        # Find line image
        image_path = None
        if line_data.get('line_path'):
            potential = Path(line_data['line_path'])
            if not potential.exists():
                # Try WSL path conversion
                path_str = str(potential)
                if path_str.startswith('/home/qj/'):
                    wsl_path = path_str.replace('/home/qj/', '\\\\wsl.localhost\\Ubuntu\\home\\qj\\').replace('/', '\\')
                    potential = Path(wsl_path)
            if potential.exists():
                image_path = potential
        
        if not image_path or not image_path.exists():
            image_path = subfolder_path / "lines" / f"{line_id}.png"
        if not image_path.exists():
            image_path = subfolder_path / f"{line_id}.png"
        
        line_result = {
            'line_id': line_id,
            'htr_text': raw_htr,
            'corrected_text': gemini_result.get('corrected_text', raw_htr),
            'bbox': line_bbox_map.get(line_id),
            'forenames': [],
            'surnames': [],
            'placenames': []
        }
        
        if image_path.exists():
            # Load image ONCE per line and compute log_probs ONCE
            # This is the key optimization - forward pass is expensive!
            try:
                image_tensor = preprocess_image(image_path)
                log_probs = get_model_log_probs(model, image_tensor)
            except Exception as e:
                print(f"    Warning: Could not load/process image {image_path}: {e}")
                image_tensor = None
                log_probs = None
            
            if image_tensor is not None and log_probs is not None:
                # Process forenames
                for fn in gemini_result.get('forenames', []):
                    if fn.get('text'):
                        candidates = db_query.find_forename_candidates(fn['text'], fn.get('declension'))
                        result = process_entity(
                            entity_text=fn['text'],
                            entity_type='forename',
                            candidates=candidates,
                            selector=selector,
                            image_tensor=image_tensor,
                            log_probs=log_probs,
                            raw_htr_text=raw_htr,
                            total_freq=db_query.total_forename_freq,
                            vocab_size=len(db_query.forenames),
                            declension=fn.get('declension')
                        )
                        line_result['forenames'].append(result)
                
                # Process surnames
                for sn in gemini_result.get('surnames', []):
                    if sn.get('text'):
                        candidates = db_query.find_surname_candidates(sn['text'])
                        result = process_entity(
                            entity_text=sn['text'],
                            entity_type='surname',
                            candidates=candidates,
                            selector=selector,
                            image_tensor=image_tensor,
                            log_probs=log_probs,
                            raw_htr_text=raw_htr,
                            total_freq=db_query.total_surname_freq,
                            vocab_size=len(db_query.surnames)
                        )
                        line_result['surnames'].append(result)
                
                # Process placenames
                for pn in gemini_result.get('placenames', []):
                    if pn.get('text'):
                        candidates = db_query.find_placename_candidates(pn['text'])
                        result = process_entity(
                            entity_text=pn['text'],
                            entity_type='placename',
                            candidates=candidates,
                            selector=selector,
                            image_tensor=image_tensor,
                            log_probs=log_probs,
                            raw_htr_text=raw_htr,
                            total_freq=db_query.total_placename_freq,
                            vocab_size=len(db_query.placenames)
                        )
                        line_result['placenames'].append(result)
        
        results['lines'].append(line_result)
    
    # Save results
    output_file = subfolder_path / "name_extraction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to {output_file}")
    time.sleep(1.0)  # Rate limiting
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian name extraction from HTR text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--subfolder", type=str, help="Process specific subfolder only")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH), help="Path to database")
    
    # Bayesian parameters
    parser.add_argument("--likelihood-weight", type=float, default=1.0,
                        help="Weight for log-likelihood (visual evidence)")
    parser.add_argument("--prior-weight", type=float, default=1.0,
                        help="Weight for log-prior (frequency)")
    parser.add_argument("--smoothing-alpha", type=float, default=1.0,
                        help="Laplace smoothing parameter")
    parser.add_argument("--original-bonus", type=float, default=0.5,
                        help="Bonus for original extraction (in log space)")
    parser.add_argument("--distance-penalty", type=float, default=1.5,
                        help="Penalty per edit distance from original (in log space)")
    parser.add_argument("--max-edit-distance", type=int, default=2,
                        help="Max edit distance for candidate search")
    
    args = parser.parse_args()
    
    config = BayesianConfig(
        likelihood_weight=args.likelihood_weight,
        prior_weight=args.prior_weight,
        smoothing_alpha=args.smoothing_alpha,
        original_bonus=args.original_bonus,
        distance_penalty=args.distance_penalty,
        max_edit_distance=args.max_edit_distance
    )
    
    print("=" * 70)
    print("BAYESIAN NAME EXTRACTION FROM HTR")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Likelihood weight: {config.likelihood_weight}")
    print(f"  Prior weight: {config.prior_weight}")
    print(f"  Smoothing α: {config.smoothing_alpha}")
    print(f"  Original bonus: {config.original_bonus}")
    print(f"  Distance penalty: {config.distance_penalty}")
    print(f"  Max edit distance: {config.max_edit_distance}")
    
    # Load model
    print("\nLoading Pylaia model...")
    checkpoint, model_arch, syms_path = find_latest_pylaia_model()
    model, syms = load_pylaia_model(checkpoint, model_arch, syms_path)
    
    # Load database
    print(f"\nLoading database: {args.db_path}")
    db_query = DatabaseQuery(Path(args.db_path), config)
    print(f"  Forenames: {len(db_query.forenames)}")
    print(f"  Surnames: {len(db_query.surnames)}")
    print(f"  Placenames: {len(db_query.placenames)}")
    
    # Initialize Gemini 3.0 Flash with paid key (required for batch mode)
    print("\nInitializing Gemini 3.0 Flash client...")
    gemini_client = get_gemini_client()
    
    # Process
    subfolders = [HTR_WORK_DIR / args.subfolder] if args.subfolder else [
        d for d in HTR_WORK_DIR.iterdir() if d.is_dir()
    ]
    
    print(f"\nProcessing {len(subfolders)} subfolder(s)...")
    
    for subfolder in subfolders:
        try:
            process_subfolder(subfolder, model, syms, db_query, config)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    db_query.close()
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
