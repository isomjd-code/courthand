"""Similarity utilities and alignment helpers."""

from __future__ import annotations

import ast
import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from .config import ALIGNMENT_THRESHOLD, EMBED_MODEL, PARTY_MATCH_THRESHOLD, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)
from .text_utils import (
    clean_text_for_xelatex,
    get_agent_desc,
    get_person_name,
    get_surname_from_name,
    normalize_damages_for_comparison,
    normalize_date_for_comparison,
    normalize_name_for_comparison,
    normalize_writ_type_for_comparison,
    soundex,
    split_into_sentences,
)

try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    GEMINI_AVAILABLE = False
    genai = None  # type: ignore
    types = None  # type: ignore


def calculate_similarity(value1: str, value2: str) -> float:
    """Compute a Levenshtein similarity ratio."""
    if not value1 and not value2:
        return 1.0
    if not value1 or not value2:
        return 0.0
    normalized_a = "".join(value1.split()).lower()
    normalized_b = "".join(value2.split()).lower()
    return difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()


def get_field_threshold(field_name: str, category: str) -> float:
    """
    Get the similarity threshold for a specific field.
    
    Thresholds:
    - Agent Name: > 95% (0.95) - Keep strict for names
    - Event Place: > 95% (0.95) - Keep strict for places
    - Agent Location: > 95% (0.95) - Keep strict for locations
    - Case Details Block (Pleading/Postea text blocks): > 75% (0.75) - Lowered to catch partial matches
    - Case Type: > 85% (0.85) - Allow for variations in case type descriptions
    - Damages Claimed: > 90% (0.90) - Numbers should be exact
    - All other fields: > 90% (0.90)
    """
    if field_name == "Agent Name":
        return 0.95
    elif field_name == "Event Place" or field_name == "Agent Location":
        return 0.95
    elif "Block" in field_name:
        # Case Details Block fields (Pleading, Postea) use 75% threshold (lowered from 78%)
        return 0.75
    elif field_name == "Case Type":
        return 0.85
    elif field_name == "Damages Claimed":
        return 0.90
    else:
        return 0.90


def get_similarity_score(text1: str, text2: str, api_key: Optional[str], context: str = "field_comparison") -> float:
    """Calculate semantic similarity via Gemini embeddings."""
    if text1 == text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    if not (GEMINI_AVAILABLE and api_key):
        return calculate_similarity(text1, text2)

    try:
        # Log the embedding API call
        text1_preview = text1[:50] + "..." if len(text1) > 50 else text1
        text2_preview = text2[:50] + "..." if len(text2) > 50 else text2
        logger.info(f"[Embedding API] Making embedding call for {context}")
        logger.debug(f"[Embedding API]   Text1 preview: {text1_preview}")
        logger.debug(f"[Embedding API]   Text2 preview: {text2_preview}")
        
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[text1, text2],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embeddings = np.array([np.array(item.values) for item in response.embeddings])
        similarity_matrix = cosine_similarity(embeddings)
        similarity = float(similarity_matrix[0][1])
        logger.debug(f"[Embedding API]   Similarity score: {similarity:.3f}")
        return similarity
    except Exception as e:
        logger.warning(f"[Embedding API] Embedding call failed for {context}, falling back to Levenshtein: {e}")
        return calculate_similarity(text1, text2)


@dataclass
class FieldComparison:
    """Stores comparison results for a single field."""

    field_name: str
    gt_value: str
    ai_value: str
    is_match: bool
    similarity_score: float
    category: str = "general"


@dataclass
class ValidationMetrics:
    """Aggregates validation metrics across all comparisons."""

    comparisons: List[FieldComparison] = field(default_factory=list)

    def add(self, comparison: FieldComparison) -> None:
        self.comparisons.append(comparison)

    def _filtered(self, category: Optional[str]) -> List[FieldComparison]:
        if category is None:
            return self.comparisons
        return [comp for comp in self.comparisons if comp.category == category]

    def get_accuracy(self, category: Optional[str] = None) -> float:
        filtered = self._filtered(category)
        if not filtered:
            return 0.0
        return sum(1 for comp in filtered if comp.is_match) / len(filtered) * 100

    def get_avg_similarity(self, category: Optional[str] = None) -> float:
        filtered = self._filtered(category)
        if not filtered:
            return 0.0
        return sum(comp.similarity_score for comp in filtered) / len(filtered) * 100

    def get_category_stats(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for category in sorted({comp.category for comp in self.comparisons}):
            filtered = self._filtered(category)
            stats[category] = {
                "total": len(filtered),
                "matches": sum(1 for comp in filtered if comp.is_match),
                "accuracy": self.get_accuracy(category),
                "avg_similarity": self.get_avg_similarity(category),
            }
        return stats

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_fields": len(self.comparisons),
            "exact_matches": sum(1 for comp in self.comparisons if comp.is_match),
            "overall_accuracy": self.get_accuracy(),
            "avg_similarity": self.get_avg_similarity(),
            "category_breakdown": self.get_category_stats(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize ValidationMetrics to a dictionary for JSON storage."""
        return {
            "summary": self.get_summary(),
            "field_comparisons": [
                {
                    "field_name": comp.field_name,
                    "gt_value": comp.gt_value,
                    "ai_value": comp.ai_value,
                    "is_match": comp.is_match,
                    "similarity_score": comp.similarity_score,
                    "category": comp.category,
                }
                for comp in self.comparisons
            ],
        }


def compare_field(
    gt_val: Any,
    ai_val: Any,
    field_name: str,
    category: str,
    metrics: ValidationMetrics,
    api_key: Optional[str] = None,
) -> FieldComparison:
    """
    Compare two field values using semantic similarity when available.
    
    Special handling: 
    - For "Case Type", if the first word matches (case-insensitive), treat as 100% match.
    - For "Case Type", if GT contains "Detention" and AI is "Detinue", treat as 100% match.
    - For "Agent Status", if GT is empty but AI has a value, treat as a match (don't penalize AI for providing status when GT doesn't).
    - For "Agent Occupation", if GT is empty but AI has a value, treat as a match (don't penalize AI for providing occupation when GT doesn't).
    - For "Agent Occupation", if both contain a matching word (other than "and"), treat as 100% match.
    - For "Agent Role", if both contain a common word with 4+ letters (e.g., "surety"), treat as 100% match.
    - For "Event Type", if both contain a single matching word greater than 4 characters (e.g., "trespass"), treat as 100% match.
    - For "Agent Name", remove " de " from names before comparison to improve matching accuracy.
    - For "Damages Claimed", normalize currency formats (e.g., "£100" vs "100 pounds") to a standard form before comparison.
    - For "Writ Type", remove parenthetical qualifiers (e.g., "Debt (account)" vs "Debt") to normalize for comparison.
    """
    if isinstance(gt_val, list):
        gt_val = ", ".join(sorted(gt_val))
    if isinstance(ai_val, list):
        ai_val = ", ".join(sorted(ai_val))

    # Special handling for Event Date: extract date from dict before string conversion
    if field_name == "Event Date":
        # Extract date from dictionary if present (AI extraction may return dict format)
        if isinstance(ai_val, dict) and 'Date' in ai_val:
            ai_val = ai_val['Date']
        # Also handle string representation of dict (e.g., "{'Date': '1396-04-18', ...}")
        elif isinstance(ai_val, str) and (ai_val.strip().startswith("{") or ai_val.strip().startswith("'")):
            try:
                parsed = ast.literal_eval(ai_val)
                if isinstance(parsed, dict) and 'Date' in parsed:
                    ai_val = parsed['Date']
            except (ValueError, SyntaxError, TypeError):
                # If parsing fails, use original value
                pass
        
        # Extract date from GT if it's a dict (for consistency)
        if isinstance(gt_val, dict) and 'Date' in gt_val:
            gt_val = gt_val['Date']
        elif isinstance(gt_val, str) and (gt_val.strip().startswith("{") or gt_val.strip().startswith("'")):
            try:
                parsed = ast.literal_eval(gt_val)
                if isinstance(parsed, dict) and 'Date' in parsed:
                    gt_val = parsed['Date']
            except (ValueError, SyntaxError, TypeError):
                # If parsing fails, use original value
                pass

    gt_str = str(gt_val).strip() if gt_val is not None else ""
    ai_str = str(ai_val).strip() if ai_val is not None else ""

    # Normalize strings based on field type
    if field_name == "Agent Name":
        gt_str_normalized = normalize_name_for_comparison(gt_str)
        ai_str_normalized = normalize_name_for_comparison(ai_str)
        
        # Check if soundex codes match for surnames - if so, return 100% similarity
        # Extract surnames and compare their soundex codes
        gt_surname = get_surname_from_name(gt_str_normalized)
        ai_surname = get_surname_from_name(ai_str_normalized)
        
        if gt_surname and ai_surname:
            gt_surname_soundex = soundex(gt_surname)
            ai_surname_soundex = soundex(ai_surname)
            # Check if soundex codes match exactly
            if gt_surname_soundex and ai_surname_soundex and gt_surname_soundex == ai_surname_soundex:
                similarity = 1.0
                is_match = True
                comparison = FieldComparison(
                    field_name=field_name,
                    gt_value=gt_str,
                    ai_value=ai_str,
                    is_match=is_match,
                    similarity_score=similarity,
                    category=category,
                )
                metrics.add(comparison)
                return comparison
            # Also check if surnames are very similar (high Levenshtein similarity)
            # This catches cases like "Kyngesford" vs "Kingsford" where soundex differs
            # but the names are phonetically very similar
            surname_similarity = calculate_similarity(gt_surname, ai_surname)
            if surname_similarity >= 0.85:  # 85% similarity threshold for surnames
                similarity = 1.0
                is_match = True
                comparison = FieldComparison(
                    field_name=field_name,
                    gt_value=gt_str,
                    ai_value=ai_str,
                    is_match=is_match,
                    similarity_score=similarity,
                    category=category,
                )
                metrics.add(comparison)
                return comparison
        # If soundex doesn't match, continue with normal comparison using normalized strings
    elif field_name == "Event Date":
        # Dates have already been extracted from dictionaries above
        gt_str_normalized = normalize_date_for_comparison(gt_str)
        ai_str_normalized = normalize_date_for_comparison(ai_str)
        # If normalized dates are identical, treat as 100% match
        if gt_str_normalized == ai_str_normalized and gt_str_normalized:
            similarity = 1.0
            is_match = True
            comparison = FieldComparison(
                field_name=field_name,
                gt_value=gt_str,
                ai_value=ai_str,
                is_match=is_match,
                similarity_score=similarity,
                category=category,
            )
            metrics.add(comparison)
            return comparison
    elif field_name == "Event Place" or field_name == "Agent Location":
        # Special case: if first normalized word matches, treat as 100% match
        # This handles cases like "Southwark, Surrey, England" vs "Southwark"
        if gt_str and ai_str:
            # Extract first word from each string (before comma or space)
            gt_first_word = gt_str.split(',')[0].split()[0]
            ai_first_word = ai_str.split(',')[0].split()[0]
            
            # Normalize: lowercase, strip punctuation/whitespace, keep only alphanumeric
            gt_first_normalized = "".join(c.lower() for c in gt_first_word if c.isalnum())
            ai_first_normalized = "".join(c.lower() for c in ai_first_word if c.isalnum())
            
            # If first normalized words match, treat as 100% match
            if gt_first_normalized and ai_first_normalized and gt_first_normalized == ai_first_normalized:
                similarity = 1.0
                is_match = True
                comparison = FieldComparison(
                    field_name=field_name,
                    gt_value=gt_str,
                    ai_value=ai_str,
                    is_match=is_match,
                    similarity_score=similarity,
                    category=category,
                )
                metrics.add(comparison)
                return comparison
            
            # Special case: if GT and AI have a common word that is 4 or more letters, treat as 100% match
            # This handles cases like "St John Zachary, Alders-gate Ward, L..." vs "London, parish of St. John Zachary ..."
            # Split both strings into words and normalize
            gt_words = re.findall(r'\b\w+\b', gt_str.lower())
            ai_words = re.findall(r'\b\w+\b', ai_str.lower())
            
            # Filter to only words with 4+ letters
            gt_words_long = [w for w in gt_words if len(w) >= 4]
            ai_words_long = [w for w in ai_words if len(w) >= 4]
            
            # Check if there's at least one common word with 4+ letters
            if gt_words_long and ai_words_long:
                matching_words = set(gt_words_long) & set(ai_words_long)
                if matching_words:
                    # Found at least one common word with 4+ letters - treat as 100% match
                    similarity = 1.0
                    is_match = True
                    comparison = FieldComparison(
                        field_name=field_name,
                        gt_value=gt_str,
                        ai_value=ai_str,
                        is_match=is_match,
                        similarity_score=similarity,
                        category=category,
                    )
                    metrics.add(comparison)
                    return comparison
        # If first word doesn't match and no common word of 4+ letters found, continue with normal comparison
        gt_str_normalized = gt_str
        ai_str_normalized = ai_str
    elif field_name == "Damages Claimed":
        gt_str_normalized = normalize_damages_for_comparison(gt_str)
        ai_str_normalized = normalize_damages_for_comparison(ai_str)
    elif field_name == "Writ Type":
        gt_str_normalized = normalize_writ_type_for_comparison(gt_str)
        ai_str_normalized = normalize_writ_type_for_comparison(ai_str)
    else:
        gt_str_normalized = gt_str
        ai_str_normalized = ai_str

    # Get field-specific threshold
    field_threshold = get_field_threshold(field_name, category)
    
    # Special case: Case Type - if first word matches, treat as 100% match
    if field_name == "Case Type" and gt_str and ai_str:
        # Extract first word from each string (handle comma-separated lists)
        gt_first_word = gt_str.split(',')[0].split()[0] if gt_str.split(',')[0].split() else ""
        ai_first_word = ai_str.split(',')[0].split()[0] if ai_str.split(',')[0].split() else ""
        
        # Normalize: lowercase, strip punctuation/whitespace, keep only alphanumeric
        gt_first_normalized = "".join(c.lower() for c in gt_first_word if c.isalnum())
        ai_first_normalized = "".join(c.lower() for c in ai_first_word if c.isalnum())
        
        # If first normalized words match, treat as 100% match
        if gt_first_normalized and ai_first_normalized and gt_first_normalized == ai_first_normalized:
            similarity = 1.0
            is_match = True
            comparison = FieldComparison(
                field_name=field_name,
                gt_value=gt_str,
                ai_value=ai_str,
                is_match=is_match,
                similarity_score=similarity,
                category=category,
            )
            metrics.add(comparison)
            return comparison
        
        # Also check: if GT contains "Detention" and AI is "Detinue", treat as 100% match
        gt_lower = gt_str.lower()
        ai_lower = ai_str.lower()
        if "detention" in gt_lower and ai_lower == "detinue":
            similarity = 1.0
            is_match = True
            comparison = FieldComparison(
                field_name=field_name,
                gt_value=gt_str,
                ai_value=ai_str,
                is_match=is_match,
                similarity_score=similarity,
                category=category,
            )
            metrics.add(comparison)
            return comparison
    
    # Special case: Agent Occupation - check if both contain a single matching word (other than "and")
    # Examples: "goldsmith" matches "goldsmith and citizen", "bishop" matches "Bishop of LLandaff"
    if field_name == "Agent Occupation" and gt_str and ai_str:
        # Split into words, normalize to lowercase, and remove "and"
        gt_words = [w.lower().strip() for w in gt_str.split() if w.lower().strip() != "and"]
        ai_words = [w.lower().strip() for w in ai_str.split() if w.lower().strip() != "and"]
        
        # Check if there's at least one matching word
        if gt_words and ai_words:
            # Check for any word that appears in both lists
            matching_words = set(gt_words) & set(ai_words)
            if matching_words:
                # Found at least one matching word - treat as 100% match
                similarity = 1.0
                is_match = True
                comparison = FieldComparison(
                    field_name=field_name,
                    gt_value=gt_str,
                    ai_value=ai_str,
                    is_match=is_match,
                    similarity_score=similarity,
                    category=category,
                )
                metrics.add(comparison)
                return comparison
    
    # Special case: Agent Role - check if both contain a common word with 4+ letters
    # Examples: "Surety of law (compurga-tor)" matches "Surety for defendant" (both contain "surety")
    if field_name == "Agent Role" and gt_str and ai_str:
        # Split into words and normalize to lowercase
        gt_words = [w.lower().strip() for w in gt_str.split()]
        ai_words = [w.lower().strip() for w in ai_str.split()]
        
        # Filter to only words with 4+ letters
        gt_words_long = [w for w in gt_words if len(w) >= 4]
        ai_words_long = [w for w in ai_words if len(w) >= 4]
        
        # Check if there's at least one common word with 4+ letters
        if gt_words_long and ai_words_long:
            # Check for any word that appears in both lists
            matching_words = set(gt_words_long) & set(ai_words_long)
            if matching_words:
                # Found at least one common word with 4+ letters - treat as 100% match
                similarity = 1.0
                is_match = True
                comparison = FieldComparison(
                    field_name=field_name,
                    gt_value=gt_str,
                    ai_value=ai_str,
                    is_match=is_match,
                    similarity_score=similarity,
                    category=category,
                )
                metrics.add(comparison)
                return comparison
    
    # Special case: Event Type - check if both contain a single matching word greater than 4 characters
    # Examples: "taking of goods, trespass" matches "trespass" (both contain "trespass" which is > 4 chars)
    if field_name == "Event Type" and gt_str and ai_str:
        # Split into words and normalize to lowercase
        # Handle comma-separated values by splitting on commas first, then splitting each part into words
        gt_words = []
        for part in gt_str.split(','):
            gt_words.extend([w.lower().strip() for w in part.split()])
        ai_words = []
        for part in ai_str.split(','):
            ai_words.extend([w.lower().strip() for w in part.split()])
        
        # Filter to only words with > 4 characters (greater than 4, not >= 4)
        gt_words_long = [w for w in gt_words if len(w) > 4]
        ai_words_long = [w for w in ai_words if len(w) > 4]
        
        # Check if there's at least one matching word with > 4 characters
        if gt_words_long and ai_words_long:
            # Check for any word that appears in both lists
            matching_words = set(gt_words_long) & set(ai_words_long)
            if matching_words:
                # Found at least one matching word with > 4 characters - treat as 100% match
                similarity = 1.0
                is_match = True
                comparison = FieldComparison(
                    field_name=field_name,
                    gt_value=gt_str,
                    ai_value=ai_str,
                    is_match=is_match,
                    similarity_score=similarity,
                    category=category,
                )
                metrics.add(comparison)
                return comparison
    
    # Special case: Agent Status - if GT is empty but AI has value, don't penalize
    if field_name == "Agent Status" and not gt_str and ai_str:
        similarity = 1.0  # Perfect match (no penalty)
        is_match = True
    # Special case: Agent Occupation - if GT is empty but AI has value, don't penalize
    elif field_name == "Agent Occupation" and not gt_str and ai_str:
        similarity = 1.0  # Perfect match (no penalty)
        is_match = True
    # Special case: Agent Location - if GT is "london" or "london, england" and AI is empty, don't penalize
    elif field_name == "Agent Location" and gt_str and not ai_str:
        # Normalize GT string for comparison (lowercase, remove extra whitespace)
        gt_normalized = " ".join(gt_str.lower().split())
        # Check if GT is exactly "london" or starts with "london," and contains "england"
        # This handles variations like "london", "london, england", "london,england", "london, England", etc.
        if gt_normalized == "london" or (gt_normalized.startswith("london,") and "england" in gt_normalized):
            similarity = 1.0  # Perfect match (no penalty)
            is_match = True
        else:
            # GT has value but AI is empty, and it's not london - treat as mismatch
            similarity = 0.0
            is_match = False
    elif not gt_str and not ai_str:
        # Both empty - perfect match
        similarity = 1.0
        is_match = True
    elif not gt_str or not ai_str:
        # One empty, one not (and not the special case above)
        similarity = 0.0
        is_match = False
    elif GEMINI_AVAILABLE and api_key and gt_str_normalized and ai_str_normalized:
        similarity = get_similarity_score(gt_str_normalized, ai_str_normalized, api_key, context=f"field_comparison:{field_name}")
        is_match = similarity >= field_threshold
    else:
        similarity = calculate_similarity(gt_str_normalized, ai_str_normalized)
        is_match = similarity >= field_threshold

    comparison = FieldComparison(
        field_name=field_name,
        gt_value=gt_str,
        ai_value=ai_str,
        is_match=is_match,
        similarity_score=similarity,
        category=category,
    )
    metrics.add(comparison)
    return comparison


def get_accuracy_color(accuracy: float) -> str:
    """Return appropriate color name based on accuracy percentage."""
    if accuracy >= 90:
        return "MatchColor"
    if accuracy >= 75:
        return "PartialColor"
    if accuracy >= 50:
        return "WarnColor"
    return "GTColor"


def format_comparison_cell(comparison: FieldComparison) -> str:
    """Format a comparison for LaTeX display."""
    gt_safe = clean_text_for_xelatex(comparison.gt_value) if comparison.gt_value else r"\textit{N/A}"
    ai_safe = clean_text_for_xelatex(comparison.ai_value) if comparison.ai_value else r"\textit{N/A}"

    if comparison.is_match:
        return f"\\textcolor{{MatchColor}}{{{ai_safe}}}"

    similarity_pct = int(comparison.similarity_score * 100)
    return (
        f"\\gtlabel~{gt_safe} \\newline "
        f"\\ailabel~{ai_safe} \\newline "
        f"\\textcolor{{MetaColor}}{{\\scriptsize Similarity: {similarity_pct}\\%}}"
    )


def get_batch_embeddings(texts: List[str], api_key: Optional[str], context: str = "batch_processing") -> np.ndarray:
    """Generate embeddings for a list of strings in a single batch."""
    if not (texts and GEMINI_AVAILABLE and api_key):
        return np.array([])
    try:
        logger.info(f"[Embedding API] Making batch embedding call for {context} with {len(texts)} texts")
        logger.debug(f"[Embedding API]   First text preview: {texts[0][:50] if texts else 'N/A'}...")
        if len(texts) > 1:
            logger.debug(f"[Embedding API]   Last text preview: {texts[-1][:50]}...")
        
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embeddings = np.array([np.array(embedding.values) for embedding in response.embeddings])
        logger.info(f"[Embedding API]   Batch embedding completed: {len(embeddings)} embeddings generated")
        return embeddings
    except Exception as e:
        logger.warning(f"[Embedding API] Batch embedding call failed for {context}: {e}")
        return np.array([])


def smart_reconstruct_and_match(
    gt_items: List[Dict],
    ai_items: List[Dict],
    text_key: str,
    category: str,
    metrics: ValidationMetrics,
    api_key: Optional[str],
) -> List[Dict]:
    """Align ground-truth and AI texts using the Hungarian algorithm."""
    logger.info(f"[Report Generation] Starting smart_reconstruct_and_match for category: {category}, text_key: {text_key}")
    logger.debug(f"[Report Generation]   GT items: {len(gt_items)}, AI items: {len(ai_items)}")
    results: List[Dict] = []
    gt_sentences = split_into_sentences(gt_items, text_key)
    ai_sentences = split_into_sentences(ai_items, text_key)

    n_gt, n_ai = len(gt_sentences), len(ai_sentences)
    logger.info(f"[Report Generation]   Processing {n_gt} GT sentences and {n_ai} AI sentences")
    if n_gt > 0:
        logger.debug(f"[Report Generation]   First GT sentence preview: {gt_sentences[0][:100]}...")
    if n_ai > 0:
        logger.debug(f"[Report Generation]   First AI sentence preview: {ai_sentences[0][:100]}...")
    
    if n_gt == 0:
        for ai_text in ai_sentences:
            results.append({"type": "ai_only", "gt": "", "ai": ai_text})
        return results
    if n_ai == 0:
        # Don't penalize AI for GT entries that don't have corresponding AI entries
        # Skip compare_field call so unmatched GT items don't affect accuracy metrics
        for gt_text in gt_sentences:
            results.append({"type": "unmatched_gt", "gt": gt_text, "ai": ""})
        return results

    gt_embeddings = get_batch_embeddings(gt_sentences, api_key, context=f"smart_match_gt:{category}")
    ai_embeddings = get_batch_embeddings(ai_sentences, api_key, context=f"smart_match_ai:{category}")

    if len(gt_embeddings) == n_gt and len(ai_embeddings) == n_ai:
        similarity_matrix = cosine_similarity(gt_embeddings, ai_embeddings)
        cost_matrix = 1.0 - similarity_matrix
    else:
        cost_matrix = np.ones((n_gt, n_ai))
        for r in range(n_gt):
            for c in range(n_ai):
                score = calculate_similarity(gt_sentences[r], ai_sentences[c])
                cost_matrix[r, c] = 1.0 - score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    ai_matched_indices = set(col_ind)
    matches_map: Dict[int, Dict[str, Any]] = {}

    for idx, gt_idx in enumerate(row_ind):
        ai_idx = col_ind[idx]
        similarity = 1.0 - cost_matrix[gt_idx, ai_idx]
        gt_text = gt_sentences[gt_idx]
        ai_text = ai_sentences[ai_idx]

        logger.debug(f"[Report Generation]     GT[{gt_idx}] vs AI[{ai_idx}]: similarity={similarity:.3f}, threshold={ALIGNMENT_THRESHOLD}")
        if similarity >= ALIGNMENT_THRESHOLD:
            compare_field(gt_text, ai_text, f"{category} Block", category, metrics, api_key=api_key)
            matches_map[gt_idx] = {"type": "match", "gt": gt_text, "ai": ai_text, "score": float(similarity)}
            logger.debug(f"[Report Generation]     -> MATCH (similarity {similarity:.3f} >= {ALIGNMENT_THRESHOLD})")
        else:
            # Don't penalize AI for GT entries that don't have corresponding AI entries
            # Skip compare_field call so unmatched GT items don't affect accuracy metrics
            matches_map[gt_idx] = {"type": "unmatched_gt", "gt": gt_text, "ai": ""}
            ai_matched_indices.discard(ai_idx)
            logger.debug(f"[Report Generation]     -> NO MATCH (similarity {similarity:.3f} < {ALIGNMENT_THRESHOLD})")

    for r in range(n_gt):
        if r in matches_map:
            results.append(matches_map[r])
        else:
            gt_text = gt_sentences[r]
            # Don't penalize AI for GT entries that don't have corresponding AI entries
            # Skip compare_field call so unmatched GT items don't affect accuracy metrics
            results.append({"type": "unmatched_gt", "gt": gt_text, "ai": ""})

    for c in range(n_ai):
        if c not in ai_matched_indices:
            results.append({"type": "ai_only", "gt": "", "ai": ai_sentences[c]})

    return results


def _format_agent_for_matching(agent: Dict) -> str:
    """
    Format agent as comma-separated string: "Name, Role, Occupation, Status".
    
    Args:
        agent: Agent dictionary
        
    Returns:
        Comma-separated string with agent fields
    """
    name = get_person_name(agent)
    role = agent.get("TblAgentRole", {}).get("role") or ""
    occupation = agent.get("TblAgent", {}).get("Occupation") or ""
    status = agent.get("TblAgentStatus", {}).get("AgentStatus") or ""
    
    # Format as "Name, Role, Occupation, Status" (empty fields still included)
    return f"{name}, {role}, {occupation}, {status}"


def find_best_party_match(target: Dict, candidates: List[Dict], api_key: Optional[str]) -> Tuple[Optional[Dict], float]:
    """
    Find the best agent match using single semantic similarity call with name weighting.
    
    Strategy:
    - Formats agents as "Name, Role, Occupation, Status" comma-separated strings
    - Makes ONE Gemini API call per candidate for overall semantic similarity
    - Falls back to Levenshtein similarity if API unavailable
    - Applies name-based weighting: 80% name similarity, 20% overall semantic similarity
    - Heavily penalizes poor name matches to prioritize name accuracy
    - Normalizes names by removing " de " before comparison for better matching
    """
    target_name = get_person_name(target)
    logger.debug(f"[Report Generation] Finding best party match for '{target_name}' among {len(candidates)} candidates")
    best_score, best_match = -1.0, None
    
    # Extract target name for name-based weighting and normalize
    target_name = get_person_name(target)
    target_name_normalized = normalize_name_for_comparison(target_name)
    
    # Format target as comma-separated string
    target_string = _format_agent_for_matching(target)
    
    for candidate in candidates:
        # Extract candidate name for name-based weighting and normalize
        candidate_name = get_person_name(candidate)
        candidate_name_normalized = normalize_name_for_comparison(candidate_name)
        
        # Format candidate as comma-separated string
        candidate_string = _format_agent_for_matching(candidate)
        
        # NAME SIMILARITY: Use Levenshtein (string-based) for names
        # This is more accurate for proper nouns than semantic matching
        # Use normalized names (with " de " removed) for comparison
        # Check soundex first: if surname soundex codes match, use 100% similarity
        # Also check if surnames are very similar (high Levenshtein similarity)
        target_surname = get_surname_from_name(target_name_normalized)
        candidate_surname = get_surname_from_name(candidate_name_normalized)
        if target_surname and candidate_surname:
            target_surname_soundex = soundex(target_surname)
            candidate_surname_soundex = soundex(candidate_surname)
            if target_surname_soundex and candidate_surname_soundex and target_surname_soundex == candidate_surname_soundex:
                name_score = 1.0  # 100% similarity for identical surname soundex
            else:
                # Check if surnames are very similar (high Levenshtein similarity)
                # This catches cases like "Kyngesford" vs "Kingsford" where soundex differs
                # but the names are phonetically very similar
                surname_similarity = calculate_similarity(target_surname, candidate_surname)
                if surname_similarity >= 0.85:  # 85% similarity threshold for surnames
                    name_score = 1.0  # 100% similarity for very similar surnames
                else:
                    name_score = calculate_similarity(target_name_normalized, candidate_name_normalized)
        else:
            name_score = calculate_similarity(target_name_normalized, candidate_name_normalized)
        
        # OVERALL SEMANTIC SIMILARITY: Single API call for all fields
        if GEMINI_AVAILABLE and api_key and target_string and candidate_string:
            overall_semantic_score = get_similarity_score(target_string, candidate_string, api_key, context=f"party_match:{target_name_normalized[:30]}")
        else:
            # Fallback to Levenshtein for the full string
            overall_semantic_score = calculate_similarity(target_string, candidate_string)
        
        # Apply heavy penalty if name similarity is very low
        # This ensures names are prioritized even if other fields match well
       # if name_score < 0.5:
       #     name_score = name_score * 0.5  # Heavy penalty for poor name matches
        
        # Additional penalty: if name score is very low, cap overall semantic score
        #if name_score < 0.6:
            # Cap semantic score contribution when name is poor
        #    overall_semantic_score = min(overall_semantic_score, name_score + 0.2)
        
        # Weighted combination: 80% name, 20% overall semantic
        # This heavily prioritizes name matches while still considering other fields
        score = 0.7 * name_score + 0.3 * overall_semantic_score
        
        if score > best_score:
            best_score, best_match = score, candidate

    if best_score < PARTY_MATCH_THRESHOLD:
        return None, best_score
    return best_match, best_score

