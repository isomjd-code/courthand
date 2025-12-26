"""Case matching utilities for matching AI-extracted cases with ground truth cases."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional, Any

from .similarity import calculate_similarity, get_similarity_score

logger = logging.getLogger("report_generator")


def extract_county(gt_case: Dict[str, Any]) -> str:
    """Extract county from ground truth case."""
    return gt_case.get("TblCase", {}).get("County", "").lower().strip()


def extract_county_ai(ai_case: Dict[str, Any]) -> str:
    """Extract county from AI case."""
    return ai_case.get("TblCase", {}).get("County", "").lower().strip()


def extract_damages(gt_case: Dict[str, Any]) -> str:
    """Extract damages claimed from ground truth case."""
    return str(gt_case.get("TblCase", {}).get("DamClaimed", "")).lower().strip()


def extract_damages_ai(ai_case: Dict[str, Any]) -> str:
    """Extract damages claimed from AI case."""
    return str(ai_case.get("TblCase", {}).get("DamClaimed", "")).lower().strip()


def extract_writ_type(gt_case: Dict[str, Any]) -> str:
    """Extract writ type from ground truth case."""
    return str(gt_case.get("TblCase", {}).get("WritType", "")).lower().strip()


def extract_writ_type_ai(ai_case: Dict[str, Any]) -> str:
    """Extract writ type from AI case."""
    return str(ai_case.get("TblCase", {}).get("WritType", "")).lower().strip()


def extract_party_names(gt_case: Dict[str, Any]) -> List[str]:
    """Extract all party names from ground truth case."""
    names = []
    agents = gt_case.get("Agents", [])
    for agent in agents:
        name_info = agent.get("TblName", {})
        if not name_info:
            continue
        christian_name = (name_info.get("Christian name") or "").strip()
        surname = (name_info.get("Surname") or "").strip()
        if christian_name or surname:
            full_name = f"{christian_name} {surname}".strip()
            if full_name:
                names.append(full_name.lower())
    return names


def extract_party_names_ai(ai_case: Dict[str, Any]) -> List[str]:
    """Extract all party names from AI case."""
    names = []
    agents = ai_case.get("Agents", [])
    for agent in agents:
        name_info = agent.get("TblName", {})
        if not name_info:
            continue
        christian_name = (name_info.get("Christian name") or "").strip()
        surname = (name_info.get("Surname") or "").strip()
        if christian_name or surname:
            full_name = f"{christian_name} {surname}".strip()
            if full_name:
                names.append(full_name.lower())
    return names


def calculate_case_match_score(
    gt_case: Dict[str, Any],
    ai_case: Dict[str, Any],
    api_key: Optional[str] = None
) -> float:
    """
    Calculate a match score between a ground truth case and an AI-extracted case.
    
    Returns a score between 0.0 and 1.0, where 1.0 is a perfect match.
    
    Scoring weights:
    - County: 30%
    - Damages: 25%
    - Writ type: 15%
    - Party names: 30%
    
    Args:
        gt_case: Ground truth case dictionary
        ai_case: AI-extracted case dictionary
        api_key: Optional API key for semantic similarity (not used currently)
        
    Returns:
        Match score between 0.0 and 1.0
    """
    scores = []
    weights = []
    
    # County match (30%)
    gt_county = extract_county(gt_case)
    ai_county = extract_county_ai(ai_case)
    if gt_county and ai_county:
        county_sim = calculate_similarity(gt_county, ai_county)
        scores.append(county_sim)
        weights.append(0.30)
    elif not gt_county and not ai_county:
        # Both missing - neutral score
        scores.append(0.5)
        weights.append(0.30)
    else:
        # One missing - penalty
        scores.append(0.0)
        weights.append(0.30)
    
    # Damages match (25%)
    gt_damages = extract_damages(gt_case)
    ai_damages = extract_damages_ai(ai_case)
    if gt_damages and ai_damages:
        # Normalize damages strings (remove currency symbols, normalize whitespace)
        gt_damages_norm = gt_damages.replace("£", "").replace(",", "").strip()
        ai_damages_norm = ai_damages.replace("£", "").replace(",", "").strip()
        damages_sim = calculate_similarity(gt_damages_norm, ai_damages_norm)
        scores.append(damages_sim)
        weights.append(0.25)
    elif not gt_damages and not ai_damages:
        scores.append(0.5)
        weights.append(0.25)
    else:
        scores.append(0.0)
        weights.append(0.25)
    
    # Writ type match (15%)
    gt_writ = extract_writ_type(gt_case)
    ai_writ = extract_writ_type_ai(ai_case)
    if gt_writ and ai_writ:
        writ_sim = calculate_similarity(gt_writ, ai_writ)
        scores.append(writ_sim)
        weights.append(0.15)
    elif not gt_writ and not ai_writ:
        scores.append(0.5)
        weights.append(0.15)
    else:
        scores.append(0.0)
        weights.append(0.15)
    
    # Party names match (30%)
    gt_names = extract_party_names(gt_case)
    ai_names = extract_party_names_ai(ai_case)
    if gt_names and ai_names:
        # Calculate best matching score between party lists
        # Use simple approach: average of best matches
        name_scores = []
        used_ai_indices = set()
        for gt_name in gt_names:
            best_match = 0.0
            best_idx = -1
            for i, ai_name in enumerate(ai_names):
                if i in used_ai_indices:
                    continue
                sim = calculate_similarity(gt_name, ai_name)
                if sim > best_match:
                    best_match = sim
                    best_idx = i
            if best_idx >= 0:
                name_scores.append(best_match)
                used_ai_indices.add(best_idx)
        # Penalize unmatched parties
        unmatched_gt = len(gt_names) - len(name_scores)
        unmatched_ai = len(ai_names) - len(used_ai_indices)
        total_parties = max(len(gt_names), len(ai_names), 1)
        match_ratio = len(name_scores) / total_parties
        if name_scores:
            avg_name_sim = sum(name_scores) / len(name_scores)
            # Combine average similarity with match ratio
            name_sim = avg_name_sim * match_ratio
        else:
            name_sim = 0.0
        scores.append(name_sim)
        weights.append(0.30)
    elif not gt_names and not ai_names:
        scores.append(0.5)
        weights.append(0.30)
    else:
        scores.append(0.0)
        weights.append(0.30)
    
    # Calculate weighted average
    if sum(weights) > 0:
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        weighted_score = 0.0
    
    return weighted_score


def find_best_case_matches(
    gt_cases: List[Dict[str, Any]],
    ai_cases: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    min_score: float = 0.3
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """
    Find the best matching pairs between ground truth cases and AI-extracted cases.
    
    Args:
        gt_cases: List of ground truth case dictionaries
        ai_cases: List of AI-extracted case dictionaries
        api_key: Optional API key for semantic similarity
        min_score: Minimum match score to include (default 0.3)
        
    Returns:
        List of tuples (gt_case, ai_case, score) sorted by score (descending)
    """
    matches = []
    
    for gt_case in gt_cases:
        best_ai_case = None
        best_score = 0.0
        
        for ai_case in ai_cases:
            score = calculate_case_match_score(gt_case, ai_case, api_key)
            if score > best_score:
                best_score = score
                best_ai_case = ai_case
        
        if best_ai_case and best_score >= min_score:
            matches.append((gt_case, best_ai_case, best_score))
            logger.info(
                f"Case match found: score={best_score:.3f}, "
                f"GT county={extract_county(gt_case)}, "
                f"AI county={extract_county_ai(best_ai_case)}"
            )
    
    # Sort by score (descending)
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

