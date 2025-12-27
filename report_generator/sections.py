"""LaTeX section builders for the validation report."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Try to import line preprocessing functions, but make it optional
try:
    from line_preprocessor_greyscale.processing import initial_line_extraction, process_line_image_greyscale
    LINE_PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Line preprocessing not available (opencv may be missing): {e}")
    LINE_PREPROCESSING_AVAILABLE = False
    initial_line_extraction = None
    process_line_image_greyscale = None

from .config import PARTY_MATCH_THRESHOLD
from .similarity import (
    GEMINI_AVAILABLE,
    ValidationMetrics,
    calculate_similarity,
    compare_field,
    find_best_party_match,
    format_comparison_cell,
    format_similarity_basis,
    get_accuracy_color,
    smart_reconstruct_and_match,
)
from .text_utils import (
    clean_text_for_xelatex,
    format_location,
    get_full_date_string,
    get_person_name,
    is_generic_location,
    normalize_string,
)


def generate_latex_preamble(meta: Dict[str, Any]) -> List[str]:
    """Generate the LaTeX document preamble."""
    case_id = clean_text_for_xelatex(meta.get("group_id", "Unknown"))
    return [
        r"\documentclass[11pt, a4paper]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{fontspec}",
        r"\usepackage{xcolor}",
        r"\usepackage{array}",
        r"\usepackage{longtable}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{multirow}",
        r"\usepackage{tikz}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{lastpage}",
        r"\usepackage{enumitem}",
        r"\usepackage{tcolorbox}",
        r"\tcbuselibrary{skins,breakable}",
        r"\usepackage{graphicx}",
        r"\setkeys{Gin}{width=0.75\textwidth,max height=4cm,keepaspectratio}",
        r"\usepackage{hyperref}",
        r"\usepackage{soul}",
        r"\usepackage{paracol}",
        r"\hypersetup{colorlinks=true, linkcolor=AccentBlue, urlcolor=AccentBlue, citecolor=AccentBlue}",
        r"\definecolor{MatchColor}{RGB}{34, 139, 34}",
        r"\definecolor{GTColor}{RGB}{178, 34, 34}",
        r"\definecolor{AIColor}{RGB}{0, 71, 171}",
        r"\definecolor{PartialColor}{RGB}{218, 165, 32}",
        r"\definecolor{WarnColor}{RGB}{255, 140, 0}",
        r"\definecolor{MetaColor}{RGB}{105, 105, 105}",
        r"\definecolor{HeaderBg}{RGB}{47, 79, 79}",
        r"\definecolor{HeaderFg}{RGB}{255, 255, 255}",
        r"\definecolor{AccentBlue}{RGB}{70, 130, 180}",
        r"\definecolor{DiffRed}{RGB}{255, 200, 200}",
        r"\definecolor{DiffGreen}{RGB}{200, 255, 200}",
        r"\definecolor{LowConfYellow}{RGB}{255, 255, 200}",
        r"\definecolor{LowConfRed}{RGB}{255, 200, 200}",
        r"\newtcolorbox{summarybox}[1][]{enhanced, breakable, colback=blue!5!white, colframe=AccentBlue, fonttitle=\bfseries\sffamily, title={#1}, boxrule=1pt, arc=3mm}",
        r"\newtcolorbox{metricbox}[1][]{enhanced, colback=gray!5!white, colframe=gray!50!black, fonttitle=\bfseries\sffamily\small, title={#1}, boxrule=0.5pt, arc=2mm, width=0.22\textwidth, halign=center}",
        r"\newtcolorbox{textbox}[1][]{enhanced, breakable, colback=gray!8!white, colframe=gray!60!black, fonttitle=\bfseries\sffamily\small, title={#1}, boxrule=0.5pt, arc=2mm, left=3mm, right=3mm, top=2mm, bottom=2mm}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        r"\fancyhead[L]{\sffamily\small AI Extraction Validation Report}",
        f"\\fancyhead[R]{{\\sffamily\\small {case_id}}}",
        r"\fancyfoot[C]{\sffamily\small Page \thepage\ of \pageref{LastPage}}",
        r"\renewcommand{\headrulewidth}{0.4pt}",
        r"\renewcommand{\footrulewidth}{0.4pt}",
        r"\usepackage{titlesec}",
        r"\titleformat{\section}{\Large\bfseries\sffamily\color{HeaderBg}}{}{0em}{}[\titlerule]",
        r"\titleformat{\subsection}{\large\bfseries\sffamily\color{AccentBlue}}{}{0em}{}",
        r"\titleformat{\subsubsection}{\normalsize\bfseries\sffamily}{}{0em}{}",
        r"\newcommand{\fieldlabel}[1]{\textbf{\sffamily #1}}",
        r"\newcommand{\gtlabel}{\textcolor{GTColor}{\textbf{[GT]}}}",
        r"\newcommand{\ailabel}{\textcolor{AIColor}{\textbf{[AI]}}}",
        r"\newcommand{\matchicon}{\textcolor{MatchColor}{\textbf{+}}}",
        r"\newcommand{\mismatchicon}{\textcolor{GTColor}{\textbf{-}}}",
        r"\newcommand{\partialicon}{\textcolor{PartialColor}{\textbf{?}}}",
        r"\newcommand{\progressbar}[2]{\begin{tikzpicture}[baseline=-0.5ex]\fill[gray!20] (0,0) rectangle (3cm,0.3cm);\fill[#1] (0,0) rectangle (#2*3cm,0.3cm);\end{tikzpicture}}",
        r"\newcommand{\diffgt}[1]{\textcolor{GTColor}{\sout{#1}}}",
        r"\newcommand{\diffai}[1]{\textcolor{MatchColor}{#1}}",
        r"\title{\sffamily\bfseries\Huge AI Extraction Validation Report}",
        f"\\author{{\\sffamily Case Reference: {case_id}}}",
        r"\date{\sffamily\today}",
    ]

def _find_low_confidence_names(
    source_material: Optional[List[Dict[str, Any]]],
    output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find all forenames, surnames, and place names with Bayesian probability < 90%.
    
    Reads from post_correction.json files to get Bayesian probabilities.
    
    Returns a list of dictionaries with 'name', 'type' ('forename', 'surname', or 'placename'),
    'probability', 'best_candidate', and 'top_alternatives'.
    """
    if not source_material:
        return []
    
    low_confidence_names = []
    seen_names = set()  # Track (name, type) pairs to avoid duplicates
    
    # Try to find post_correction.json files
    if not output_dir:
        return []
    
    # Build a map of image filename to post_correction.json path
    post_correction_files = {}
    for source in source_material:
        filename = source.get("filename", "")
        if not filename:
            continue
        
        # Post-correction files are saved as {filename}_post_correction.json in the same directory as master_record.json
        # Example: "CP40-562 320a.jpg" -> "CP40-562 320a.jpg_post_correction.json"
        post_correction_path = os.path.join(output_dir, f"{filename}_post_correction.json")
        if os.path.exists(post_correction_path):
            post_correction_files[filename] = post_correction_path
    
    # Process each source material entry
    for source in source_material:
        filename = source.get("filename", "")
        post_correction_path = post_correction_files.get(filename)
        
        if not post_correction_path or not os.path.exists(post_correction_path):
            continue
        
        try:
            with open(post_correction_path, "r", encoding="utf-8") as f:
                post_correction_data = json.load(f)
            
            # Handle case where JSON file contains a JSON string (double-encoded)
            if isinstance(post_correction_data, str):
                post_correction_data = json.loads(post_correction_data)
            
            # Ensure we have a dictionary
            if not isinstance(post_correction_data, dict):
                logger.warning(f"[Names to Check] post_correction.json for {filename} is not a dict, skipping")
                continue
            
            lines = post_correction_data.get("lines", [])
            
            for line in lines:
                # Process forenames
                for fn in line.get("forenames", []):
                    best_candidate = fn.get("best_candidate")
                    if not best_candidate:
                        continue
                    
                    probability = best_candidate.get("probability", 1.0)
                    if probability < 0.9:  # < 90%
                        name = best_candidate.get("text", fn.get("original", ""))
                        key = (name, "forename")
                        if key not in seen_names:
                            seen_names.add(key)
                            # Get top 3 alternatives
                            candidates = fn.get("candidates", [])
                            top_alternatives = sorted(
                                [c for c in candidates if c.get("text") != name],
                                key=lambda x: x.get("probability", 0.0),
                                reverse=True
                            )[:3]
                            
                            low_confidence_names.append({
                                "name": name,
                                "type": "forename",
                                "probability": probability,
                                "best_candidate": best_candidate,
                                "top_alternatives": top_alternatives,
                                "original": fn.get("original", "")
                            })
                
                # Process surnames
                for sn in line.get("surnames", []):
                    best_candidate = sn.get("best_candidate")
                    if not best_candidate:
                        continue
                    
                    probability = best_candidate.get("probability", 1.0)
                    if probability < 0.9:  # < 90%
                        name = best_candidate.get("text", sn.get("original", ""))
                        key = (name, "surname")
                        if key not in seen_names:
                            seen_names.add(key)
                            # Get top 3 alternatives
                            candidates = sn.get("candidates", [])
                            top_alternatives = sorted(
                                [c for c in candidates if c.get("text") != name],
                                key=lambda x: x.get("probability", 0.0),
                                reverse=True
                            )[:3]
                            
                            low_confidence_names.append({
                                "name": name,
                                "type": "surname",
                                "probability": probability,
                                "best_candidate": best_candidate,
                                "top_alternatives": top_alternatives,
                                "original": sn.get("original", "")
                            })
                
                # Process placenames
                for pn in line.get("placenames", []):
                    best_candidate = pn.get("best_candidate")
                    if not best_candidate:
                        continue
                    
                    probability = best_candidate.get("probability", 1.0)
                    if probability < 0.9:  # < 90%
                        name = best_candidate.get("text", pn.get("original", ""))
                        key = (name, "placename")
                        if key not in seen_names:
                            seen_names.add(key)
                            # Get top 3 alternatives
                            candidates = pn.get("candidates", [])
                            top_alternatives = sorted(
                                [c for c in candidates if c.get("text") != name],
                                key=lambda x: x.get("probability", 0.0),
                                reverse=True
                            )[:3]
                            
                            low_confidence_names.append({
                                "name": name,
                                "type": "placename",
                                "probability": probability,
                                "best_candidate": best_candidate,
                                "top_alternatives": top_alternatives,
                                "original": pn.get("original", "")
                            })
        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.debug(f"[Names to Check] Error reading post_correction.json for {filename}: {e}")
            continue
    
    return low_confidence_names


def _find_lines_containing_name(
    name: str,
    source_material: List[Dict],
    anglicized: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find all HTR lines that contain the given name (case-insensitive fuzzy matching).
    
    Returns a list of dictionaries with line information including filename, line_id, and text.
    """
    matching_lines = []
    name_lower = name.lower().strip()
    search_terms = [name_lower]
    
    # Also search for anglicized form if provided
    if anglicized:
        search_terms.append(anglicized.lower().strip())
    
    # Remove common medieval characters for matching
    def normalize_for_search(text: str) -> str:
        """Normalize text for fuzzy matching."""
        # Remove combining marks and normalize whitespace
        text = normalize_string(text).lower()
        # Remove common medieval abbreviations
        text = text.replace("ꝫ", "").replace("ꝛ", "r").replace("ſ", "s")
        return text
    
    for source in source_material:
        filename = source.get("filename", "")
        lines = source.get("lines", [])
        
        for line in lines:
            htr_text = line.get("text_htr", "")
            diplomatic_text = line.get("text_diplomatic", "")
            
            # Check both HTR and diplomatic text
            for text in [htr_text, diplomatic_text]:
                if not text:
                    continue
                
                normalized_text = normalize_for_search(text)
                
                # Check if any search term appears in the text
                for term in search_terms:
                    normalized_term = normalize_for_search(term)
                    if normalized_term in normalized_text or term in normalized_text.lower():
                        matching_lines.append({
                            "filename": filename,
                            "line_id": line.get("line_id", ""),
                            "original_file_id": line.get("original_file_id", ""),
                            "htr_text": htr_text,
                            "diplomatic_text": diplomatic_text,
                            "kraken_polygon": line.get("kraken_polygon"),
                            "kraken_bbox": line.get("kraken_bbox")
                        })
                        break  # Only add each line once
                if matching_lines and matching_lines[-1]["line_id"] == line.get("line_id"):
                    break  # Already added this line
    
    return matching_lines


def _find_kraken_json(output_dir: str, image_filename: str) -> Optional[str]:
    """
    Find the Kraken JSON file for a given image.
    
    Looks for kraken.json in subdirectories matching the image name.
    
    Args:
        output_dir: Directory containing the output files (where master_record.json is)
        image_filename: Image filename (e.g., "CP40-565 112a.jpg")
        
    Returns:
        Path to kraken.json file, or None if not found.
    """
    # Extract base name without extension
    base_name = os.path.splitext(image_filename)[0]
    
    # Look for subdirectories that might contain the kraken.json
    if not os.path.exists(output_dir):
        return None
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and base_name in item:
            kraken_json_path = os.path.join(item_path, "kraken.json")
            if os.path.exists(kraken_json_path):
                return kraken_json_path
    
    return None


def _get_baseline_from_kraken_json(kraken_json_path: str, line_id: str) -> Optional[List[Tuple[int, int]]]:
    """
    Extract baseline coordinates for a specific line from Kraken JSON.
    
    Args:
        kraken_json_path: Path to Kraken JSON file
        line_id: Line identifier (original_file_id from master_record, or line_id)
        
    Returns:
        List of (x, y) tuples for baseline points, or None if not found.
    """
    try:
        with open(kraken_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        lines = data.get("lines", [])
        for line in lines:
            # Try matching by id (could be original_file_id or line_id)
            if line.get("id") == line_id:
                baseline = line.get("baseline", [])
                if baseline:
                    return [tuple(point) for point in baseline]
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        logger.debug(f"[Baseline] Error reading Kraken JSON {kraken_json_path}: {e}")
    
    return None


def _derive_baseline_from_polygon(polygon: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Derive a baseline approximation from polygon coordinates.
    
    Uses the bottom edge of the polygon as the baseline.
    
    Args:
        polygon: List of [x, y] coordinate pairs
        
    Returns:
        List of (x, y) tuples for baseline points.
    """
    if not polygon or len(polygon) < 2:
        return []
    
    # Convert to tuples
    points = [(p[0], p[1]) for p in polygon]
    
    # Find the maximum y (bottom) for each x range
    # Simple approach: use the bottom-most points
    max_y = max(p[1] for p in points)
    bottom_points = [p for p in points if abs(p[1] - max_y) < 10]  # Within 10 pixels of bottom
    
    if len(bottom_points) < 2:
        # Fallback: use leftmost and rightmost points
        leftmost = min(points, key=lambda p: (p[0], -p[1]))  # Leftmost, prefer bottom
        rightmost = max(points, key=lambda p: (p[0], -p[1]))  # Rightmost, prefer bottom
        return [leftmost, rightmost]
    
    # Sort by x and take a few representative points
    bottom_points.sort(key=lambda p: p[0])
    
    # Take leftmost, middle, and rightmost points for a better baseline
    if len(bottom_points) > 3:
        indices = [0, len(bottom_points) // 2, len(bottom_points) - 1]
        return [bottom_points[i] for i in indices]
    
    return bottom_points


def _find_image_file(image_filename: str, search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    Find an image file with flexible matching (case-insensitive, spacing variations).
    
    Args:
        image_filename: The filename to search for
        search_dirs: List of directories to search in. If None, searches common locations.
    
    Returns:
        Path to the found image file, or None if not found.
    """
    if search_dirs is None:
        # Try common locations
        search_dirs = ["input_images", "../input_images", "."]
    
    # Normalize filename for matching
    base_name = os.path.splitext(image_filename)[0]
    ext = os.path.splitext(image_filename)[1] or ".jpg"
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        # Try exact match first
        exact_path = os.path.join(search_dir, image_filename)
        if os.path.exists(exact_path):
            return os.path.abspath(exact_path)
        
        # Try case-insensitive match
        for file in os.listdir(search_dir):
            if file.lower() == image_filename.lower():
                return os.path.abspath(os.path.join(search_dir, file))
        
        # Try normalized name (handle spacing variations)
        normalized_base = base_name.replace(" ", "-").replace("_", "-")
        for file in os.listdir(search_dir):
            file_base = os.path.splitext(file)[0]
            file_ext = os.path.splitext(file)[1] or ".jpg"
            normalized_file_base = file_base.replace(" ", "-").replace("_", "-")
            
            if normalized_file_base.lower() == normalized_base.lower() and file_ext.lower() == ext.lower():
                return os.path.abspath(os.path.join(search_dir, file))
    
    return None


def _find_existing_workflow_line_image(
    image_filename: str,
    line_id: str,
    original_file_id: Optional[str],
    output_dir: Optional[str]
) -> Optional[str]:
    """
    Find an existing line image from workflow processing directories.
    
    Checks in locations like:
    - {output_dir}/{basename}/lines/{line_id}.png
    - {output_dir}/{basename}/lines/{original_file_id}.png
    
    Args:
        image_filename: The source image filename (e.g., "CP40-559 055-a.jpg")
        line_id: Line identifier (e.g., "L02")
        original_file_id: Original file ID from Kraken (optional)
        output_dir: Output directory where workflow processing results are stored
    
    Returns:
        Path to the found line image file, or None if not found.
    """
    if not output_dir or not os.path.exists(output_dir):
        return None
    
    # Get base name without extension
    base_name = os.path.splitext(image_filename)[0]
    
    # Try different variations of the base name (with/without spaces, etc.)
    base_variants = [
        base_name,
        base_name.replace(" ", "_"),
        base_name.replace("_", " "),
    ]
    
    # Try different line identifiers
    line_identifiers = [line_id]
    if original_file_id:
        line_identifiers.append(original_file_id)
    
    # Try different extensions
    extensions = ['.png', '.jpg', '.jpeg']
    
    for base_var in base_variants:
        lines_dir = os.path.join(output_dir, base_var, "lines")
        if os.path.exists(lines_dir) and os.path.isdir(lines_dir):
            for line_ident in line_identifiers:
                for ext in extensions:
                    line_image_path = os.path.join(lines_dir, f"{line_ident}{ext}")
                    if os.path.exists(line_image_path):
                        logger.debug(f"[Line Image] Found existing workflow line image: {line_image_path}")
                        return os.path.abspath(line_image_path)
    
    return None

def _extract_and_process_line_image(
    image_path: str,
    polygon: Optional[List[List[int]]],
    line_id: str,
    output_path: str,
    output_dir: Optional[str] = None,
    image_filename: Optional[str] = None,
    original_file_id: Optional[str] = None
) -> bool:
    """Extract and process a line image, saving as a high-compatibility JPEG."""
    if not LINE_PREPROCESSING_AVAILABLE:
        logger.warning(f"[Line Image] Line preprocessing not available (opencv missing), skipping line image extraction for {line_id}")
        return False
    
    try:
        if not os.path.exists(image_path):
            logger.warning(f"[Line Image] Image not found: {image_path}")
            return False
        
        if not polygon or len(polygon) < 2:
            logger.warning(f"[Line Image] Invalid polygon for {line_id}")
            return False
        
        page_image = Image.open(image_path)
        
        # Convert polygon to list of tuples
        polygon_tuples = [(p[0], p[1]) for p in polygon]
        
        # Try to get baseline from Kraken JSON
        baseline_points = None
        if output_dir and image_filename:
            kraken_json_path = _find_kraken_json(output_dir, image_filename)
            if kraken_json_path and original_file_id:
                baseline_points = _get_baseline_from_kraken_json(kraken_json_path, original_file_id)
        
        if not baseline_points:
            baseline_points = _derive_baseline_from_polygon(polygon)
        
        if not baseline_points:
            logger.warning(f"[Line Image] Could not get baseline for {line_id}")
            return False
        
        baseline_str = " ".join(f"{int(p[0])},{int(p[1])}" for p in baseline_points)
        
        # Use wider padding to include 1-2 words on either side for better context
        initial_result = initial_line_extraction(
            page_image,
            polygon_tuples,
            baseline_str,
            padding=50,  # Increased from 10 to 50 pixels to show surrounding context
        )
        
        if not initial_result:
            logger.warning(f"[Line Image] Initial extraction failed for {line_id}")
            return False
        
        line_rect_img, line_polygon_coords, line_baseline_points = initial_result
        
        if not process_line_image_greyscale:
            logger.warning(f"[Line Image] process_line_image_greyscale not available")
            return False
        
        final_image = process_line_image_greyscale(
            line_rect_img,
            line_polygon_coords,
            line_baseline_points,
            final_canvas_height=128,
            line_id_for_debug=line_id,
        )
        
        if not final_image:
            logger.warning(f"[Line Image] Processing failed for {line_id}")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # --- NEW APPROACH: FORCE JPEG ---
        # 1. Convert to RGB. This flattens any transparency (Alpha channel) which 
        #    often causes LaTeX to crash.
        final_image = final_image.convert('RGB')
            
        # 2. Save as JPEG. This format is much more robust for LaTeX inclusion.
        # Ensure the filename ends in .jpg
        if output_path.lower().endswith('.png'):
            output_path = output_path[:-4] + ".jpg"
            
        final_image.save(output_path, format="JPEG", quality=95)
        # --------------------------------
        
        logger.info(f"[Line Image] Successfully saved line image: {output_path}")
        return True
    except Exception as e:
        logger.error(f"[Line Image] Exception processing {line_id}: {e}", exc_info=True)
        return False

def generate_executive_summary(
    metrics: ValidationMetrics,
    extracted_entities: Optional[Dict[str, Any]] = None,
    source_material: Optional[List[Dict]] = None,
    input_images_dir: Optional[str] = None,
    output_dir: Optional[str] = None
) -> List[str]:
    """Generate the executive summary section with key metrics."""
    summary = metrics.get_summary()
    overall_acc = summary["overall_accuracy"]
    avg_sim = summary["avg_similarity"]
    acc_color = get_accuracy_color(overall_acc)
    sim_color = get_accuracy_color(avg_sim)

    latex = [
        r"\section{Executive Summary}",
        r"\begin{summarybox}[Validation Overview]",
        r"\begin{center}\begin{tabular}{cccc}",
        f"\\begin{{metricbox}}[Total Fields] \\Huge {summary['total_fields']} \\end{{metricbox}} &",
        f"\\begin{{metricbox}}[Exact Matches] \\Huge\\textcolor{{MatchColor}}{{{summary['exact_matches']}}} \\end{{metricbox}} &",
        f"\\begin{{metricbox}}[Accuracy] \\Huge\\textcolor{{{acc_color}}}{{{overall_acc:.1f}}}\\% \\end{{metricbox}} &",
        f"\\begin{{metricbox}}[Avg Similarity] \\Huge\\textcolor{{{sim_color}}}{{{avg_sim:.1f}}}\\% \\end{{metricbox}}",
        r"\end{tabular}\end{center}",
        r"\textit{See \hyperref[sec:field-comparison]{Detailed Field Comparison} section for field-by-field analysis.}",
        r"\end{summarybox}",
        r"\subsection{Accuracy by Category}",
        r"\begin{center}\begin{tabular}{lcccc}\toprule",
        r"\textbf{Category} & \textbf{Total} & \textbf{Matches} & \textbf{Accuracy} & \textbf{Visual} \\ \midrule",
    ]

    for category, stats in summary["category_breakdown"].items():
        accuracy = stats["accuracy"]
        color = get_accuracy_color(accuracy)
        latex.append(
            f"{clean_text_for_xelatex(category.replace('_', ' ').title())} & "
            f"{stats['total']} & {stats['matches']} & "
            f"\\textcolor{{{color}}}{{{accuracy:.1f}}}\\% & "
            f"\\progressbar{{{color}}}{{{accuracy/100.0:.2f}}} \\\\"
        )

    latex.extend([r"\bottomrule", r"\end{tabular}\end{center}"])
    
    # Add "Names to Check" section
    logger.info(f"[Names to Check] Checking conditions: source_material={source_material is not None}, output_dir={output_dir is not None}")
    
    # Check if image processing should be skipped (for faster report generation)
    skip_image_processing = os.getenv("SKIP_REPORT_IMAGE_PROCESSING", "false").lower() == "true"
    if skip_image_processing:
        logger.info("[Names to Check] Image processing disabled via SKIP_REPORT_IMAGE_PROCESSING environment variable")
    
    if source_material:
        low_confidence_names = _find_low_confidence_names(source_material, output_dir)
        
        if low_confidence_names:
            latex.append(r"\subsection{Names to Check}\label{sec:names-to-check}")
            latex.append(
                r"This section lists forenames, surnames, and place names with Bayesian probability less than 90\%. "
                r"Recurring entities are grouped together to reduce report length. "
                r"Each entry shows the Bayesian probability, the selected name, and the top 3 alternatives with their probabilities. "
                r"Each entry includes the HTR image lines where the name appears for visual verification."
            )
            
            # Group names by (name, type) to show recurring entities once
            from collections import defaultdict
            name_groups = defaultdict(list)
            for name_info in low_confidence_names:
                key = (name_info["name"], name_info["type"])
                name_groups[key].append(name_info)
            
            # Sort by number of occurrences (most frequent first), then by probability (lowest first)
            grouped_names = sorted(
                name_groups.items(),
                key=lambda x: (len(x[1]), -min(ni.get("probability", 1.0) for ni in x[1])),
                reverse=True
            )
            
            # Create directory for line images (in same directory as LaTeX output)
            line_images_dir = None
            if output_dir:
                line_images_dir = os.path.join(output_dir, "line_images")
                try:
                    os.makedirs(line_images_dir, exist_ok=True)
                    logger.info(f"[Names to Check] Created line_images directory: {line_images_dir}")
                except Exception as e:
                    logger.warning(f"[Names to Check] Could not create line_images directory: {e}")
                    line_images_dir = None
            
            total_names = len(grouped_names)
            logger.info(f"[Names to Check] Processing {total_names} grouped low-confidence names...")
            for idx, ((name, name_type), name_occurrences) in enumerate(grouped_names, 1):
                # Use the first occurrence's data as representative (they should all be similar)
                name_info = name_occurrences[0]
                probability = name_info.get("probability", 0.0)
                original = name_info.get("original", "")
                top_alternatives = name_info.get("top_alternatives", [])
                if idx % 10 == 0 or idx == total_names:
                    logger.info(f"[Names to Check] Processing name {idx}/{total_names}: {name}")
                
                # Format probability as percentage
                prob_pct = probability * 100.0
                prob_color = "red!70!black" if prob_pct < 50 else "orange!70!black" if prob_pct < 75 else "yellow!70!black"
                
                # Format name type
                type_label_map = {
                    "forename": "Forename",
                    "surname": "Surname",
                    "placename": "Place Name"
                }
                type_label = type_label_map.get(name_type, name_type.title())
                
                # Build header with probability and occurrence count
                occurrence_count = len(name_occurrences)
                header = f"{clean_text_for_xelatex(name)} ({type_label})"
                header += f" \\textcolor{{{prob_color}}}{{\\textbf{{[{prob_pct:.1f}\\%]}}}}"
                if occurrence_count > 1:
                    header += f" \\textit{{(Recurring: {occurrence_count} occurrences)}}"
                latex.append(f"\\subsubsection*{{{header}}}")
                
                # Show original if different from selected name
                if original and original != name:
                    latex.append(f"\\textit{{Original extraction: {clean_text_for_xelatex(original)}}}\\\\")
                
                # Show simplified Bayesian breakdown for selected name (only percentage, no raw logs)
                best_candidate = name_info.get("best_candidate", {})
                frequency = best_candidate.get("frequency")
                
                # Only show frequency if available, skip raw log values
                if frequency is not None:
                    latex.append(r"\textbf{Database Context:}")
                    latex.append(r"\begin{itemize}")
                    latex.append(f"\\item Database frequency: {frequency} occurrences")
                    latex.append(r"\end{itemize}")
                    latex.append("")
                
                # Show top 3 alternatives with Bayesian breakdown
                if top_alternatives:
                    latex.append(r"\textbf{Top 3 Alternatives:}")
                    latex.append(r"\begin{itemize}")
                    for alt_idx, alt in enumerate(top_alternatives, 1):
                        alt_name = alt.get("text", "")
                        alt_prob = alt.get("probability", 0.0) * 100.0
                        alt_log_prior = alt.get("log_prior")
                        alt_normalized_log_likelihood = alt.get("normalized_log_likelihood")
                        alt_frequency = alt.get("frequency")
                        
                        latex.append(f"\\item \\textbf{{{alt_idx}. {clean_text_for_xelatex(alt_name)}}}: {alt_prob:.1f}\\% probability")
                        # Only show frequency if available, skip raw log values
                        if alt_frequency is not None:
                            latex.append(f"  \\begin{{itemize}}")
                            latex.append(f"  \\item Database frequency: {alt_frequency} occurrences")
                            latex.append(f"  \\end{{itemize}}")
                    latex.append(r"\end{itemize}")
                    latex.append("")
                
                # Find matching lines for all occurrences of this name
                all_matching_lines = []
                for name_occ in name_occurrences:
                    occ_name = name_occ["name"]
                    occ_original = name_occ.get("original", "")
                    matching_lines = _find_lines_containing_name(occ_name, source_material)
                    if occ_original and occ_original != occ_name:
                        # Also search for original
                        original_lines = _find_lines_containing_name(occ_original, source_material)
                        # Merge and deduplicate
                        seen_line_ids = set()
                        merged_lines = []
                        for line in matching_lines + original_lines:
                            line_id = line.get("line_id")
                            if line_id and line_id not in seen_line_ids:
                                seen_line_ids.add(line_id)
                                merged_lines.append(line)
                        matching_lines = merged_lines
                    all_matching_lines.extend(matching_lines)
                
                # Deduplicate by line_id
                seen_line_ids = set()
                unique_matching_lines = []
                for line in all_matching_lines:
                    line_id = line.get("line_id")
                    if line_id and line_id not in seen_line_ids:
                        seen_line_ids.add(line_id)
                        unique_matching_lines.append(line)
                
                logger.info(f"[Names to Check] Found {len(unique_matching_lines)} unique lines containing '{name}' (across {occurrence_count} occurrences)")
                
                if unique_matching_lines:
                    # Show line references
                    line_refs = sorted([line.get("line_id", "") for line in unique_matching_lines if line.get("line_id")])
                    if line_refs:
                        latex.append(f"\\textbf{{Line References:}} {', '.join(line_refs[:20])}" + ("..." if len(line_refs) > 20 else ""))
                        latex.append("")
                    
                    latex.append(r"\textbf{Representative Occurrences (showing up to 3 images):}")
                    latex.append(r"\begin{itemize}")
                    
                    for line_idx, line_info in enumerate(unique_matching_lines[:3], 1):  # Limit to 3 images per grouped name
                        filename = line_info["filename"]
                        line_id = line_info["line_id"]
                        htr_text = line_info.get("htr_text", "")
                        diplomatic_text = line_info.get("diplomatic_text", "")
                        
                        # Prefer diplomatic transcription (from Step 1 LLM) over HTR text
                        display_text = diplomatic_text if diplomatic_text else htr_text
                        
                        # Try to extract and save line image using proper preprocessing
                        image_included = False
                        image_extracted = False
                        output_image_path = None
                        # Skip image processing if disabled or if we've already processed too many images
                        max_images_per_name = 3  # Limit to 3 images per name to speed up processing
                        
                        # Initialize output_image_path early so we can use it for extraction attempts
                        max_images_per_grouped_name = 3  # Limit to 3 images for grouped names
                        if line_images_dir and line_idx <= max_images_per_grouped_name:
                            # Create safe filename for image
                            safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)[:20]
                            image_filename = f"line_{line_id.replace('L', '')}_{safe_name}.jpg"
                            output_image_path = os.path.join(line_images_dir, image_filename)
                        
                        # First, check if image already exists (even if skip_image_processing is True)
                        if line_images_dir and line_idx <= max_images_per_grouped_name and output_image_path:
                            # Check if image already exists
                            if os.path.exists(output_image_path):
                                image_extracted = True
                                logger.debug(f"[Names to Check] Using existing line image: {output_image_path}")
                            else:
                                image_extracted = False
                                
                                # First, try to find existing workflow line image and copy it
                                # This works even when skip_image_processing is True
                                existing_workflow_image = _find_existing_workflow_line_image(
                                    filename,
                                    line_id,
                                    line_info.get("original_file_id"),
                                    output_dir
                                )
                                
                                if existing_workflow_image and os.path.exists(existing_workflow_image):
                                    try:
                                        # Copy the existing workflow image to the report line_images directory
                                        # Convert to JPEG if needed for LaTeX compatibility
                                        workflow_img = Image.open(existing_workflow_image)
                                        # Convert to RGB to remove alpha channel if present
                                        if workflow_img.mode in ('RGBA', 'LA', 'P'):
                                            workflow_img = workflow_img.convert('RGB')
                                        # Save as JPEG
                                        workflow_img.save(output_image_path, format="JPEG", quality=95)
                                        image_extracted = True
                                        logger.info(f"[Names to Check] Copied existing workflow line image from {existing_workflow_image} to {output_image_path}")
                                    except Exception as e:
                                        logger.warning(f"[Names to Check] Error copying workflow line image: {e}, will try to extract new one...")
                                        image_extracted = False
                                else:
                                    # Log when no existing workflow image is found (for debugging)
                                    if line_idx == 1:  # Only log once per name to avoid spam
                                        logger.debug(f"[Names to Check] No existing workflow line image found for {line_id} (filename: {filename}, original_file_id: {line_info.get('original_file_id', 'None')[:20] if line_info.get('original_file_id') else 'None'}...)")
                                
                                # If no existing workflow image found, try to extract new one (unless skipped)
                                if not image_extracted:
                                    if skip_image_processing:
                                        logger.debug(f"[Names to Check] Image processing skipped (SKIP_REPORT_IMAGE_PROCESSING=True) - will not extract new images, but existing workflow images were already checked")
                                    elif not line_info.get("kraken_polygon"):
                                        logger.debug(f"[Names to Check] No kraken_polygon available for line {line_id}, cannot extract image")
                                    else:
                                        logger.info(f"[Names to Check] Attempting to extract line image for {line_id} (name: {name})")
                                        # Find the image file (with flexible matching)
                                        search_dirs = []
                                        if input_images_dir:
                                            search_dirs.append(input_images_dir)
                                        # Also try relative to output directory
                                        if output_dir:
                                            parent_dir = os.path.dirname(output_dir)
                                            search_dirs.extend([
                                                os.path.join(parent_dir, "input_images"),
                                                os.path.join(parent_dir, "..", "input_images"),
                                            ])
                                        
                                        image_path = _find_image_file(filename, search_dirs)
                                        if image_path:
                                            logger.info(f"[Names to Check] Found source image: {image_path}")
                                        else:
                                            logger.warning(f"[Names to Check] Could not find image file: {filename} (searched in: {search_dirs})")
                                        
                                        if image_path:
                                            logger.info(f"[Names to Check] Extracting line image for {line_id}...")
                                            # Try to extract, but don't block if it fails or takes too long
                                            try:
                                                # Add a simple timeout by checking elapsed time
                                                import time
                                                start_time = time.time()
                                                image_extracted = _extract_and_process_line_image(
                                                    image_path,
                                                    line_info.get("kraken_polygon"),
                                                    line_id,
                                                    output_image_path,
                                                    output_dir=output_dir,
                                                    image_filename=filename,
                                                    original_file_id=line_info.get("original_file_id")
                                                )
                                                elapsed = time.time() - start_time
                                                if image_extracted:
                                                    logger.info(f"[Names to Check] Successfully extracted line image for {line_id} in {elapsed:.1f}s")
                                                else:
                                                    logger.warning(f"[Names to Check] Image extraction failed for {line_id}")
                                                if elapsed > 5.0:
                                                    logger.warning(f"[Names to Check] Image extraction took {elapsed:.1f}s for {line_id} (slow)")
                                            except Exception as e:
                                                logger.error(f"[Names to Check] Error during image extraction for {line_id}: {e}", exc_info=True)
                                                image_extracted = False
                        
                        # Include image in LaTeX if it exists (whether newly extracted or already existing)
                        if image_extracted and output_image_path:
                            # Verify image was actually created
                            # Handle case where function enforced .jpg extension even if we passed .png
                            real_path = output_image_path
                            if not os.path.exists(real_path) and real_path.endswith('.png'):
                                real_path = real_path.replace('.png', '.jpg')
                                
                            # Double-check file exists and is readable before including
                            if os.path.exists(real_path) and os.path.getsize(real_path) > 0:
                                try:
                                    # Try to verify it's a valid image file
                                    test_img = Image.open(real_path)
                                    test_img.verify()
                                    test_img.close()
                                    
                                    # Use relative path for LaTeX (relative to output_dir where LaTeX compiles)
                                    if output_dir:
                                        # Normalize paths to absolute before computing relative path
                                        abs_real_path = os.path.abspath(real_path)
                                        abs_output_dir = os.path.abspath(output_dir)
                                        rel_image_path = os.path.relpath(abs_real_path, abs_output_dir).replace("\\", "/")
                                        logger.debug(f"[Names to Check] Image path: real_path={real_path}, output_dir={output_dir}, rel_path={rel_image_path}")
                                    else:
                                        # Fallback to absolute path if output_dir not available
                                        rel_image_path = os.path.abspath(real_path).replace("\\", "/")
                                        logger.warning(f"[Names to Check] output_dir not available, using absolute path: {rel_image_path}")
                                    
                                    latex.append(
                                        f"\\item \\textbf{{{clean_text_for_xelatex(line_id)}}} "
                                        f"(from {clean_text_for_xelatex(filename)}): "
                                        f"\\textit{{{clean_text_for_xelatex(display_text)}}}"
                                    )
                                    latex.append(r"\par")
                                    latex.append(r"\vspace{0.2cm}")
                                    latex.append(r"\noindent")
                                    latex.append(r"\begin{center}")
                                    
                                    # --- LaTeX code for the image ---
                                    # Use constrained size to prevent huge images and blank pages
                                    # Note: Path is not cleaned with clean_text_for_xelatex because file paths
                                    # in \includegraphics should be literal, not escaped
                                    # Use full text width (less margins) to fill page width
                                    latex.append(f"\\includegraphics[width=\\textwidth,keepaspectratio]{{{rel_image_path}}}")
                                    
                                    latex.append(r"\end{center}")
                                    latex.append(r"\vspace{0.2cm}")
                                    image_included = True
                                    logger.info(f"[Names to Check] Added image to LaTeX: {rel_image_path} (exists: {os.path.exists(real_path)})")
                                except Exception as e:
                                    logger.warning(f"[Names to Check] Error validating image {real_path}: {e}, skipping image inclusion")
                                    image_included = False
                            else:
                                logger.warning(f"[Names to Check] Image file does not exist or is empty at expected path: {real_path}")
                                image_included = False
                        
                        if not image_included:
                            latex.append(
                                f"\\item \\textbf{{{clean_text_for_xelatex(line_id)}}} "
                                f"(from {clean_text_for_xelatex(filename)}): "
                                f"\\textit{{{clean_text_for_xelatex(display_text)}}}"
                            )
                    
                    latex.append(r"\end{itemize}")
                    latex.append("")  # Add spacing
        else:
            latex.append(r"\subsection{Names to Check}")
            latex.append(r"\textit{No names with Bayesian probability less than 90\% found.}")
    
    return latex


def _normalize_ai_case_structure(ai_case: Dict, ai_ref: Dict) -> Dict:
    """
    Normalize the AI case structure from final_index.json format to match ground truth format.
    This allows the comparison code to work with both structures.
    """
    normalized = {}
    
    # Normalize TblCase
    ai_tbl = ai_case.get("TblCase", {})
    normalized_tbl = {}
    # Map WritClassification to WritType, or use Writ if available, or use WritType if already correct
    if "WritType" in ai_tbl:
        normalized_tbl["WritType"] = ai_tbl["WritType"]
    elif "WritClassification" in ai_tbl:
        normalized_tbl["WritType"] = ai_tbl["WritClassification"]
    elif "Writ" in ai_tbl:
        normalized_tbl["WritType"] = ai_tbl["Writ"]
    # Map Damages to DamClaimed, or use DamClaimed if already correct
    if "DamClaimed" in ai_tbl:
        normalized_tbl["DamClaimed"] = ai_tbl["DamClaimed"]
    elif "Damages" in ai_tbl:
        normalized_tbl["DamClaimed"] = ai_tbl["Damages"]
    
    # Also check case_details for writ_type and damages_claimed (final_index.json format)
    case_details = ai_case.get("case_details", {})
    if case_details:
        if not normalized_tbl.get("WritType"):
            writ_type = case_details.get("writ_type", "")
            if writ_type:
                normalized_tbl["WritType"] = writ_type
        if not normalized_tbl.get("DamClaimed"):
            damages_claimed = case_details.get("damages_claimed") or case_details.get("damages", "")
            if damages_claimed:
                normalized_tbl["DamClaimed"] = damages_claimed
    
    # Preserve any other fields from ai_tbl that are already in the correct format
    # (e.g., County, Term, CaseRot, etc.)
    for key, value in ai_tbl.items():
        if key not in normalized_tbl and value is not None and value != "":
            normalized_tbl[key] = value
    
    normalized["TblCase"] = normalized_tbl
    
    # Normalize TblCaseType - derive from WritClassification if available
    if "WritClassification" in ai_tbl:
        writ_class = ai_tbl["WritClassification"]
        # Try to extract case type from writ classification
        case_type = []
        if "Debt" in writ_class:
            case_type.append("Debt")
        if "Rent" in writ_class or "Annuity" in writ_class:
            case_type.append("Real action  / rents / damage to real estate")
        if case_type:
            normalized["TblCaseType"] = {"CaseType": case_type}
    
    # Normalize Agents from TblAgents, TblInvolved, TblEntity, TblParty, or entities array (final_index.json format)
    # First check if Agents is already in the correct format
    agents_normalized = False
    if "Agents" in ai_case and isinstance(ai_case.get("Agents"), list) and len(ai_case.get("Agents", [])) > 0:
        # Check if it's already in the correct format (has TblName structure)
        first_agent = ai_case["Agents"][0]
        if isinstance(first_agent, dict) and "TblName" in first_agent:
            normalized["Agents"] = ai_case["Agents"]
            agents_normalized = True
    
    # If not already normalized, try other sources
    if not agents_normalized and "TblAgents" in ai_case:
        # Handle TblAgents format (from final_index.json)
        normalized_agents = []
        for agent_entry in ai_case.get("TblAgents", []):
            # Parse name - try to split into first name and surname
            full_name = agent_entry.get("AgentName", "")
            name_parts = full_name.split(maxsplit=1) if full_name else []
            christian_name = name_parts[0] if len(name_parts) > 0 else ""
            surname = name_parts[1] if len(name_parts) > 1 else (name_parts[0] if name_parts else "")
            
            # Get role and status
            role = agent_entry.get("AgentRole", "")
            agent_status = agent_entry.get("AgentStatus")
            
            # Try to extract occupation from AgentStatus if it looks like one
            occupation = None
            if agent_status:
                status_lower = str(agent_status).lower()
                # Check for common occupations
                if "prior" in status_lower:
                    occupation = "prior"
                elif "dean" in status_lower:
                    occupation = "dean"
                elif "bishop" in status_lower:
                    occupation = "bishop"
                elif "attorney" in status_lower:
                    occupation = "attorney"
            
            # Map role names if needed
            role_mapping = {
                "Attorney": "Attorney of plaintiff",  # Default, may need refinement
                "Predecessor": "Other",
                "Mentioned": "Other"
            }
            mapped_role = role_mapping.get(role, role)
            
            agent = {
                "TblName": {
                    "Christian_name": christian_name,
                    "Surname": surname,
                    "Suffix": ""
                },
                "TblAgentRole": {
                    "role": mapped_role
                },
                "TblAgent": {
                    "Occupation": occupation,
                    "AgentStatus": None
                },
                "TblAgentStatus": {
                    "AgentStatus": agent_status
                }
            }
            normalized_agents.append(agent)
        
        normalized["Agents"] = normalized_agents
        agents_normalized = True
    
    if not agents_normalized and ("TblInvolved" in ai_case or "TblEntity" in ai_case or "TblParty" in ai_case):
        normalized_agents = []
        # Use TblEntity if available, otherwise TblInvolved, otherwise TblParty
        entity_list = ai_case.get("TblEntity", ai_case.get("TblInvolved", ai_case.get("TblParty", [])))
        for involved in entity_list:
            # Parse name - try to split into first name and surname
            full_name = involved.get("Name", "")
            name_parts = full_name.split(maxsplit=1) if full_name else []
            christian_name = name_parts[0] if len(name_parts) > 0 else ""
            surname = name_parts[1] if len(name_parts) > 1 else (name_parts[0] if name_parts else "")
            
            # Convert TblInvolved/TblEntity/TblParty format to Agents format
            # Use Occupation field if available (TblParty format), otherwise try to extract from Status
            occupation = involved.get("Occupation")
            if not occupation:
                occupation = None
            
            agent = {
                "TblName": {
                    "Christian_name": christian_name,
                    "Surname": surname,
                    "Suffix": ""
                },
                "TblAgentRole": {
                    "role": involved.get("Role", "")
                },
                "TblAgent": {
                    "Occupation": occupation,
                    "AgentStatus": None
                },
                "TblAgentStatus": {
                    "AgentStatus": involved.get("Status")
                }
            }
            # Try to extract occupation from Status if it looks like one and occupation not already set
            status = involved.get("Status", "")
            if not occupation and status:
                status_lower = status.lower()
                # Check for common occupations
                if "prior" in status_lower:
                    agent["TblAgent"]["Occupation"] = "prior"
                elif "dean" in status_lower:
                    agent["TblAgent"]["Occupation"] = "dean"
                elif "bishop" in status_lower:
                    agent["TblAgent"]["Occupation"] = "bishop"
                elif "attorney" in status_lower:
                    agent["TblAgent"]["Occupation"] = "attorney"
            
            normalized_agents.append(agent)
            
            # Handle Attorney field - add attorney as separate agent if specified
            attorney_name = involved.get("Attorney")
            if attorney_name and attorney_name != full_name:  # Don't duplicate if attorney is the same person
                # Determine attorney role based on the person they represent
                person_role = involved.get("Role", "")
                if person_role == "Plaintiff":
                    attorney_role = "Attorney of plaintiff"
                elif person_role == "Defendant":
                    attorney_role = "Attorney of defendant"
                else:
                    attorney_role = "Attorney"
                
                # Check if attorney is already in the list
                attorney_exists = any(
                    get_person_name(agent) == attorney_name 
                    for agent in normalized_agents
                )
                
                if not attorney_exists:
                    attorney_name_parts = attorney_name.split(maxsplit=1) if attorney_name else []
                    attorney_agent = {
                        "TblName": {
                            "Christian_name": attorney_name_parts[0] if len(attorney_name_parts) > 0 else "",
                            "Surname": attorney_name_parts[1] if len(attorney_name_parts) > 1 else "",
                            "Suffix": ""
                        },
                        "TblAgentRole": {
                            "role": attorney_role
                        },
                        "TblAgent": {
                            "Occupation": None,
                            "AgentStatus": None
                        },
                        "TblAgentStatus": {
                            "AgentStatus": None
                        }
                    }
                    normalized_agents.append(attorney_agent)
        
        # Also handle TblAttorney if present (separate array for attorneys)
        if "TblAttorney" in ai_case:
            for attorney_entry in ai_case.get("TblAttorney", []):
                attorney_name = attorney_entry.get("AttorneyName", "")
                if attorney_name and attorney_name.lower() != "unnamed":
                    # Check if attorney is already in the list
                    attorney_exists = any(
                        get_person_name(agent) == attorney_name 
                        for agent in normalized_agents
                    )
                    
                    if not attorney_exists:
                        # Determine attorney role based on the party they represent
                        party = attorney_entry.get("Party", "")
                        if "plaintiff" in party.lower() or "Plaintiff" in party:
                            attorney_role = "Attorney of plaintiff"
                        elif "defendant" in party.lower() or "Defendant" in party:
                            attorney_role = "Attorney of defendant"
                        else:
                            attorney_role = "Attorney"
                        
                        attorney_name_parts = attorney_name.split(maxsplit=1) if attorney_name else []
                        attorney_agent = {
                            "TblName": {
                                "Christian_name": attorney_name_parts[0] if len(attorney_name_parts) > 0 else "",
                                "Surname": attorney_name_parts[1] if len(attorney_name_parts) > 1 else "",
                                "Suffix": ""
                            },
                            "TblAgentRole": {
                                "role": attorney_role
                            },
                            "TblAgent": {
                                "Occupation": None,
                                "AgentStatus": None
                            },
                            "TblAgentStatus": {
                                "AgentStatus": None
                            }
                        }
                        normalized_agents.append(attorney_agent)
        
        normalized["Agents"] = normalized_agents
        agents_normalized = True
    
    if not agents_normalized and "entities" in ai_case:
        # Handle entities array from final_index.json format
        normalized_agents = []
        for entity in ai_case["entities"]:
            # Parse name - try to split into first name and surname
            full_name = entity.get("name", "")
            name_parts = full_name.split(maxsplit=1) if full_name else []
            christian_name = name_parts[0] if len(name_parts) > 0 else ""
            surname = name_parts[1] if len(name_parts) > 1 else ""
            
            # Map role from final_index.json format to ground truth format
            role = entity.get("role", "")
            # Convert role names if needed
            role_mapping = {
                "Plaintiff": "Plaintiff",
                "Defendant": "Defendant",
                "Mainpernor": "Surety for defendant",
                "Pledge": "Surety for defendant",
                "Attorney": "Attorney of plaintiff",  # Default, may need refinement
            }
            mapped_role = role_mapping.get(role, role)
            
            # Extract occupation and status from status field
            status = entity.get("status", "")
            occupation = None
            agent_status = None
            
            if status:
                status_lower = status.lower()
                # Check for common occupations
                if "prior" in status_lower:
                    occupation = "prior"
                elif "dean" in status_lower:
                    occupation = "dean"
                elif "bishop" in status_lower:
                    occupation = "bishop"
                elif "attorney" in status_lower:
                    occupation = "attorney"
                elif "citizen" in status_lower:
                    agent_status = "citizen"
                elif "skinner" in status_lower or "mercer" in status_lower or "husbandman" in status_lower:
                    # Extract occupation from status
                    occupation = status_lower.split()[0] if status_lower.split() else None
                else:
                    agent_status = status
            
            agent = {
                "TblName": {
                    "Christian_name": christian_name,
                    "Surname": surname,
                    "Suffix": ""
                },
                "TblAgentRole": {
                    "role": mapped_role
                },
                "TblAgent": {
                    "Occupation": occupation,
                    "AgentStatus": agent_status
                },
                "TblAgentStatus": {
                    "AgentStatus": agent_status
                }
            }
            normalized_agents.append(agent)
        
        # Also check case_details for plaintiff/defendant if they're not in entities
        case_details = ai_case.get("case_details", {})
        if case_details:
            plaintiff_name = case_details.get("plaintiff", "")
            defendant_name = case_details.get("defendant", "")
            
            # Check if plaintiff is already in entities
            plaintiff_found = any(
                entity.get("name", "").split() and plaintiff_name.split() 
                and entity.get("name", "").split()[0] == plaintiff_name.split()[0] 
                and entity.get("role") == "Plaintiff"
                for entity in ai_case.get("entities", [])
            )
            
            # Check if defendant is already in entities
            defendant_found = any(
                entity.get("name", "").split() and defendant_name.split()
                and entity.get("name", "").split()[0] == defendant_name.split()[0]
                and entity.get("role") == "Defendant"
                for entity in ai_case.get("entities", [])
            )
            
            # Add plaintiff if not found
            if plaintiff_name and not plaintiff_found:
                name_parts = plaintiff_name.split(maxsplit=1)
                normalized_agents.append({
                    "TblName": {
                        "Christian_name": name_parts[0] if len(name_parts) > 0 else "",
                        "Surname": name_parts[1] if len(name_parts) > 1 else "",
                        "Suffix": ""
                    },
                    "TblAgentRole": {
                        "role": "Plaintiff"
                    },
                    "TblAgent": {
                        "Occupation": None,
                        "AgentStatus": None
                    },
                    "TblAgentStatus": {
                        "AgentStatus": None
                    }
                })
            
            # Add defendant if not found
            if defendant_name and not defendant_found:
                name_parts = defendant_name.split(maxsplit=1)
                normalized_agents.append({
                    "TblName": {
                        "Christian_name": name_parts[0] if len(name_parts) > 0 else "",
                        "Surname": name_parts[1] if len(name_parts) > 1 else "",
                        "Suffix": ""
                    },
                    "TblAgentRole": {
                        "role": "Defendant"
                    },
                    "TblAgent": {
                        "Occupation": None,
                        "AgentStatus": None
                    },
                    "TblAgentStatus": {
                        "AgentStatus": None
                    }
                })
        
        normalized["Agents"] = normalized_agents
        agents_normalized = True
    
    # Normalize TblEvents from EventDate or case_details.event_dates
    if "EventDate" in ai_case:
        normalized_events = []
        for event in ai_case["EventDate"]:
            # Format date for EventDate list (needs Date and DateType)
            event_date_list = []
            if event.get("Date"):
                event_date_list.append({
                    "Date": event.get("Date", ""),
                    "DateType": event.get("Type", "")
                })
            normalized_event = {
                "EventType": event.get("Type", ""),
                "EventDate": event_date_list,
                "LocationDetails": {
                    "SpecificPlace": event.get("Place"),
                    "County": None,
                    "Country": "England"
                } if event.get("Place") else {}
            }
            normalized_events.append(normalized_event)
        normalized["TblEvents"] = normalized_events
    elif "case_details" in ai_case:
        # Handle event_dates from final_index.json format
        case_details = ai_case.get("case_details", {})
        event_dates = case_details.get("event_dates", [])
        if event_dates:
            normalized_events = []
            for event in event_dates:
                event_date_list = []
                if event.get("date"):
                    # Convert date to datetime format if needed
                    date_str = event.get("date", "")
                    if date_str and not " " in date_str:
                        date_str = f"{date_str} 00:00:00"
                    event_date_list.append({
                        "Date": date_str,
                        "DateType": "initial"  # Default type
                    })
                
                # Parse place string to extract location details
                place_str = event.get("place", "")
                location_details = {}
                if place_str:
                    # Try to extract parish, ward, county from place string
                    # Format is typically: "London, parish of St. John Walbrook, Ward of Cordwainer Street"
                    location_details = {
                        "SpecificPlace": place_str.split(",")[0].strip() if "," in place_str else place_str,
                        "Parish": "",
                        "Ward": "",
                        "County": None,
                        "Country": "England"
                    }
                    # Try to extract parish
                    if "parish" in place_str.lower():
                        parish_match = place_str.lower().split("parish")[1].split(",")[0].strip() if "parish" in place_str.lower() else ""
                        if parish_match:
                            location_details["Parish"] = parish_match.replace("of", "").strip()
                    # Try to extract ward
                    if "ward" in place_str.lower():
                        ward_match = place_str.lower().split("ward")[1].split(",")[0].strip() if "ward" in place_str.lower() else ""
                        if ward_match:
                            location_details["Ward"] = ward_match.replace("of", "").strip()
                
                normalized_event = {
                    "EventType": [event.get("event", "")],
                    "EventDate": event_date_list,
                    "LocationDetails": location_details
                }
                normalized_events.append(normalized_event)
            normalized["TblEvents"] = normalized_events
    
    # Normalize TblPleadings from CaseDetails, case_details.pleadings, or existing TblPleadings
    if "TblPleadings" in ai_case and isinstance(ai_case["TblPleadings"], list):
        # Already in expected format, use as-is (but ensure PleadingText key exists)
        pleadings = []
        for pleading in ai_case["TblPleadings"]:
            if isinstance(pleading, dict):
                pleading_text = pleading.get("PleadingText", "")
                if pleading_text:
                    pleadings.append({"PleadingText": pleading_text})
        if pleadings:
            normalized["TblPleadings"] = pleadings
    elif "CaseDetails" in ai_case:
        case_details = ai_case["CaseDetails"]
        pleadings = []
        if case_details.get("Count"):
            pleadings.append({"PleadingText": case_details["Count"]})
        if case_details.get("Defense"):
            pleadings.append({"PleadingText": case_details["Defense"]})
        if case_details.get("Replication"):
            pleadings.append({"PleadingText": case_details["Replication"]})
        if pleadings:
            normalized["TblPleadings"] = pleadings
    elif "case_details" in ai_case:
        # Handle pleadings from final_index.json format
        case_details = ai_case.get("case_details", {})
        pleadings_list = case_details.get("pleadings", [])
        if pleadings_list:
            pleadings = []
            for pleading in pleadings_list:
                if isinstance(pleading, dict):
                    phase = pleading.get("phase", "")
                    argument = pleading.get("argument", "")
                    if argument:
                        # Format as "Phase: Argument"
                        pleading_text = f"{phase}: {argument}" if phase else argument
                        pleadings.append({"PleadingText": pleading_text})
                    # Also check for PleadingText directly
                    elif pleading.get("PleadingText"):
                        pleadings.append({"PleadingText": pleading.get("PleadingText")})
                elif isinstance(pleading, str):
                    # Handle case where pleading is just a string
                    pleadings.append({"PleadingText": pleading})
            if pleadings:
                normalized["TblPleadings"] = pleadings
    
    # Normalize TblPostea - combine Event and Description into PosteaText, or use existing PosteaText
    if "TblPostea" in ai_case and isinstance(ai_case["TblPostea"], list):
        normalized_postea = []
        for postea in ai_case["TblPostea"]:
            if not isinstance(postea, dict):
                continue
                
            # First check if PosteaText already exists (use it directly)
            if "PosteaText" in postea and postea.get("PosteaText"):
                postea_text = postea["PosteaText"]
            else:
                # Combine Event and Description into PosteaText
                event = postea.get("Event", "")
                description = postea.get("Description", "")
                # Properly combine event and description, only add colon if both exist
                if event and description:
                    postea_text = f"{event}: {description}"
                elif event:
                    postea_text = event
                elif description:
                    postea_text = description
                else:
                    postea_text = ""
            
            if postea_text:  # Only add if we have text
                normalized_postea_item = {
                    "PosteaText": postea_text,
                    "Date": postea.get("Date", "")
                }
                normalized_postea.append(normalized_postea_item)
        if normalized_postea:  # Only add if we have items
            normalized["TblPostea"] = normalized_postea
    
    return normalized


def _generate_case_metadata_block(
    gt_case: Dict,
    ai_case: Dict,
    ai_ref: Dict,
    metrics: ValidationMetrics,
    api_key: Optional[str],
) -> List[str]:
    """Helper to generate the metadata key-value block."""
    logger.info("[Report Generation] Generating case metadata block (will use field-level embeddings)")
    gt_tbl = gt_case.get("TblCase", {})
    ai_tbl = ai_case.get("TblCase", {})
    # Handle TblCaseType - it might be a dict, list, or None
    gt_case_type_obj = gt_case.get("TblCaseType", {})
    if isinstance(gt_case_type_obj, dict):
        gt_type = gt_case_type_obj.get("CaseType")
    elif isinstance(gt_case_type_obj, list):
        gt_type = ", ".join(str(x) for x in gt_case_type_obj) if gt_case_type_obj else None
    else:
        gt_type = None
    
    ai_case_type_obj = ai_case.get("TblCaseType", {})
    if isinstance(ai_case_type_obj, dict):
        ai_type = ai_case_type_obj.get("CaseType")
        # Handle case where CaseType itself might be a list
        if isinstance(ai_type, list):
            ai_type = ", ".join(str(x) for x in ai_type) if ai_type else None
    elif isinstance(ai_case_type_obj, list):
        ai_type = ", ".join(str(x) for x in ai_case_type_obj) if ai_case_type_obj else None
    else:
        ai_type = None
    
    # Normalize AI case structure if needed
    normalized_ai_case = _normalize_ai_case_structure(ai_case, ai_ref)
    normalized_ai_tbl = normalized_ai_case.get("TblCase", ai_tbl)
    # Handle normalized TblCaseType - it might be a dict, list, or None
    normalized_case_type_obj = normalized_ai_case.get("TblCaseType")
    if normalized_case_type_obj:
        if isinstance(normalized_case_type_obj, dict):
            normalized_ai_type = normalized_case_type_obj.get("CaseType")
            # Handle case where CaseType itself might be a list
            if isinstance(normalized_ai_type, list):
                normalized_ai_type = ", ".join(str(x) for x in normalized_ai_type) if normalized_ai_type else None
        elif isinstance(normalized_case_type_obj, list):
            normalized_ai_type = ", ".join(str(x) for x in normalized_case_type_obj) if normalized_case_type_obj else None
        else:
            normalized_ai_type = ai_type
    else:
        normalized_ai_type = ai_type

    # Extract term and year from ai_ref (handle both formats)
    ai_term = ai_ref.get("Term") or ai_ref.get("term", "")
    ai_year = ai_ref.get("CalendarYear") or ai_ref.get("dateyear", "")
    term_str = f"{ai_term} {ai_year}".strip() if ai_term or ai_year else ""

    field_pairs = [
        ("Term", gt_tbl.get("Term"), term_str, "Metadata"),
        ("County", gt_tbl.get("County"), ai_ref.get("County"), "Metadata"),
        ("Writ Type", gt_tbl.get("WritType"), normalized_ai_tbl.get("WritType"), "Metadata"),
        ("Damages Claimed", gt_tbl.get("DamClaimed"), normalized_ai_tbl.get("DamClaimed"), "Case Details"),
        ("Case Type", gt_type, normalized_ai_type, "Case Details"),
    ]

    latex = [r"\begin{tabularx}{\textwidth}{@{}lX@{}}"]
    for name, gt_val, ai_val, category in field_pairs:
        comparison = compare_field(gt_val, ai_val, name, category, metrics, api_key=api_key)
        latex.append(f"\\fieldlabel{{{name}:}} & {format_comparison_cell(comparison)} \\\\")
    latex.append(r"\end{tabularx}")
    return latex


def _generate_pleadings_and_postea_blocks(
    gt_case: Dict, ai_case: Dict, ai_ref: Dict, metrics: ValidationMetrics, api_key: Optional[str]
) -> List[str]:
    """Generate pleading and postea blocks using smart alignment with cross-matching.
    
    Allows GT pleadings to match AI pleadings or postea, and GT postea to match AI pleadings or postea.
    """
    logger.info("[Report Generation] Generating pleadings and postea blocks with cross-matching")
    logger.info("[Report Generation] ⚠ REQUIRING Gemini embeddings for postea and pleadings matching")
    
    # Verify embeddings are available before proceeding
    if not GEMINI_AVAILABLE:
        raise RuntimeError(
            "CRITICAL: Gemini embeddings are REQUIRED for postea and pleadings matching, "
            "but the Gemini library is not available. Please install: pip install google-genai"
        )
    if not api_key:
        raise RuntimeError(
            "CRITICAL: Gemini API key is REQUIRED for postea and pleadings matching. "
            "Please set GEMINI_API_KEY environment variable."
        )
    logger.info(f"[Report Generation] ✓ Gemini library available")
    logger.info(f"[Report Generation] ✓ API key available for embeddings")
    # Normalize AI case structure
    normalized_ai_case = _normalize_ai_case_structure(ai_case, ai_ref)
    
    # Collect all AI items (pleadings + postea) into a combined pool with source labels
    ai_pleadings = normalized_ai_case.get("TblPleadings", ai_case.get("TblPleadings", []))
    if not ai_pleadings and "case_details" in ai_case:
        pleadings_alt = ai_case.get("case_details", {}).get("pleadings", [])
        if pleadings_alt:
            logger.info(f"[Report Generation]   Found {len(pleadings_alt)} pleadings in case_details.pleadings, using them")
            ai_pleadings = pleadings_alt
    
    ai_postea = normalized_ai_case.get("TblPostea", ai_case.get("TblPostea", []))
    
    # Combine AI items with normalized text key and source labels
    # Normalize to use a consistent "Text" key that works for both pleadings and postea
    combined_ai_items = []
    for item in ai_pleadings:
        normalized_item = {**item, "_source": "pleading"}
        # Add a "Text" field that contains the PleadingText value for consistent matching
        if "PleadingText" in normalized_item:
            normalized_item["Text"] = normalized_item["PleadingText"]
        combined_ai_items.append(normalized_item)
    for item in ai_postea:
        normalized_item = {**item, "_source": "postea"}
        # Add a "Text" field that contains the PosteaText value for consistent matching
        if "PosteaText" in normalized_item:
            normalized_item["Text"] = normalized_item["PosteaText"]
        combined_ai_items.append(normalized_item)
    
    logger.info(f"[Report Generation]   Combined AI pool: {len(ai_pleadings)} pleadings + {len(ai_postea)} postea = {len(combined_ai_items)} total items")
    
    # Get GT items and normalize them to use "Text" key as well
    gt_pleadings = gt_case.get("TblPleadings", [])
    gt_postea = gt_case.get("TblPostea", [])
    
    # Normalize GT items to use "Text" key for consistent matching
    normalized_gt_pleadings = []
    for item in gt_pleadings:
        normalized_item = {**item}
        if "PleadingText" in normalized_item:
            normalized_item["Text"] = normalized_item["PleadingText"]
        normalized_gt_pleadings.append(normalized_item)
    
    normalized_gt_postea = []
    for item in gt_postea:
        normalized_item = {**item}
        if "PosteaText" in normalized_item:
            normalized_item["Text"] = normalized_item["PosteaText"]
        normalized_gt_postea.append(normalized_item)
    
    logger.info(f"[Report Generation]   GT items: {len(gt_pleadings)} pleadings, {len(gt_postea)} postea")
    
    # Match GT pleadings against combined AI pool
    latex = [r"\subsubsection*{Pleading}"]
    matched_ai_indices = set()
    if normalized_gt_pleadings:
        # Use "Text" as the key for consistent matching across both pleadings and postea
        aligned_pleadings = smart_reconstruct_and_match(
            normalized_gt_pleadings, combined_ai_items, "Text", "Case Details", metrics, api_key
        )
        
        # Track which AI items were matched
        # Since smart_reconstruct_and_match splits text into sentences, we need to find
        # which original AI items contain the matched sentences
        for item in aligned_pleadings:
            if item["type"] == "match":
                ai_text = item.get("ai", "").strip()
                if not ai_text:
                    continue
                # Find the index of the matched AI item by checking if the matched text
                # appears in the original item's text
                for idx, ai_item in enumerate(combined_ai_items):
                    if idx in matched_ai_indices:
                        continue  # Already matched
                    ai_item_text = str(ai_item.get("Text", "")).strip()
                    if not ai_item_text:
                        continue
                    # Check if the matched sentence appears in this item's text
                    # (accounting for sentence splitting)
                    if ai_text in ai_item_text or ai_item_text.startswith(ai_text[:50]) or ai_text.startswith(ai_item_text[:50]):
                        matched_ai_indices.add(idx)
                        break
        
        # Filter out matched items from combined pool for postea matching
        remaining_ai_items = [item for idx, item in enumerate(combined_ai_items) if idx not in matched_ai_indices]
        
        for item in aligned_pleadings:
            latex.append(r"\begin{textbox}")
            if item["type"] == "match":
                latex.extend(
                    [
                        f"\\gtlabel~{clean_text_for_xelatex(item['gt'])}\\\\",
                        f"\\ailabel~{clean_text_for_xelatex(item['ai'])}\\\\",
                        f"\\textcolor{{{get_accuracy_color(item['score']*100)}}}{{\\scriptsize Similarity: {item['score']*100:.1f}\\% (Embedding)}}",
                    ]
                )
            elif item["type"] == "unmatched_gt":
                latex.extend(
                    [
                        f"\\gtlabel~{clean_text_for_xelatex(item['gt'])}\\\\",
                        r"\textit{(No matching AI extraction found)}\\",
                    ]
                )
            else:
                latex.extend(
                    [
                        r"\textbf{(AI Only)}\\",
                        f"\\ailabel~{clean_text_for_xelatex(item['ai'])}\\\\",
                    ]
                )
            latex.append(r"\end{textbox}")
    else:
        logger.warning(f"[Report Generation]   No GT pleadings found in gt_case keys: {list(gt_case.keys())}")
        remaining_ai_items = combined_ai_items

    # Match GT postea against remaining AI items (unmatched pleadings + postea)
    latex.append(r"\subsubsection*{Postea}")
    if normalized_gt_postea:
        # Use "Text" as the key for consistent matching across both pleadings and postea
        aligned_postea = smart_reconstruct_and_match(
            normalized_gt_postea, remaining_ai_items, "Text", "Case Details", metrics, api_key
        )

        for item in aligned_postea:
            latex.append(r"\begin{textbox}")
            if item["type"] == "match":
                latex.extend(
                    [
                        f"\\gtlabel~{clean_text_for_xelatex(item['gt'])}\\\\",
                        f"\\ailabel~{clean_text_for_xelatex(item['ai'])}\\\\",
                        f"\\textcolor{{{get_accuracy_color(item['score']*100)}}}{{\\scriptsize Similarity: {item['score']*100:.1f}\\% (Embedding)}}",
                    ]
                )
            elif item["type"] == "unmatched_gt":
                latex.extend(
                    [
                        f"\\gtlabel~{clean_text_for_xelatex(item['gt'])}\\\\",
                        r"\textit{(No matching AI extraction found)}\\",
                    ]
                )
            else:
                latex.extend(
                    [
                        r"\textbf{(AI Only)}\\",
                        f"\\ailabel~{clean_text_for_xelatex(item['ai'])}\\\\",
                    ]
                )
            latex.append(r"\end{textbox}")
    else:
        logger.warning(f"[Report Generation]   No GT postea found in gt_case keys: {list(gt_case.keys())}")
    
    return latex


# Removed _format_certainty_badge - no longer using High/Medium/Low certainty heuristic


def _generate_events_table(
    gt_case: Dict,
    ai_case: Dict,
    ai_ref: Dict,
    metrics: ValidationMetrics,
    api_key: Optional[str],
    extracted_entities: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Helper to generate the Events table."""
    logger.info("[Report Generation] Generating events table (will use field-level embeddings)")
    # Normalize AI case structure
    normalized_ai_case = _normalize_ai_case_structure(ai_case, ai_ref)
    
    gt_events = gt_case.get("TblEvents", [])
    ai_events = list(normalized_ai_case.get("TblEvents", ai_case.get("TblEvents", [])))
    logger.info(f"[Report Generation]   Processing {len(gt_events)} GT events and {len(ai_events)} AI events")

    latex = [
        r"\subsubsection*{Events}",
        r"\begin{longtable}{|p{0.25\textwidth}|p{0.4\textwidth}|p{0.25\textwidth}|}",
        r"\hline \textbf{Type} & \textbf{Place} & \textbf{Date} \\ \hline \endhead",
    ]

    for gt_event in gt_events:
        best_match, best_score = None, -1.0
        gt_type_norm = normalize_string(gt_event.get("EventType"))
        gt_date_norm = normalize_string(get_full_date_string(gt_event.get("EventDate", [])))

        for ai_event in ai_events:
            ai_type_norm = normalize_string(ai_event.get("EventType"))
            ai_date_norm = normalize_string(get_full_date_string(ai_event.get("EventDate", [])))
            score = (calculate_similarity(gt_type_norm, ai_type_norm) + calculate_similarity(gt_date_norm, ai_date_norm)) / 2
            if score > best_score:
                best_score, best_match = score, ai_event

        if best_match and best_score > 0.5:
            ai_events.remove(best_match)
            type_comp = compare_field(
                gt_event.get("EventType"),
                best_match.get("EventType"),
                "Event Type",
                "Case Details",
                metrics,
                api_key=api_key,
            )
            ai_place_str = format_location(best_match.get("LocationDetails"))
            place_comp = compare_field(
                format_location(gt_event.get("LocationDetails")),
                ai_place_str,
                "Event Place",
                "Case Details",
                metrics,
                api_key=api_key,
            )
            place_display = format_comparison_cell(place_comp)
            
            date_comp = compare_field(
                get_full_date_string(gt_event.get("EventDate")),
                get_full_date_string(best_match.get("EventDate")),
                "Event Date",
                "Case Details",
                metrics,
                api_key=api_key,
            )
            latex.append(
                f"{format_comparison_cell(type_comp)} & "
                f"{place_display} & "
                f"{format_comparison_cell(date_comp)} \\\\ \\hline"
            )
        else:
            latex.append(
                f"\\gtlabel~{clean_text_for_xelatex(gt_event.get('EventType'))} & "
                f"\\gtlabel~{clean_text_for_xelatex(format_location(gt_event.get('LocationDetails')))} & "
                f"\\gtlabel~{clean_text_for_xelatex(get_full_date_string(gt_event.get('EventDate')))} \\\\ \\hline"
            )

    for ai_event in ai_events:
        ai_place_str = format_location(ai_event.get("LocationDetails"))
        place_display = f"\\ailabel~{clean_text_for_xelatex(ai_place_str)}"
        
        latex.append(
            f"\\ailabel~{clean_text_for_xelatex(ai_event.get('EventType'))} & "
            f"{place_display} & "
            f"\\ailabel~{clean_text_for_xelatex(get_full_date_string(ai_event.get('EventDate')))} \\\\ \\hline"
        )

    latex.append(r"\end{longtable}")
    return latex


def _generate_individuals_table(
    gt_case: Dict,
    ai_case: Dict,
    ai_ref: Dict,
    metrics: ValidationMetrics,
    api_key: Optional[str],
    extracted_entities: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Helper to generate the Individuals table with Hungarian algorithm matching."""
    logger.info("[Report Generation] Generating individuals table (will use party matching embeddings)")
    # Normalize AI case structure
    normalized_ai_case = _normalize_ai_case_structure(ai_case, ai_ref)
    
    gt_agents = gt_case.get("Agents", [])
    ai_agents = list(normalized_ai_case.get("Agents", ai_case.get("Agents", [])))
    logger.info(f"[Report Generation]   Processing {len(gt_agents)} GT agents and {len(ai_agents)} AI agents")

    latex = [
        r"\subsubsection*{Individuals}",
        r"\begin{longtable}{|p{0.18\textwidth}|p{0.11\textwidth}|p{0.11\textwidth}|p{0.18\textwidth}|p{0.28\textwidth}|}",
        r"\hline \textbf{Individual} & \textbf{Status} & \textbf{Occupation} & \textbf{Role} & \textbf{Location} \\ \hline \endhead",
    ]

    if not gt_agents:
        # No GT agents, show all AI agents
        for ai_agent in ai_agents:
            ai_name = get_person_name(ai_agent)
            name_display = f"\\ailabel~{clean_text_for_xelatex(ai_name)}"
            ai_location = format_location(ai_agent.get("TblAgent", {}).get("LocationDetails"))
            location_display = f"\\ailabel~{clean_text_for_xelatex(ai_location)}" if ai_location else r"\textit{N/A}"
            
            latex.append(
                f"{name_display} & "
                f"\\ailabel~{clean_text_for_xelatex(ai_agent.get('TblAgentStatus', {}).get('AgentStatus'))} & "
                f"\\ailabel~{clean_text_for_xelatex(ai_agent.get('TblAgent', {}).get('Occupation'))} & "
                f"\\ailabel~{clean_text_for_xelatex(ai_agent.get('TblAgentRole', {}).get('role'))} & "
                f"{location_display} \\\\ \\hline"
            )
        latex.append(r"\end{longtable}")
        return latex

    if not ai_agents:
        # No AI agents, show all GT agents
        for gt_agent in gt_agents:
            gt_location = format_location(gt_agent.get("TblAgent", {}).get("LocationDetails"))
            if is_generic_location(gt_location):
                location_display = r"\textit{N/A}"
            else:
                location_display = f"\\gtlabel~{clean_text_for_xelatex(gt_location)}"
            latex.append(
                f"\\gtlabel~{clean_text_for_xelatex(get_person_name(gt_agent))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgentStatus', {}).get('AgentStatus'))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgent', {}).get('Occupation'))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgentRole', {}).get('role'))} & "
                f"{location_display} \\\\ \\hline"
            )
        latex.append(r"\end{longtable}")
        return latex

    # Build cost matrix: rows = GT agents, columns = AI agents
    # Cost = 1 - similarity_score (lower is better for Hungarian algorithm)
    n_gt = len(gt_agents)
    n_ai = len(ai_agents)
    cost_matrix = np.ones((n_gt, n_ai))
    
    logger.info(f"[Report Generation]   Building cost matrix: {len(gt_agents)} x {len(ai_agents)} comparisons (each uses 1 embedding call)")
    for i, gt_agent in enumerate(gt_agents):
        for j, ai_agent in enumerate(ai_agents):
            _, similarity = find_best_party_match(gt_agent, [ai_agent], api_key)
            # Convert similarity to cost (1 - similarity)
            # If no match found, similarity will be low, cost will be high
            cost_matrix[i, j] = 1.0 - similarity
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Track which AI agents were matched
    matched_ai_indices = set(col_ind)
    
    # Process matched pairs
    for idx, gt_idx in enumerate(row_ind):
        ai_idx = col_ind[idx]
        gt_agent = gt_agents[gt_idx]
        ai_agent = ai_agents[ai_idx]
        
        # Get the similarity score for this match
        similarity = 1.0 - cost_matrix[gt_idx, ai_idx]
        
        # Only show as matched if similarity meets threshold
        if similarity >= PARTY_MATCH_THRESHOLD:
            ai_name = get_person_name(ai_agent)
            
            name_comp = compare_field(
                get_person_name(gt_agent),
                ai_name,
                "Agent Name",
                "Agents",
                metrics,
                api_key=api_key,
            )
            
            name_display = format_comparison_cell(name_comp)
            status_comp = compare_field(
                gt_agent.get("TblAgentStatus", {}).get("AgentStatus"),
                ai_agent.get("TblAgentStatus", {}).get("AgentStatus"),
                "Agent Status",
                "Agents",
                metrics,
                api_key=api_key,
            )
            occ_comp = compare_field(
                gt_agent.get("TblAgent", {}).get("Occupation"),
                ai_agent.get("TblAgent", {}).get("Occupation"),
                "Agent Occupation",
                "Agents",
                metrics,
                api_key=api_key,
            )
            role_comp = compare_field(
                gt_agent.get("TblAgentRole", {}).get("role"),
                ai_agent.get("TblAgentRole", {}).get("role"),
                "Agent Role",
                "Agents",
                metrics,
                api_key=api_key,
            )
            # Compare agent location (skip if GT location is just "England")
            gt_location = format_location(gt_agent.get("TblAgent", {}).get("LocationDetails"))
            if is_generic_location(gt_location):
                # Skip comparison and display for generic locations
                location_display = r"\textit{N/A}"
            else:
                ai_location = format_location(ai_agent.get("TblAgent", {}).get("LocationDetails"))
                location_comp = compare_field(
                    gt_location,
                    ai_location,
                    "Agent Location",
                    "Agents",
                    metrics,
                    api_key=api_key,
                )
                location_display = format_comparison_cell(location_comp)
            latex.append(
                f"{name_display} & {format_comparison_cell(status_comp)} & "
                f"{format_comparison_cell(occ_comp)} & {format_comparison_cell(role_comp)} & "
                f"{location_display} \\\\ \\hline"
            )
        else:
            # Similarity too low, treat as unmatched GT
            gt_location = format_location(gt_agent.get("TblAgent", {}).get("LocationDetails"))
            if is_generic_location(gt_location):
                location_display = r"\textit{N/A}"
            else:
                location_display = f"\\gtlabel~{clean_text_for_xelatex(gt_location)}"
            latex.append(
                f"\\gtlabel~{clean_text_for_xelatex(get_person_name(gt_agent))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgentStatus', {}).get('AgentStatus'))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgent', {}).get('Occupation'))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgentRole', {}).get('role'))} & "
                f"{location_display} \\\\ \\hline"
            )
            matched_ai_indices.discard(ai_idx)  # Don't count as matched
    
    # Process unmatched GT agents (if any)
    matched_gt_indices = set(row_ind)
    for i, gt_agent in enumerate(gt_agents):
        if i not in matched_gt_indices:
            gt_location = format_location(gt_agent.get("TblAgent", {}).get("LocationDetails"))
            if is_generic_location(gt_location):
                location_display = r"\textit{N/A}"
            else:
                location_display = f"\\gtlabel~{clean_text_for_xelatex(gt_location)}"
            latex.append(
                f"\\gtlabel~{clean_text_for_xelatex(get_person_name(gt_agent))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgentStatus', {}).get('AgentStatus'))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgent', {}).get('Occupation'))} & "
                f"\\gtlabel~{clean_text_for_xelatex(gt_agent.get('TblAgentRole', {}).get('role'))} & "
                f"{location_display} \\\\ \\hline"
            )

    # Process unmatched AI agents
    for j, ai_agent in enumerate(ai_agents):
        if j not in matched_ai_indices:
            ai_name = get_person_name(ai_agent)
            name_display = f"\\ailabel~{clean_text_for_xelatex(ai_name)}"
            ai_location = format_location(ai_agent.get("TblAgent", {}).get("LocationDetails"))
            
            latex.append(
                f"{name_display} & "
                f"\\ailabel~{clean_text_for_xelatex(ai_agent.get('TblAgentStatus', {}).get('AgentStatus'))} & "
                f"\\ailabel~{clean_text_for_xelatex(ai_agent.get('TblAgent', {}).get('Occupation'))} & "
                f"\\ailabel~{clean_text_for_xelatex(ai_agent.get('TblAgentRole', {}).get('role'))} & "
                f"\\ailabel~{clean_text_for_xelatex(ai_location)} \\\\ \\hline"
            )

    latex.append(r"\end{longtable}")
    return latex


def generate_case_comparison_section(
    gt_case: Dict,
    ai_case: Dict,
    ai_ref: Dict,
    metrics: ValidationMetrics,
    api_key: Optional[str],
    extracted_entities: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate the main comparison section."""
    logger.info("[Report Generation] ===== Starting case comparison section generation =====")
    logger.info("[Report Generation] This section will make multiple embedding API calls:")
    logger.info("[Report Generation]   - Field-level comparisons (1 call per comparison)")
    logger.info("[Report Generation]   - Batch embeddings for pleadings/postea alignment")
    logger.info("[Report Generation]   - Party matching (1 call per GT x AI agent pair)")
    case_id = clean_text_for_xelatex(gt_case.get("TblCase", {}).get("CaseRot", "Unknown Rotulus"))
    latex = [
        r"\section{Case Record Comparison}",
        f"\\subsection*{{Court of Common Pleas, CP40-565 {case_id}}}",
        r"\begin{tcolorbox}[colback=white, colframe=gray!75, breakable, sharp corners]",
    ]
    
    # Add warning if ground truth data is missing
    if not gt_case or not gt_case.get("TblCase"):
        latex.extend([
            r"\begin{tcolorbox}[colback=WarnColor!20, colframe=WarnColor, title=Warning: Missing Ground Truth Data, breakable]",
            r"\textbf{No ground truth data found in database for this case.}",
            r"\vspace{0.2cm}",
            r"\\",
            r"This report shows only AI extraction results. Ground truth fields will appear as \textit{N/A}.",
            r"\\",
            r"Possible reasons:",
            r"\begin{itemize}",
            r"\item Case not yet entered in the CP40 database",
            r"\item Roll/rotulus number mismatch (check database for exact format)",
            r"\item Database query returned no results",
            r"\end{itemize}",
            r"\end{tcolorbox}",
            r"\vspace{0.3cm}",
        ])
    latex.extend(_generate_case_metadata_block(gt_case, ai_case, ai_ref, metrics, api_key))
    latex.extend(_generate_pleadings_and_postea_blocks(gt_case, ai_case, ai_ref, metrics, api_key))
    latex.extend(_generate_events_table(gt_case, ai_case, ai_ref, metrics, api_key, extracted_entities=extracted_entities))
    latex.extend(_generate_individuals_table(gt_case, ai_case, ai_ref, metrics, api_key, extracted_entities=extracted_entities))
    latex.append(r"\end{tcolorbox}")
    logger.info("[Report Generation] ===== Case comparison section generation complete =====")
    return latex


def generate_transcription_section(
    source_material: List[Dict],
    extracted_entities: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> List[str]:
    """Generate the diplomatic transcription section with confidence-based color coding."""
    latex = [r"\newpage", r"\section{Diplomatic Transcription}"]
    if not source_material:
        latex.append(r"\textit{No source material available.}")
        return latex
    
    # Build a map of low-confidence names for highlighting
    low_confidence_map = {}  # word -> confidence
    if extracted_entities:
        for name_type in ["surnames", "place_names"]:
            entities = extracted_entities.get(name_type, [])
            for entity in entities:
                term = entity.get("term", "").lower().strip()
                probability = entity.get("probability")
                # Handle None case - default to 1.0 (high confidence) if not available
                if probability is None:
                    probability = 1.0
                if probability < 0.9:  # < 90% confidence
                    low_confidence_map[term] = probability
                    # Also check anglicized form for place names
                    if name_type == "place_names":
                        anglicized = entity.get("anglicized", "").lower().strip()
                        if anglicized:
                            low_confidence_map[anglicized] = probability
    
    # Also check post_correction.json files for forenames and surnames
    if output_dir:
        for source in source_material:
            filename = source.get("filename", "")
            if not filename:
                continue
            post_correction_path = os.path.join(output_dir, f"{filename}_post_correction.json")
            if os.path.exists(post_correction_path):
                try:
                    with open(post_correction_path, "r", encoding="utf-8") as f:
                        post_correction_data = json.load(f)
                    if isinstance(post_correction_data, str):
                        post_correction_data = json.loads(post_correction_data)
                    if isinstance(post_correction_data, dict):
                        lines = post_correction_data.get("lines", [])
                        for line in lines:
                            for fn in line.get("forenames", []):
                                best_candidate = fn.get("best_candidate")
                                if best_candidate:
                                    prob = best_candidate.get("probability")
                                    # Handle None case - default to 1.0 (high confidence) if not available
                                    if prob is None:
                                        prob = 1.0
                                    if prob < 0.9:
                                        name = best_candidate.get("text", "").lower().strip()
                                        if name:
                                            low_confidence_map[name] = prob
                            for sn in line.get("surnames", []):
                                best_candidate = sn.get("best_candidate")
                                if best_candidate:
                                    prob = best_candidate.get("probability")
                                    # Handle None case - default to 1.0 (high confidence) if not available
                                    if prob is None:
                                        prob = 1.0
                                    if prob < 0.9:
                                        name = best_candidate.get("text", "").lower().strip()
                                        if name:
                                            low_confidence_map[name] = prob
                except Exception as e:
                    logger.debug(f"[Transcription] Error reading post_correction.json for {filename}: {e}")

    def highlight_low_confidence_words(text: str) -> str:
        """Highlight low-confidence words in the text."""
        words = text.split()
        highlighted_words = []
        for word in words:
            # Remove punctuation for matching
            word_clean = word.lower().strip(".,;:!?()[]{}")
            confidence = low_confidence_map.get(word_clean)
            if confidence is not None:
                if confidence < 0.5:
                    # Red highlight for < 50%
                    highlighted_words.append(f"\\textcolor{{LowConfRed}}{{\\textbf{{{clean_text_for_xelatex(word)}}}}}")
                elif confidence < 0.9:
                    # Yellow highlight for 50-90%
                    highlighted_words.append(f"\\textcolor{{LowConfYellow!70!black}}{{\\textbf{{{clean_text_for_xelatex(word)}}}}}")
                else:
                    highlighted_words.append(clean_text_for_xelatex(word))
            else:
                highlighted_words.append(clean_text_for_xelatex(word))
        return " ".join(highlighted_words)

    latex.append(r"\begin{tcolorbox}[colback=blue!5!white, colframe=AccentBlue, title=Confidence Legend, breakable]")
    latex.append(r"\textbf{Color Coding:}")
    latex.append(r"\begin{itemize}")
    latex.append(r"\item \textcolor{LowConfRed}{\textbf{Red}}: Confidence < 50\%")
    latex.append(r"\item \textcolor{LowConfYellow!70!black}{\textbf{Yellow}}: Confidence 50-90\%")
    latex.append(r"\item \textbf{Black}: Confidence \textgreater{} 90\%")
    latex.append(r"\end{itemize}")
    latex.append(r"\end{tcolorbox}")
    latex.append(r"\vspace{0.3cm}")

    for source in source_material:
        filename = clean_text_for_xelatex(source.get("filename", "Unknown"))
        lines = sorted(source.get("lines", []), key=lambda item: item.get("line_id", ""))
        latex.append(f"\\begin{{textbox}}[Image: {filename}]")
        if lines:
            latex.append(r"\begin{description}[style=nextline, leftmargin=1.5cm, labelwidth=1.2cm]")
            for line in lines:
                line_id = clean_text_for_xelatex(line.get("line_id", ""))
                text = line.get("text_diplomatic", "")
                highlighted_text = highlight_low_confidence_words(text)
                latex.append(f"\\item[{line_id}] {highlighted_text}")
            latex.append(r"\end{description}")
        else:
            latex.append(r"\textit{No lines transcribed.}")
        latex.append(r"\end{textbox}\vspace{0.3cm}")
        # Only add newpage if there are more sources to avoid trailing blank pages
        if source != source_material[-1]:
            latex.append(r"\newpage")
    return latex


def generate_full_text_section(master_data: Dict[str, Any]) -> List[str]:
    """Generate the full text reconstructions section with aligned three-column layout."""
    text_content = master_data.get("text_content", {})
    consensus_diplomatic = (
        " ".join(
            line.get("text_diplomatic", "")
            for source in master_data.get("source_material", [])
            for line in sorted(source.get("lines", []), key=lambda item: item.get("line_id", ""))
        )
        or "N/A"
    )
    
    latin_text = text_content.get("latin_reconstructed", "N/A")
    english_text = text_content.get("english_translation", "N/A")
    
    latex = [
        r"\newpage",
        r"\section{Full Text Reconstructions}",
    ]
    
    # Show texts in separate subsections to avoid alignment issues and empty pages
    # This is clearer and avoids the problem of mismatched paragraph counts
    latex.append(r"\subsection*{Diplomatic Transcription}")
    latex.append(r"\begin{textbox}[Consensus Diplomatic Transcription]")
    if consensus_diplomatic and consensus_diplomatic != "N/A":
        latex.append(clean_text_for_xelatex(consensus_diplomatic))
    else:
        latex.append("N/A")
    latex.append(r"\end{textbox}")
    latex.append(r"\vspace{0.5cm}")
    
    latex.append(r"\subsection*{Expanded Latin}")
    latex.append(r"\begin{textbox}[Reconstructed Latin Text]")
    if latin_text and latin_text != "N/A":
        latex.append(clean_text_for_xelatex(latin_text))
    else:
        latex.append("N/A")
    latex.append(r"\end{textbox}")
    latex.append(r"\vspace{0.5cm}")
    
    latex.append(r"\subsection*{English Translation}")
    latex.append(r"\begin{textbox}[English Translation]")
    if english_text and english_text != "N/A":
        latex.append(clean_text_for_xelatex(english_text))
    else:
        latex.append("N/A")
    latex.append(r"\end{textbox}")
    
    return latex


def _get_short_basis(basis: str) -> str:
    """Get shortened abbreviation for similarity basis to fit in table."""
    basis_map = {
        "embedding": "Emb",
        "soundex": "Sdx",
        "levenshtein": "Lev",
        "exact_match": "Exact",
        "special_rule": "Rule",
        "date_normalization": "Date",
        "unknown": "?",
    }
    return basis_map.get(basis, basis[:4])


def generate_field_level_report(metrics: ValidationMetrics) -> List[str]:
    """Generate detailed field-level comparison report."""
    latex = [
        r"\newpage",
        r"\section{Detailed Field Comparison}\label{sec:field-comparison}",
        r"\begin{tcolorbox}[colback=blue!5!white, colframe=AccentBlue, title=Similarity Thresholds, breakable]",
        r"\textbf{Match Criteria:}",
        r"\begin{itemize}",
        r"\item \textbf{Agent Name} and \textbf{Event Place}: Similarity > 95\% required for match",
        r"\item \textbf{Case Details}: Similarity > 78\% required for match",
        r"\item \textbf{All other fields}: Similarity > 90\% required for match",
        r"\end{itemize}",
        r"\textbf{Mismatch Types:}",
        r"\begin{itemize}",
        r"\item \textbf{OCR Error}: Text was read incorrectly (e.g., 'Thom'am' vs 'Thomas')",
        r"\item \textbf{Classification Error}: Value not in AI's schema/ontology (e.g., 'Auditor' not recognized as valid role)",
        r"\item \textbf{Value not in schema}: AI's training set didn't include this categorical value",
        r"\end{itemize}",
        r"\end{tcolorbox}",
        r"\vspace{0.3cm}",
        r"\small",
        r"\begin{longtable}{|p{0.15\textwidth}|p{0.20\textwidth}|p{0.20\textwidth}|c|p{0.18\textwidth}|}",
        r"\hline \textbf{Field} & \textbf{Ground Truth} & \textbf{AI Extraction} & \textbf{Match} & \textbf{Sim (Basis)} \\ \hline \endhead",
        r"\hline \endfoot",
    ]

    for comparison in sorted(metrics.comparisons, key=lambda comp: (comp.category, comp.field_name)):
        gt_display = comparison.gt_value[:30] + "..." if len(comparison.gt_value) > 30 else comparison.gt_value
        ai_display = comparison.ai_value[:30] + "..." if len(comparison.ai_value) > 30 else comparison.ai_value
        field_display = comparison.field_name[:25] + "..." if len(comparison.field_name) > 25 else comparison.field_name
        similarity_pct = int(comparison.similarity_score * 100)
        sim_color = get_accuracy_color(similarity_pct)
        match_icon = r"\matchicon" if comparison.is_match else r"\mismatchicon"
        
        # Add explanation for categorical mismatches (keep it short to avoid overflow)
        mismatch_note = ""
        if not comparison.is_match and comparison.category in ["Agents", "Metadata"]:
            # Check if this might be a schema/classification error
            if similarity_pct < 50:
                mismatch_note = r" \tiny{(schema?)}"
            else:
                mismatch_note = r" \tiny{(OCR?)}"
        
        # Format similarity basis with shorter abbreviations
        basis_short = _get_short_basis(comparison.similarity_basis)

        latex.append(
            f"{clean_text_for_xelatex(field_display)} & "
            f"{clean_text_for_xelatex(gt_display)} & "
            f"{clean_text_for_xelatex(ai_display)} & "
            f"{match_icon} & "
            f"\\textcolor{{{sim_color}}}{{{similarity_pct}}}\\%~\\tiny{{({basis_short})}}{mismatch_note} \\\\ \\hline"
        )

    latex.extend([r"\end{longtable}", r"\normalsize"])
    return latex


def generate_cost_breakdown_section(master_data: Dict[str, Any]) -> List[str]:
    """Generate cost breakdown section showing token usage and costs by step."""
    token_usage = master_data.get("token_usage", {})
    estimated_cost = master_data.get("estimated_cost", {})
    
    logger.info(f"[Cost Breakdown] token_usage present: {bool(token_usage)}, estimated_cost present: {bool(estimated_cost)}")
    
    if not token_usage or not estimated_cost:
        logger.warning(f"[Cost Breakdown] Missing data - token_usage keys: {list(token_usage.keys()) if token_usage else 'None'}, estimated_cost keys: {list(estimated_cost.keys()) if estimated_cost else 'None'}")
        return [
            r"\newpage",
            r"\section{Processing Cost Breakdown}",
            r"\textit{No cost information available.}",
        ]
    
    latex = [
        r"\newpage",
        r"\section{Processing Cost Breakdown}",
        r"\begin{summarybox}[Cost Overview]",
        r"\textbf{Model:} Gemini 3 Flash Preview",
        r"\begin{itemize}",
        r"\item Input tokens: \$0.25 per million tokens",
        r"\item Output tokens (including thinking): \$1.50 per million tokens",
        r"\end{itemize}",
        r"\end{summarybox}",
        r"\vspace{0.5cm}",
    ]
    
    # Step names mapping
    step_display_names = {
        "step1_diplomatic_transcription": "Step 1: Diplomatic Transcription",
        "step2a_merge_and_extract": "Step 2a: Merge and Extract",
        "step2b_expand_abbreviations": "Step 2b: Expand Abbreviations",
        "step3_translation": "Step 3: Translation",
        "step4_indexing": "Step 4: Indexing",
    }
    
    # Generate table with step-by-step breakdown
    latex.extend([
        r"\subsection{Cost Breakdown by Processing Step}",
        r"\begin{center}",
        r"\footnotesize",  # Use even smaller font to fit more columns
        r"\begin{tabular}{|p{3cm}|r|r|r|r|r|r|r|}",
        r"\hline",
        r"\textbf{Step} & \textbf{Input} & \textbf{Thinking} & \textbf{Output} & \textbf{Input \$} & \textbf{Thinking \$} & \textbf{Output \$} & \textbf{Total} \\",
        r"\hline",
    ])
    
    # Process each step
    total_input_tokens_all = 0
    total_thinking_tokens_all = 0
    total_output_tokens_all = 0
    total_input_cost_all = 0.0
    total_thinking_cost_all = 0.0
    total_output_cost_all = 0.0
    total_cost_all = 0.0
    
    # Get step order - include all steps that exist in either token_usage or estimated_cost
    # First, try the standard order
    step_order = [
        "step1_diplomatic_transcription",
        "step2a_merge_and_extract",
        "step2b_expand_abbreviations",
        "step3_translation",
        "step4_indexing",
    ]
    
    # Also check for any additional steps that might exist
    all_steps = set(step_order)
    all_steps.update(token_usage.keys())
    all_steps.update(estimated_cost.keys())
    all_steps.discard("_totals")  # Remove totals from step list
    
    # Process steps in order
    for step_name in step_order:
        # Show step if it exists in either token_usage or estimated_cost
        if step_name not in token_usage and step_name not in estimated_cost:
            continue
        
        usage = token_usage.get(step_name, {})
        cost_data = estimated_cost.get(step_name, {})
        breakdown = cost_data.get("breakdown", {})
        
        # Get token counts - handle both old and new formats
        input_tokens = breakdown.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage.get("prompt_tokens", 0) + usage.get("cached_tokens", 0)
        
        thinking_tokens = usage.get("thoughts_tokens", 0)
        
        output_tokens = usage.get("response_tokens", 0)
        
        # Get costs - handle both old and new formats
        input_cost = breakdown.get("input_cost")
        if input_cost is None:
            # Old format: calculate from prompt_cost + cached_cost
            input_cost = breakdown.get("prompt_cost", 0.0) + breakdown.get("cached_cost", 0.0)
        
        # Thinking cost is always calculated from thinking tokens at output rate
        thinking_cost = (thinking_tokens / 1_000_000) * 1.50  # $1.50 per million thinking tokens
        
        output_cost = breakdown.get("output_cost")
        if output_cost is None:
            # Old format: calculate from response_cost only (thinking is separate now)
            output_cost = breakdown.get("response_cost", 0.0)
        
        # If output_cost includes thinking from old format, subtract thinking_cost
        # But only if we got it from old format breakdown
        if breakdown.get("output_cost") is None:
            # In old format, response_cost + thoughts_cost was combined as output_cost
            # But we've already extracted response_cost, so we need to recalculate
            old_output_cost = breakdown.get("response_cost", 0.0) + breakdown.get("thoughts_cost", 0.0)
            if old_output_cost > 0:
                # Recalculate: output = response only, thinking = thoughts only
                output_cost = (output_tokens / 1_000_000) * 1.50
                thinking_cost = (thinking_tokens / 1_000_000) * 1.50
        else:
            # New format already separates them, but we need to ensure thinking is calculated
            if thinking_tokens > 0 and thinking_cost == 0:
                thinking_cost = (thinking_tokens / 1_000_000) * 1.50
        
        total_cost = cost_data.get("cost_usd", 0.0)
        # If total_cost wasn't provided, calculate it
        if total_cost == 0.0:
            total_cost = input_cost + thinking_cost + output_cost
        
        # Format numbers - escape dollar signs, use comma formatting but wrap in texttt
        input_tokens_str = f"{input_tokens:,}"
        thinking_tokens_str = f"{thinking_tokens:,}"
        output_tokens_str = f"{output_tokens:,}"
        input_cost_str = f"${input_cost:.6f}".replace('$', r'\$')
        thinking_cost_str = f"${thinking_cost:.6f}".replace('$', r'\$')
        output_cost_str = f"${output_cost:.6f}".replace('$', r'\$')
        total_cost_str = f"${total_cost:.6f}".replace('$', r'\$')
        
        display_name = step_display_names.get(step_name, step_name.replace("_", " ").title())
        
        # Escape dollar signs and wrap comma-separated numbers in texttt
        latex.append(
            f"{clean_text_for_xelatex(display_name)} & "
            f"\\texttt{{{input_tokens_str}}} & "
            f"\\texttt{{{thinking_tokens_str}}} & "
            f"\\texttt{{{output_tokens_str}}} & "
            f"{input_cost_str} & "
            f"{thinking_cost_str} & "
            f"{output_cost_str} & "
            f"\\textbf{{{total_cost_str}}} \\\\"
        )
        
        total_input_tokens_all += input_tokens
        total_thinking_tokens_all += thinking_tokens
        total_output_tokens_all += output_tokens
        total_input_cost_all += input_cost
        total_thinking_cost_all += thinking_cost
        total_output_cost_all += output_cost
        total_cost_all += total_cost
    
    # Add total row
    total_input_str = f"{total_input_tokens_all:,}"
    total_thinking_str = f"{total_thinking_tokens_all:,}"
    total_output_str = f"{total_output_tokens_all:,}"
    total_input_cost_str = f"${total_input_cost_all:.6f}".replace('$', r'\$')
    total_thinking_cost_str = f"${total_thinking_cost_all:.6f}".replace('$', r'\$')
    total_output_cost_str = f"${total_output_cost_all:.6f}".replace('$', r'\$')
    total_cost_str = f"${total_cost_all:.6f}".replace('$', r'\$')
    
    latex.extend([
        r"\hline",
        f"\\textbf{{Total}} & "
        f"\\textbf{{\\texttt{{{total_input_str}}}}} & "
        f"\\textbf{{\\texttt{{{total_thinking_str}}}}} & "
        f"\\textbf{{\\texttt{{{total_output_str}}}}} & "
        f"\\textbf{{{total_input_cost_str}}} & "
        f"\\textbf{{{total_thinking_cost_str}}} & "
        f"\\textbf{{{total_output_cost_str}}} & "
        f"\\textbf{{{total_cost_str}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\normalsize",  # Return to normal font size
        r"\end{center}",
        r"\vspace{0.5cm}",
    ])
    
    # Add summary by input/output
    total_breakdown = estimated_cost.get("_total_breakdown", {})
    
    # Pre-format cost strings with escaped dollar signs
    if total_breakdown:
        input_cost_summary_str = f"${total_breakdown.get('input_cost', total_input_cost_all):.6f}".replace('$', r'\$')
        thinking_cost_summary_str = f"${total_thinking_cost_all:.6f}".replace('$', r'\$')
        output_cost_summary_str = f"${total_breakdown.get('output_cost', total_output_cost_all):.6f}".replace('$', r'\$')
    else:
        input_cost_summary_str = f"${total_input_cost_all:.6f}".replace('$', r'\$')
        thinking_cost_summary_str = f"${total_thinking_cost_all:.6f}".replace('$', r'\$')
        output_cost_summary_str = f"${total_output_cost_all:.6f}".replace('$', r'\$')
    total_cost_summary_str = f"${total_cost_all:.6f}".replace('$', r'\$')
    
    if total_breakdown:
        latex.extend([
            r"\subsection{Total Cost Summary}",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{|l|r|r|}",
            r"\hline",
            r"\textbf{Category} & \textbf{Tokens} & \textbf{Cost (USD)} \\",
            r"\hline",
            f"Input Tokens & \\texttt{{{total_breakdown.get('input_tokens', total_input_tokens_all):,}}} & {input_cost_summary_str} \\\\",
            f"Thinking Tokens & \\texttt{{{total_thinking_tokens_all:,}}} & {thinking_cost_summary_str} \\\\",
            f"Output Tokens & \\texttt{{{total_breakdown.get('output_tokens', total_output_tokens_all):,}}} & {output_cost_summary_str} \\\\",
            r"\hline",
            f"\\textbf{{Grand Total}} & \\textbf{{\\texttt{{{total_input_tokens_all + total_thinking_tokens_all + total_output_tokens_all:,}}}}} & \\textbf{{{total_cost_summary_str}}} \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\normalsize",
            r"\end{center}",
        ])
    else:
        # Fallback if _total_breakdown not available
        latex.extend([
            r"\subsection{Total Cost Summary}",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{|l|r|r|}",
            r"\hline",
            r"\textbf{Category} & \textbf{Tokens} & \textbf{Cost (USD)} \\",
            r"\hline",
            f"Input Tokens & \\texttt{{{total_input_tokens_all:,}}} & {input_cost_summary_str} \\\\",
            f"Thinking Tokens & \\texttt{{{total_thinking_tokens_all:,}}} & {thinking_cost_summary_str} \\\\",
            f"Output Tokens & \\texttt{{{total_output_tokens_all:,}}} & {output_cost_summary_str} \\\\",
            r"\hline",
            f"\\textbf{{Grand Total}} & \\textbf{{\\texttt{{{total_input_tokens_all + total_thinking_tokens_all + total_output_tokens_all:,}}}}} & \\textbf{{{total_cost_summary_str}}} \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\normalsize",
            r"\end{center}",
        ])
    
    return latex