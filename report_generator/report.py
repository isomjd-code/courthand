"""High-level entrypoints for generating comparison reports."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from .config import DEFAULT_API_KEY, INPUT_FILE, OUTPUT_LATEX_PATH

# Configure logging for report generator
def _configure_report_logger() -> logging.Logger:
    """Configure logging for report generator with both console and file handlers."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("report_generator")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers when re-imported
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(log_dir, "report_generator.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Configure logging
_configure_report_logger()
logger = logging.getLogger("report_generator")
from .sections import (
    generate_case_comparison_section,
    generate_cost_breakdown_section,
    generate_executive_summary,
    generate_field_level_report,
    generate_full_text_section,
    generate_latex_preamble,
    generate_transcription_section,
)
from .similarity import ValidationMetrics
from .sections import _find_low_confidence_names, _find_lines_containing_name, _find_image_file, _extract_and_process_line_image
from .case_matching import find_best_case_matches


def _save_validation_report_data(
    master_data: Dict[str, Any],
    metrics: ValidationMetrics,
    extracted_entities: Optional[Dict[str, Any]],
    source_material: Optional[List[Dict]],
    output_dir: Optional[str],
    input_images_dir: Optional[str]
) -> None:
    """
    Save all validation report data to master_record.json.
    
    Adds a 'validation_report' section containing:
    - Validation metrics and field comparisons
    - Names to check with line image paths
    """
    try:
        # Find master_record.json in the output directory
        if not output_dir:
            logger.warning("[Validation Report] No output directory, cannot save validation data")
            return
        
        master_record_path = os.path.join(output_dir, "master_record.json")
        if not os.path.exists(master_record_path):
            logger.warning(f"[Validation Report] master_record.json not found at {master_record_path}")
            return
        
        # Read existing master_record.json
        with open(master_record_path, "r", encoding="utf-8") as f:
            master_record = json.load(f)
        
        # Build validation report data
        validation_data: Dict[str, Any] = {
            "metrics": metrics.to_dict(),
            "names_to_check": []
        }
        
        # Add names to check data
        if source_material:
            low_confidence_names = _find_low_confidence_names(source_material, output_dir)
            line_images_dir = os.path.join(output_dir, "line_images") if output_dir else None
            
            for name_info in low_confidence_names:
                name = name_info["name"]
                name_type = name_info["type"]
                probability = name_info.get("probability", 0.0)
                original = name_info.get("original", "")
                top_alternatives = name_info.get("top_alternatives", [])
                
                matching_lines = _find_lines_containing_name(name, source_material)
                if original and original != name:
                    # Also search for original
                    original_lines = _find_lines_containing_name(original, source_material)
                    # Merge and deduplicate
                    seen_line_ids = set()
                    merged_lines = []
                    for line in matching_lines + original_lines:
                        line_id = line.get("line_id")
                        if line_id and line_id not in seen_line_ids:
                            seen_line_ids.add(line_id)
                            merged_lines.append(line)
                    matching_lines = merged_lines
                
                name_entry = {
                    "name": name,
                    "type": name_type,
                    "probability": probability,
                    "original": original,
                    "top_alternatives": [
                        {
                            "text": alt.get("text", ""),
                            "probability": alt.get("probability", 0.0)
                        }
                        for alt in top_alternatives
                    ],
                    "matching_lines": []
                }
                
                # Process each matching line
                for line_info in matching_lines[:5]:  # Limit to 5 lines per name
                    filename_img = line_info["filename"]
                    line_id = line_info["line_id"]
                    htr_text = line_info.get("htr_text", "")
                    diplomatic_text = line_info.get("diplomatic_text", "")
                    
                    # Prefer diplomatic transcription
                    display_text = diplomatic_text if diplomatic_text else htr_text
                    
                    line_entry = {
                        "filename": filename_img,
                        "line_id": line_id,
                        "original_file_id": line_info.get("original_file_id"),
                        "htr_text": htr_text,
                        "diplomatic_text": diplomatic_text,
                        "line_image_path": None
                    }
                    
                    # Try to extract and save line image
                    if line_images_dir and line_info.get("kraken_polygon"):
                        # Find the image file
                        search_dirs = []
                        if input_images_dir:
                            search_dirs.append(input_images_dir)
                        if output_dir:
                            parent_dir = os.path.dirname(output_dir)
                            search_dirs.extend([
                                os.path.join(parent_dir, "input_images"),
                                os.path.join(parent_dir, "..", "input_images"),
                            ])
                        
                        image_path = _find_image_file(filename_img, search_dirs)
                        
                        if image_path:
                            # Create safe filename for image
                            safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)[:20]
                            image_filename = f"line_{line_id.replace('L', '')}_{safe_name}.png"
                            output_image_path = os.path.join(line_images_dir, image_filename)
                            
                            # Extract and process line image
                            if _extract_and_process_line_image(
                                image_path,
                                line_info.get("kraken_polygon"),
                                line_id,
                                output_image_path,
                                output_dir=output_dir,
                                image_filename=filename_img,
                                original_file_id=line_info.get("original_file_id")
                            ):
                                if os.path.exists(output_image_path):
                                    # Store relative path from master_record.json location
                                    rel_path = os.path.relpath(output_image_path, output_dir)
                                    line_entry["line_image_path"] = rel_path.replace("\\", "/")
                    
                    name_entry["matching_lines"].append(line_entry)
                
                if name_entry["matching_lines"]:
                    validation_data["names_to_check"].append(name_entry)
        
        # Add validation_report section to master_record
        master_record["validation_report"] = validation_data
        
        # Save updated master_record.json
        with open(master_record_path, "w", encoding="utf-8") as f:
            json.dump(master_record, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Validation Report] Saved validation data to {master_record_path}")
        
    except Exception as e:
        logger.error(f"[Validation Report] Error saving validation data: {e}", exc_info=True)


def _extract_ground_truth_cases(ground_truth_data: List[Any]) -> List[Dict[str, Any]]:
    """Extract and normalize ground truth cases from various formats."""
    gt_cases = []
    
    if not ground_truth_data:
        return gt_cases
    
    # Handle different formats
    for item in ground_truth_data:
        if item is None:
            continue
        elif isinstance(item, list):
            # Nested list format: [[dict1, dict2, ...]]
            gt_cases.extend([c for c in item if isinstance(c, dict)])
        elif isinstance(item, dict):
            # Direct dict format
            gt_cases.append(item)
    
    return gt_cases


def generate_latex_report_for_match(
    gt_case: Dict[str, Any],
    ai_case: Dict[str, Any],
    master_data: Dict[str, Any],
    filename: Optional[str] = None,
    api_key: Optional[str] = None,
    match_score: Optional[float] = None,
    match_index: Optional[int] = None
) -> ValidationMetrics:
    """
    Generate a LaTeX report for a single case match.
    
    This is the core report generation logic extracted from generate_latex_report.
    """
    metrics = ValidationMetrics()
    ai_ref = master_data.get("legal_index", {}).get("TblReference", {})
    meta = master_data.get("case_metadata", {})
    
    # Determine output directory
    output_dir_for_validation = None
    if filename:
        abs_filename = os.path.abspath(filename)
        output_dir_for_validation = os.path.dirname(abs_filename) if os.path.dirname(abs_filename) else os.getcwd()
    else:
        for search_dir in [os.getcwd(), os.path.dirname(os.getcwd())]:
            master_record_path = os.path.join(search_dir, "master_record.json")
            if os.path.exists(master_record_path):
                output_dir_for_validation = search_dir
                break
    
    # Generate filename based on roll and rotulus if not provided
    if filename is None:
        roll_number = meta.get("roll_number", "unknown")
        rotulus_number = meta.get("rotulus_number", "unknown")
        base_filename = f"comparison_report_CP40-{roll_number}_{rotulus_number}"
        if match_index is not None:
            base_filename += f"_match{match_index + 1}"
        filename = f"{base_filename}.tex"
        if output_dir_for_validation:
            filename = os.path.join(output_dir_for_validation, filename)
    
    # Check if the file already exists - if so, check if we need to regenerate
    if os.path.exists(filename):
        # If we have ground truth data but the existing file shows "Missing Ground Truth Data",
        # we should regenerate to include the GT data
        should_regenerate = False
        if gt_case and gt_case.get("TblCase"):
            # We have GT data - check if the existing file has the missing GT warning
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                    if "Warning: Missing Ground Truth Data" in existing_content or "No ground truth data found" in existing_content:
                        logger.info(f"[Report] GT data now available but file was generated without it. Regenerating: {filename}")
                        should_regenerate = True
                    # Also check for old problematic hyperlink code that causes LaTeX errors
                    if "\\hyperlink{sec:field-comparison}" in existing_content and "\\Huge\\hyperlink" in existing_content:
                        logger.info(f"[Report] Detected old hyperlink code that causes LaTeX errors. Regenerating: {filename}")
                        should_regenerate = True
            except Exception as e:
                logger.warning(f"[Report] Could not read existing file to check for GT data: {e}")
        
        if not should_regenerate:
            logger.info(f"[Report] LaTeX file already exists, skipping generation: {filename}")
            # Return empty metrics since we're not generating
            return ValidationMetrics()
    
    latex = generate_latex_preamble(meta)
    
    # Add match score info if available
    if match_score is not None:
        latex.extend([
            r"\begin{center}\begin{tcolorbox}[colback=blue!5!white, colframe=blue!75!black, width=0.95\textwidth]",
            f"\\textbf{{Match Score: {match_score*100:.1f}\\%}} - This report compares one of multiple cases found for this roll/rotulus.",
            r"\end{tcolorbox}\end{center}",
        ])
    
    latex.extend(
        [
            r"\begin{document}",
            r"\maketitle",
            r"\thispagestyle{fancy}",
            r"\begin{center}\begin{tcolorbox}[colback=white, colframe=gray!50, width=0.95\textwidth, halign=center,valign=center]",
            r"\sffamily\matchicon\ \textbf{Exact Match} \quad \mismatchicon\ \textbf{Mismatch} \quad \gtlabel\ Ground Truth \quad \ailabel\ AI Extraction",
            r"\end{tcolorbox}\end{center}",
        ]
    )

    extracted_entities = master_data.get("extracted_entities", {})
    source_material = master_data.get("source_material", [])
    case_content = generate_case_comparison_section(
        gt_case, ai_case, ai_ref, metrics, api_key=api_key, extracted_entities=extracted_entities
    )
    
    # Determine output directory and input images directory
    if filename:
        abs_filename = os.path.abspath(filename)
        output_dir = os.path.dirname(abs_filename) if os.path.dirname(abs_filename) else os.getcwd()
    else:
        output_dir = output_dir_for_validation if output_dir_for_validation else os.getcwd()
    
    input_images_dir = None
    search_paths = []
    if output_dir:
        current = output_dir
        for _ in range(5):
            parent = os.path.dirname(current)
            if parent == current:
                break
            img_path = os.path.join(parent, "input_images")
            if os.path.exists(img_path):
                search_paths.append(img_path)
            current = parent
    
    search_paths.extend([
        "input_images",
        "../input_images",
        "../../input_images",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "input_images"),
    ])
    
    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            input_images_dir = abs_path
            break
    
    logger.info("[Report] Generating executive summary...")
    summary_content = generate_executive_summary(
        metrics,
        extracted_entities=extracted_entities,
        source_material=source_material,
        input_images_dir=input_images_dir,
        output_dir=output_dir
    )
    logger.info("[Report] Executive summary generated")

    latex.extend(summary_content)
    logger.info("[Report] Generating cost breakdown section...")
    latex.extend(generate_cost_breakdown_section(master_data))
    latex.extend(case_content)
    logger.info("[Report] Generating transcription section...")
    latex.extend(generate_transcription_section(
        master_data.get("source_material", []),
        extracted_entities=extracted_entities,
        output_dir=output_dir
    ))
    logger.info("[Report] Generating full text section...")
    latex.extend(generate_full_text_section(master_data))
    logger.info("[Report] Generating field level report...")
    latex.extend(generate_field_level_report(metrics))
    latex.append(r"\end{document}")

    logger.info(f"[Report] Writing LaTeX file to {filename}...")
    try:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write("\n".join(latex))
        logger.info(f"[Report] LaTeX file written successfully")
    except IOError as e:
        error_msg = f"ERROR: Failed to write LaTeX file to {filename}: {e}"
        print(f"\n{'='*80}", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print(f"Make sure the directory exists and is writable: {os.path.dirname(os.path.abspath(filename))}", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        logger.error(error_msg, exc_info=True)
        raise

    summary = metrics.get_summary()
    print(f"\n{'='*60}\n  VALIDATION REPORT GENERATED: {filename}\n{'='*60}")
    print(f"  Total Fields Compared: {summary['total_fields']}\n  Exact Matches: {summary['exact_matches']}")
    print(f"  Overall Accuracy: {summary['overall_accuracy']:.1f}%\n  Average Similarity: {summary['avg_similarity']:.1f}%")
    if match_score is not None:
        print(f"  Case Match Score: {match_score*100:.1f}%")
    print(f"\n  Compile with: xelatex {filename}\n{'='*60}\n")

    return metrics


def generate_latex_report(master_data: Dict[str, Any], filename: Optional[str] = None, api_key: Optional[str] = None) -> ValidationMetrics:
    """
    Generate the complete LaTeX validation report(s).
    
    If multiple ground truth cases or AI-extracted cases are found, this function will:
    1. Match cases using similarity scoring
    2. Generate separate PDF reports for each match
    
    Returns the metrics from the first (best) match, or the single match if only one exists.
    """
    # Fallback to environment variable if api_key is not provided
    if not api_key or not api_key.strip():
        api_key = os.getenv("GEMINI_API_KEY") or DEFAULT_API_KEY
        if not api_key or not api_key.strip():
            error_msg = (
                "CRITICAL: Gemini API key is REQUIRED for postea and pleadings matching. "
                "Please set GEMINI_API_KEY environment variable."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    try:
        # Extract ground truth cases
        ground_truth_data = master_data.get("ground_truth_from_db", [])
        logger.info(f"[Report] Found {len(ground_truth_data)} item(s) in ground_truth_from_db")
        gt_cases = _extract_ground_truth_cases(ground_truth_data)
        logger.info(f"[Report] Extracted {len(gt_cases)} GT case(s) after initial extraction")
        
        # Extract AI cases
        ai_cases = master_data.get("legal_index", {}).get("Cases", [])
        # Filter out empty cases - check for non-empty dicts with actual content
        ai_cases = [
            c for c in ai_cases 
            if c and isinstance(c, dict) and (c.get("TblCase") or c.get("Agents") or len(c) > 0)
        ]
        
        # Handle empty cases
        if not ai_cases:
            # Fallback to treating the whole legal_index as a case
            legal_index = master_data.get("legal_index", {})
            if legal_index and (legal_index.get("TblCase") or legal_index.get("Cases")):
                ai_cases = [legal_index]
        
        # Filter out empty GT cases too
        gt_cases_before_filter = len(gt_cases)
        gt_cases = [
            c for c in gt_cases 
            if c and isinstance(c, dict) and (c.get("TblCase") or c.get("Agents") or len(c) > 0)
        ]
        if gt_cases_before_filter != len(gt_cases):
            logger.warning(f"[Report] Filtered out {gt_cases_before_filter - len(gt_cases)} empty GT case(s)")
        logger.info(f"[Report] Final GT cases after filtering: {len(gt_cases)}")
        
        # Only treat as "multiple cases" if there are ACTUALLY 2+ valid cases on at least one side
        # If there's only 1 GT case and 1 AI case, treat as single case report (no _match suffix)
        has_multiple_gt = len(gt_cases) >= 2
        has_multiple_ai = len(ai_cases) >= 2
        
        # Only generate match files if there are truly multiple cases (2+ on at least one side)
        if has_multiple_gt or has_multiple_ai:
            # Multiple cases found - use case matching and generate separate reports
            logger.info(f"[Report] Multiple cases detected: {len(gt_cases)} GT case(s), {len(ai_cases)} AI case(s)")
            
            if not gt_cases:
                logger.warning("[Report] No ground truth cases found, but multiple AI cases exist. Using first AI case.")
                gt_cases = [{}]
            if not ai_cases:
                logger.warning("[Report] No AI cases found, but multiple GT cases exist. Using first GT case.")
                ai_cases = [{}]
            
            # Find best matches
            matches = find_best_case_matches(gt_cases, ai_cases, api_key=api_key, min_score=0.3)
            
            if not matches:
                logger.warning("[Report] No good matches found between GT and AI cases. Using first GT and first AI case.")
                matches = [(gt_cases[0] if gt_cases else {}, ai_cases[0] if ai_cases else {}, 0.0)]
            
            # Generate separate reports for each match
            all_metrics = []
            meta = master_data.get("case_metadata", {})
            roll_number = meta.get("roll_number", "unknown")
            rotulus_number = meta.get("rotulus_number", "unknown")
            
            for idx, (gt_case, ai_case, match_score) in enumerate(matches):
                logger.info(f"[Report] Generating report {idx + 1}/{len(matches)} (match score: {match_score:.3f})")
                
                # Generate filename for this match
                match_filename = None
                if filename:
                    # If filename was provided, modify it to include match index
                    base_path, ext = os.path.splitext(filename)
                    match_filename = f"{base_path}_match{idx + 1}{ext}"
                else:
                    # Generate new filename
                    base_filename = f"comparison_report_CP40-{roll_number}_{rotulus_number}_match{idx + 1}.tex"
                    # Try to find output directory
                    output_dir = None
                    for search_dir in [os.getcwd(), os.path.dirname(os.getcwd())]:
                        master_record_path = os.path.join(search_dir, "master_record.json")
                        if os.path.exists(master_record_path):
                            output_dir = search_dir
                            break
                    if output_dir:
                        match_filename = os.path.join(output_dir, base_filename)
                    else:
                        match_filename = base_filename
                
                # Skip if file already exists
                if os.path.exists(match_filename):
                    logger.info(f"[Report] Report file already exists, skipping: {match_filename}")
                    # Create empty metrics for consistency
                    all_metrics.append(ValidationMetrics())
                    continue
                
                metrics = generate_latex_report_for_match(
                    gt_case=gt_case,
                    ai_case=ai_case,
                    master_data=master_data,
                    filename=match_filename,
                    api_key=api_key,
                    match_score=match_score,
                    match_index=idx
                )
                all_metrics.append(metrics)
            
            logger.info(f"[Report] Generated {len(matches)} separate reports")
            
            # Return metrics from the best match
            return all_metrics[0] if all_metrics else ValidationMetrics()
        
        else:
            # Single case - delegate to the match function
            gt_case = gt_cases[0] if gt_cases else {}
            ai_case = ai_cases[0] if ai_cases else {}
            
            if not ai_case:
                ai_case = master_data.get("legal_index", {})
            
            return generate_latex_report_for_match(
                gt_case=gt_case,
                ai_case=ai_case,
                master_data=master_data,
                filename=filename,
                api_key=api_key,
                match_score=None,
                match_index=None
            )
    except Exception as e:
        error_msg = f"ERROR: Failed to generate LaTeX report: {e}"
        print(f"\n{'='*80}", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        logger.error(error_msg, exc_info=True)
        print("Waiting 5 seconds for you to see this error...")
        time.sleep(5)
        raise


def main() -> None:
    """CLI entrypoint mirroring the legacy script."""
    try:
        target_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
        if not os.path.exists(target_file):
            error_msg = f"ERROR: Master Record not found at {target_file}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            logger.error(error_msg)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            sys.exit(1)

        try:
            with open(target_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as e:
            error_msg = f"ERROR: Failed to parse JSON from {target_file}: {e}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            logger.error(error_msg, exc_info=True)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            sys.exit(1)
        except Exception as e:
            error_msg = f"ERROR: Failed to read {target_file}: {e}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            logger.error(error_msg, exc_info=True)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            sys.exit(1)

        # Get API key from environment or config, with proper validation
        api_key = os.getenv("GEMINI_API_KEY") or DEFAULT_API_KEY
        if not api_key or not api_key.strip():
            error_msg = (
                "ERROR: GEMINI_API_KEY environment variable is REQUIRED for postea and pleadings matching.\n"
                "Please set GEMINI_API_KEY environment variable with a valid Gemini API key.\n"
                "You can set it by running: export GEMINI_API_KEY=your_api_key_here\n"
                "Or add it to your .env file if using python-dotenv."
            )
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            logger.error(error_msg)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            sys.exit(1)
        
        # Determine output directory from master_record.json location
        master_record_dir = os.path.dirname(os.path.abspath(target_file))
        
        # Generate filename will be created automatically from case_metadata
        # Pass None for filename so it's generated, but ensure output_dir is set correctly
        try:
            generate_latex_report(data, filename=None, api_key=api_key)
        except Exception as e:
            error_msg = f"ERROR: Failed to generate LaTeX report: {e}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            logger.error(error_msg, exc_info=True)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        error_msg = f"ERROR: Unexpected error generating report: {e}"
        print(f"\n{'='*80}", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        logger.error(error_msg, exc_info=True)
        print("Waiting 5 seconds for you to see this error...")
        time.sleep(5)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

