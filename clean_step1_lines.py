"""
Step 1 JSON Line Cleaner with Ground Truth Surname Correction

Opens step1.json files and master_record.json files in subdirectories.
Validates and cleans transcription lines according to get_step1_instruction_text(),
then uses fuzzy matching to correct surnames based on ground truth from validation_report.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from difflib import SequenceMatcher

# Allowed characters from get_step1_instruction_text() (Section 7)
ALLOWED_CHARS: Set[str] = set([
    # Uppercase Latin
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    # Lowercase Latin
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # Punctuation
    ".", ",", ";", "¶", "·",
    # Medieval special characters
    "⁊",  # U+204A Tironian Et
    "ſ",  # U+017F Long S
    "ħ",  # U+0127 H with Stroke
    "ł",  # U+0142 L with Stroke
    "đ",  # U+0111 D with Stroke
    "ꝛ",  # U+A75B R Rotunda
    "ꝝ",  # U+A75D Rum Rotunda
    "ꝫ",  # U+A76B Et/Us/Que
    "ꝑ",  # U+A751 P with Stroke (per)
    "ꝓ",  # U+A753 P with Flourish (pro)
    "ꝙ",  # U+A759 Q with Diagonal
    "ꝰ",  # U+A770 Us Modifier
    "ꝭ",  # U+A76D Is Modifier
    "ı",  # U+0131 Dotless I
    "þ",  # U+00FE Thorn
    "ȝ",  # U+021D Yogh
    "÷",  # U+00F7 Est Marker
    # Combining diacritics
    "\u0305",  # ̅ Combining Overline
    "\u036C",  # ͬ Combining Superscript R
    # Superscript vowels
    "ᵃ", "ᵉ", "ⁱ", "ᵒ", "ᵘ",
    # Whitespace (essential for text)
    " ",
])

# Character replacements for known forbidden characters
REPLACEMENTS: Dict[str, str] = {
    "'": "",      # Apostrophe - omit
    "'": "",      # Right single quote - omit
    "'": "",      # Left single quote - omit
    "—": "",      # Em dash - omit
    "–": "",      # En dash - omit
    "-": "",      # Regular hyphen - omit (not in allowed list)
    ":": ".",     # Colon -> period
    "&": "⁊",     # Ampersand -> Tironian Et
    
    # Common substitutions for characters that might appear incorrectly
    "ē": "e\u0305",  # e-macron -> e + combining overline
    "ā": "a\u0305",  # a-macron -> a + combining overline
    "ō": "o\u0305",  # o-macron -> o + combining overline
    "ī": "i\u0305",  # i-macron -> i + combining overline  
    "ū": "u\u0305",  # u-macron -> u + combining overline
    "n̄": "n\u0305",  # n with macron
    
    # Normalize some Unicode variants
    "ꝑ": "ꝑ",  # Ensure correct per symbol
    "ꝓ": "ꝓ",  # Ensure correct pro symbol
}

# Characters after which R Rotunda (ꝛ) should be used (rounded letters)
ROUNDED_LETTERS = set("obpdhOBPDH")

# Fuzzy matching threshold for surname correction
SURNAME_MATCH_THRESHOLD = 0.70


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for fuzzy matching by:
    - Converting to lowercase
    - Replacing medieval special chars with modern equivalents
    - Removing diacritics
    """
    # Map medieval chars to modern for comparison purposes
    medieval_to_modern = {
        'ſ': 's', 'ħ': 'h', 'ł': 'l', 'đ': 'd',
        'ꝛ': 'r', 'ꝝ': 'rum', 'ꝫ': 'us',
        'ꝑ': 'per', 'ꝓ': 'pro', 'ꝙ': 'q',
        'þ': 'th', 'ȝ': 'gh',
    }
    result = text.lower()
    for old, new in medieval_to_modern.items():
        result = result.replace(old, new)
    # Remove combining diacritics
    result = result.replace('\u0305', '').replace('\u036C', '')
    return result


def fuzzy_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, s1, s2).ratio()


def extract_gt_surnames(master_record: Dict) -> List[Dict]:
    """
    Extract ground truth surnames from master_record.json validation_report.
    
    Returns list of dicts with:
    - full_name: e.g., "William Warrene"
    - surname: e.g., "Warrene"  
    - ai_surname: what the AI extracted (for potential fuzzy matching)
    """
    surnames = []
    
    validation_report = master_record.get("validation_report", {})
    field_comparisons = validation_report.get("field_comparisons", [])
    
    for field in field_comparisons:
        if field.get("field_name") == "Agent Name":
            gt_value = field.get("gt_value", "")
            ai_value = field.get("ai_value", "")
            
            # Extract surname (second word, assuming "FirstName Surname" format)
            gt_parts = gt_value.split()
            ai_parts = ai_value.split()
            
            if len(gt_parts) >= 2:
                gt_surname = gt_parts[-1]  # Last word is surname
                ai_surname = ai_parts[-1] if len(ai_parts) >= 2 else ""
                
                surnames.append({
                    "full_name": gt_value,
                    "gt_surname": gt_surname,
                    "ai_surname": ai_surname,
                    "normalized_gt": normalize_for_matching(gt_surname),
                    "normalized_ai": normalize_for_matching(ai_surname),
                })
    
    return surnames


def find_surname_in_text(text: str, gt_surnames: List[Dict], threshold: float = SURNAME_MATCH_THRESHOLD) -> List[Dict]:
    """
    Find potential surname matches in text using fuzzy matching.
    
    Returns list of matches with position, original text, and suggested replacement.
    """
    matches = []
    
    # Split text into words while preserving positions
    words_with_positions = []
    current_pos = 0
    for word in re.split(r'(\s+)', text):
        if word and not word.isspace():
            words_with_positions.append({
                "word": word,
                "start": current_pos,
                "end": current_pos + len(word)
            })
        current_pos += len(word)
    
    for word_info in words_with_positions:
        word = word_info["word"]
        normalized_word = normalize_for_matching(word)
        
        # Skip very short words (unlikely to be surnames)
        if len(normalized_word) < 3:
            continue
        
        for surname_info in gt_surnames:
            gt_surname = surname_info["gt_surname"]
            normalized_gt = surname_info["normalized_gt"]
            ai_surname = surname_info.get("ai_surname", "")
            normalized_ai = surname_info.get("normalized_ai", "")
            
            # Check similarity with GT surname
            similarity_gt = fuzzy_ratio(normalized_word, normalized_gt)
            
            # Also check against what AI extracted (might be in text)
            similarity_ai = fuzzy_ratio(normalized_word, normalized_ai) if normalized_ai else 0
            
            # If word matches AI's wrong version but GT is different, suggest correction
            if similarity_ai >= threshold and normalized_gt != normalized_ai:
                # Found a word that matches what AI got wrong - correct it to GT
                matches.append({
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "original": word,
                    "replacement": gt_surname,
                    "similarity_to_ai": similarity_ai,
                    "reason": f"matches_ai_error_{ai_surname}_should_be_{gt_surname}"
                })
            elif similarity_gt >= threshold and similarity_gt < 0.98:
                # Close match to GT but not exact - might need minor correction
                # Only suggest if not already correct
                if normalized_word != normalized_gt:
                    matches.append({
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "original": word,
                        "replacement": gt_surname,
                        "similarity_to_gt": similarity_gt,
                        "reason": f"fuzzy_match_to_gt_{gt_surname}"
                    })
    
    # Remove duplicates (same position)
    seen_positions = set()
    unique_matches = []
    for m in matches:
        pos_key = (m["start"], m["end"])
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            unique_matches.append(m)
    
    return unique_matches


def apply_surname_corrections(text: str, corrections: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Apply surname corrections to text.
    
    Returns (corrected_text, list of changes made)
    """
    if not corrections:
        return text, []
    
    # Sort by position (descending) to replace from end to start
    sorted_corrections = sorted(corrections, key=lambda x: x["start"], reverse=True)
    
    result = text
    changes = []
    
    for corr in sorted_corrections:
        original = corr["original"]
        replacement = corr["replacement"]
        start = corr["start"]
        end = corr["end"]
        
        # Preserve case pattern from original
        if original and replacement and original[0].isupper() and replacement[0].islower():
            replacement = replacement[0].upper() + replacement[1:]
        
        result = result[:start] + replacement + result[end:]
        changes.append({
            "position": start,
            "original": original,
            "replacement": replacement,
            "action": "surname_correction",
            "reason": corr.get("reason", "")
        })
    
    return result, changes


def normalize_r_rotunda(text: str) -> Tuple[str, List[Dict]]:
    """
    Normalize r/ꝛ usage according to manuscript conventions.
    
    Rules from get_step1_instruction_text() Section 3E:
    - R Rotunda (ꝛ) should ONLY appear after rounded letters: o, b, p, d, h
    - Normal 'r' should appear after all other letters (e, i, u, a, c, t, etc.)
    
    Returns:
        Tuple of (normalized_text, list of changes made)
    """
    if not text:
        return text, []
    
    changes = []
    result = list(text)  # Convert to list for in-place modification
    
    for i, char in enumerate(text):
        if i == 0:
            # First character - if it's ꝛ at start of word, convert to r
            if char == 'ꝛ':
                result[i] = 'r'
                changes.append({
                    "position": i,
                    "original": "ꝛ",
                    "replacement": "r",
                    "reason": "r_rotunda_at_word_start"
                })
            continue
        
        prev_char = text[i - 1]
        prev_is_rounded = prev_char in ROUNDED_LETTERS
        
        if char == 'r':
            # Check if 'r' should be ꝛ (after rounded letter)
            if prev_is_rounded:
                result[i] = 'ꝛ'
                changes.append({
                    "position": i,
                    "original": "r",
                    "replacement": "ꝛ",
                    "reason": f"r_after_rounded_letter_{prev_char}"
                })
        elif char == 'ꝛ':
            # Check if ꝛ should be 'r' (NOT after rounded letter)
            if not prev_is_rounded:
                result[i] = 'r'
                changes.append({
                    "position": i,
                    "original": "ꝛ",
                    "replacement": "r",
                    "reason": f"r_rotunda_after_non_rounded_letter_{prev_char}"
                })
    
    return "".join(result), changes


def clean_transcription(text: str, gt_surnames: Optional[List[Dict]] = None) -> Tuple[str, List[Dict]]:
    """
    Clean a transcription line with full pipeline:
    1. Character replacement/removal
    2. R/R-rotunda normalization
    3. Surname correction (if gt_surnames provided)
    
    Returns:
        Tuple of (cleaned_text, list of changes made)
    """
    if not text:
        return text, []
    
    all_changes = []
    result = []
    
    # PHASE 1: Character replacement/removal
    for i, char in enumerate(text):
        if char in ALLOWED_CHARS:
            result.append(char)
        elif char in REPLACEMENTS:
            replacement = REPLACEMENTS[char]
            result.append(replacement)
            all_changes.append({
                "position": i,
                "original": char,
                "original_codepoint": f"U+{ord(char):04X}",
                "replacement": replacement if replacement else "(removed)",
                "action": "char_replacement"
            })
        else:
            # Unknown character - remove it
            all_changes.append({
                "position": i,
                "original": char,
                "original_codepoint": f"U+{ord(char):04X}",
                "replacement": "(removed)",
                "action": "removed_unknown"
            })
    
    intermediate_text = "".join(result)
    
    # PHASE 2: R/R-rotunda normalization
    normalized_text, r_changes = normalize_r_rotunda(intermediate_text)
    all_changes.extend(r_changes)
    
    # PHASE 3: Surname correction (if GT surnames provided)
    if gt_surnames:
        surname_matches = find_surname_in_text(normalized_text, gt_surnames)
        final_text, surname_changes = apply_surname_corrections(normalized_text, surname_matches)
        all_changes.extend(surname_changes)
    else:
        final_text = normalized_text
    
    return final_text, all_changes


def process_directory(dir_path: Path, dry_run: bool = False) -> Dict:
    """
    Process a single directory containing step1.json and master_record.json files.
    """
    report = {"dir": str(dir_path), "files_processed": 0, "gt_surnames": []}
    
    # Load master_record.json to get ground truth surnames
    master_record_path = dir_path / "master_record.json"
    gt_surnames = []
    
    if master_record_path.exists():
        try:
            with open(master_record_path, 'r', encoding='utf-8') as f:
                master_record = json.load(f)
            gt_surnames = extract_gt_surnames(master_record)
            report["gt_surnames"] = [s["gt_surname"] for s in gt_surnames]
            if gt_surnames:
                print(f"  Found {len(gt_surnames)} GT surnames: {report['gt_surnames']}")
        except Exception as e:
            print(f"  Warning: Could not load master_record.json: {e}")
    
    # Process all step1.json files in this directory
    step1_files = list(dir_path.glob("*step1.json"))
    
    if not step1_files:
        return report
    
    for step1_path in step1_files:
        print(f"  Processing: {step1_path.name}")
        
        try:
            with open(step1_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lines = data.get("lines", [])
            cleaned_lines = []
            total_changes = 0
            
            for line in lines:
                line_id = line.get("id", "unknown")
                original_text = line.get("transcription", "")
                
                # Clean with surname correction
                cleaned_text, changes = clean_transcription(original_text, gt_surnames)
                
                cleaned_line = {
                    "id": line_id,
                    "transcription": cleaned_text
                }
                
                if changes:
                    cleaned_line["changes_made"] = changes
                    cleaned_line["original_transcription"] = original_text
                    total_changes += len(changes)
                
                cleaned_lines.append(cleaned_line)
            
            data["cleaned_lines"] = cleaned_lines
            data["cleaning_metadata"] = {
                "total_lines": len(lines),
                "total_changes": total_changes,
                "gt_surnames_used": [s["gt_surname"] for s in gt_surnames]
            }
            
            if not dry_run:
                with open(step1_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"    ✓ Saved with {total_changes} changes")
            else:
                print(f"    [DRY RUN] Would save with {total_changes} changes")
            
            report["files_processed"] += 1
            
        except Exception as e:
            print(f"    ERROR: {e}")
    
    return report


def main():
    """Main entry point."""
    # Configure the base directory - adjust as needed
    # Use Linux path when running from WSL, Windows path when running from Windows
    BASE_DIR = Path("/home/qj/projects/latin_bho/cp40_processing/output")
    
    # Set to True to preview changes without modifying files
    DRY_RUN = False
    
    print("=" * 70)
    print("Step 1 JSON Line Cleaner with Ground Truth Surname Correction")
    print("=" * 70)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Allowed characters: {len(ALLOWED_CHARS)} total")
    print(f"Fuzzy match threshold: {SURNAME_MATCH_THRESHOLD}")
    
    if DRY_RUN:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")
    
    # Find all subdirectories
    try:
        subdirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"ERROR: Could not access base directory: {e}")
        return
    
    print(f"\nFound {len(subdirs)} subdirectories to process\n")
    
    summary = {
        "total_dirs": len(subdirs),
        "dirs_processed": 0,
        "total_files": 0
    }
    
    for subdir in sorted(subdirs):
        print(f"\n{'='*50}")
        print(f"Directory: {subdir.name}")
        report = process_directory(subdir, dry_run=DRY_RUN)
        if report["files_processed"] > 0:
            summary["dirs_processed"] += 1
            summary["total_files"] += report["files_processed"]
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Directories processed: {summary['dirs_processed']}/{summary['total_dirs']}")
    print(f"Total step1.json files processed: {summary['total_files']}")
    
    if DRY_RUN:
        print("\n*** This was a dry run. Set DRY_RUN = False to apply changes ***")


if __name__ == "__main__":
    main()

