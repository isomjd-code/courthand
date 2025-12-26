"""General helper functions."""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def get_cp40_info(roll_number: str) -> Optional[Dict[str, str]]:
    """
    Get archival metadata for a CP40 roll number.

    Calculates calendar year, term, and regnal year based on roll number.
    Supports rolls 555-650 covering the reigns of Henry IV, Henry V, and early Henry VI.

    Args:
        roll_number: Roll number as string (e.g., "565", "562").

    Returns:
        Dictionary with keys:
        - "Roll": Roll number (as integer)
        - "Calendar Year": Year (e.g., 1400)
        - "Term": "Hilary", "Easter", "Trinity", or "Michaelmas"
        - "Regnal Year": Formatted string (e.g., "1 Henry IV")
        Returns None if roll number is outside supported range or invalid.
    """
    terms = ["Hilary", "Easter", "Trinity", "Michaelmas"]
    try:
        roll_number = int(roll_number)
    except (ValueError, TypeError):
        return None

    if 555 <= roll_number <= 608:
        if roll_number == 555:
            return {
                "Roll": roll_number,
                "Calendar Year": 1399,
                "Term": "Michaelmas",
                "Regnal Year": "1 Henry IV",
            }

        offset = roll_number - 556
        year = 1400 + (offset // 4)
        term_name = terms[offset % 4]

        regnal_num = year - 1399
        if term_name == "Michaelmas":
            regnal_num += 1

        return {
            "Roll": roll_number,
            "Calendar Year": year,
            "Term": term_name,
            "Regnal Year": f"{regnal_num} Henry IV",
        }

    if 609 <= roll_number <= 650:
        offset = roll_number - 609
        current_global_index = 1 + offset
        year = 1413 + (current_global_index // 4)
        term_name = terms[current_global_index % 4]
        monarch = "Henry V"
        regnal_num = year - 1413
        if term_name != "Hilary":
            regnal_num += 1

        if year == 1422 and term_name == "Michaelmas":
            monarch = "Henry VI"
            regnal_num = 1
        elif year > 1422:
            monarch = "Henry VI"
            regnal_num = year - 1422 + (1 if term_name == "Michaelmas" else 0)

        return {
            "Roll": roll_number,
            "Calendar Year": year,
            "Term": term_name,
            "Regnal Year": f"{regnal_num} {monarch}",
        }

    return None


def repair_json_string(json_str: str) -> str:
    """
    Attempt to repair common JSON malformations.
    
    Fixes issues like:
    - Closing parentheses ')' instead of closing braces '}'
    - Missing closing brackets/braces
    - Trailing commas before closing brackets/braces
    
    Args:
        json_str: Potentially malformed JSON string.
        
    Returns:
        Repaired JSON string (may still be invalid if issues are too severe).
    """
    if not json_str:
        return json_str
    
    # Count braces and brackets to detect imbalances
    brace_count = json_str.count('{') - json_str.count('}')
    bracket_count = json_str.count('[') - json_str.count(']')
    
    result = json_str
    
    # Fix closing parentheses that should be closing braces
    # This handles cases where the JSON ends with ) instead of }
    if brace_count > 0:
        # Find the last closing parenthesis that might be a misplaced brace
        # Work backwards from the end
        lines = result.split('\n')
        braces_fixed = 0
        
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            stripped = line.rstrip()
            
            # If line ends with ) and we still need to fix braces
            if stripped.endswith(')') and braces_fixed < brace_count:
                # Replace the last ) with }
                lines[i] = line[:-1] + '}'
                braces_fixed += 1
                if braces_fixed >= brace_count:
                    break
        
        if braces_fixed > 0:
            result = '\n'.join(lines)
            # Recalculate counts after fixing
            brace_count = result.count('{') - result.count('}')
            bracket_count = result.count('[') - result.count(']')
    
    # Remove trailing commas before closing brackets/braces
    result = re.sub(r',(\s*[}\]])', r'\1', result)
    
    # Add missing closing brackets/braces at the end
    result = result.rstrip()
    # Add braces first (inner structures), then brackets (outer structures)
    if brace_count > 0:
        result += '\n' + '}' * brace_count
    if bracket_count > 0:
        result += '\n' + ']' * bracket_count
    
    return result


def clean_json_string(text: str) -> str:
    """
    Extract JSON content from text that may contain markdown code blocks.

    Removes markdown formatting (```json ... ``` or ``` ... ```) and extracts
    only the JSON content. Also handles cases where JSON is embedded in other text.
    Prefers JSON objects over arrays to avoid extracting nested arrays.

    Args:
        text: Raw text that may contain JSON in markdown code blocks or plain text.

    Returns:
        Clean JSON string with markdown removed. Returns empty string if input is empty.
    """
    if not text:
        return ""

    # First try to extract from markdown code blocks
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Check if we have a JSON object first (prefer objects to avoid extracting nested arrays)
    obj_start = text.find("{")
    arr_start = text.find("[")
    
    # If we have both, prefer object if it comes first or if array is inside object
    if obj_start != -1:
        # Find the matching closing brace by counting braces
        brace_count = 0
        for i in range(obj_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete object
                    return text[obj_start : i + 1]
    
    # If no object found, try to find JSON array (for cases where we expect arrays)
    if arr_start != -1:
        # Find the matching closing bracket
        bracket_count = 0
        for i in range(arr_start, len(text)):
            if text[i] == '[':
                bracket_count += 1
            elif text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return text[arr_start : i + 1]

    # Fallback: try to find any JSON structure
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()

