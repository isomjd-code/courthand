"""Text cleaning and formatting helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union


def clean_text_for_xelatex(text: Any) -> str:
    """Escape LaTeX reserved characters while preserving Unicode content."""
    if text is None:
        return ""
    if not isinstance(text, str):
        if isinstance(text, list):
            return ", ".join(clean_text_for_xelatex(item) for item in text)
        return str(text)

    text = text.replace("\\", r"\textbackslash{}")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def normalize_string(value: Any) -> str:
    """Normalize a string for comparison."""
    if not value:
        return ""
    if isinstance(value, list):
        value = " ".join(value)
    return "".join(str(value).split()).lower()


def get_person_name(agent_obj: Dict) -> str:
    """Extract display name from TblName structure."""
    if not agent_obj:
        return "Unknown"
    name_node = agent_obj.get("TblName", {})
    return f"{name_node.get('Christian_name', '')} {name_node.get('Surname', '')}".strip()


def soundex(name: str) -> str:
    """
    Calculate Soundex code for a name.
    
    Soundex is a phonetic algorithm that indexes names by sound.
    Names with the same soundex code are considered phonetically similar.
    
    Args:
        name: Name string to encode
        
    Returns:
        Soundex code (4 characters: letter + 3 digits)
    """
    if not name:
        return ""
    
    # Convert to uppercase and remove non-alphabetic characters
    name = re.sub(r'[^A-Za-z]', '', name.upper())
    if not name:
        return ""
    
    # Soundex mapping: letters to digits
    # 0: A, E, I, O, U, H, W, Y (vowels and silent letters)
    # 1: B, F, P, V
    # 2: C, G, J, K, Q, S, X, Z
    # 3: D, T
    # 4: L
    # 5: M, N
    # 6: R
    soundex_map = {
        'A': '0', 'E': '0', 'I': '0', 'O': '0', 'U': '0', 'H': '0', 'W': '0', 'Y': '0',
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    # First letter is kept
    first_letter = name[0]
    soundex_code = first_letter
    
    # Process remaining letters
    prev_code = soundex_map.get(first_letter, '0')
    for char in name[1:]:
        code = soundex_map.get(char, '0')
        # Skip vowels and silent letters (code '0'), and consecutive same codes
        if code != '0' and code != prev_code:
            soundex_code += code
            if len(soundex_code) >= 4:
                break
        prev_code = code
    
    # Pad with zeros if needed
    soundex_code = soundex_code.ljust(4, '0')
    
    return soundex_code[:4]


def normalize_name_for_comparison(name: str) -> str:
    """
    Normalize person name for similarity comparison by removing " de " (typically before last name).
    
    This function removes " de " from names to improve matching accuracy when comparing
    names in PDF reports, as " de " is often inconsistently included/excluded.
    
    Args:
        name: Person name string
        
    Returns:
        Normalized name with " de " removed
    """
    if not name:
        return ""
    # Remove " de " (with spaces) from the name
    # Use regex to handle variations like " de ", " de", "de "
    # Remove " de " with word boundaries to avoid removing "de" from other words
    normalized = re.sub(r'\s+de\s+', ' ', name, flags=re.IGNORECASE)
    # Also handle cases where "de" appears at the start or end of a word boundary
    normalized = re.sub(r'\s+de\b', ' ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bde\s+', ' ', normalized, flags=re.IGNORECASE)
    # Clean up multiple spaces
    normalized = ' '.join(normalized.split())
    return normalized.strip()


def get_surname_from_name(name: str) -> str:
    """
    Extract surname from a full name.
    
    For names like "Thomas Kyngesford" or "William de Warrene",
    returns the last word (surname).
    
    Args:
        name: Full name string
        
    Returns:
        Surname (last word) or empty string if no surname found
    """
    if not name:
        return ""
    parts = name.strip().split()
    if not parts:
        return ""
    # Return the last part (surname)
    return parts[-1]


def get_agent_desc(agent: Dict) -> str:
    """Create a descriptive string for an agent including name, status, occupation, and role."""
    name = get_person_name(agent)
    status = agent.get("TblAgentStatus", {}).get("AgentStatus") or ""
    occupation = agent.get("TblAgent", {}).get("Occupation") or ""
    role = agent.get("TblAgentRole", {}).get("role") or ""
    return f"Person: {name}. Role: {role}. Occupation: {occupation}. Status: {status}."


def get_full_date_string(date_list: Union[List[Dict], List[str], None]) -> str:
    """Format EventDate list into a string."""
    if not date_list:
        return ""
    # Handle case where date_list might not be a list
    if not isinstance(date_list, list):
        return str(date_list) if date_list else ""
    
    result = []
    for entry in date_list:
        # Handle case where entry might be a string instead of a dict
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, dict):
            date_val = entry.get("Date")
            if date_val:
                date_type = entry.get("DateType", "")
                result.append(f"{date_val} ({date_type})" if date_type else date_val)
        else:
            # Fallback for any other type
            result.append(str(entry))
    return "; ".join(result)


def format_location(loc_obj: Union[Dict, str]) -> str:
    """Format location details into a printable string."""
    if not loc_obj:
        return ""
    # If it's already a string, return it as-is
    if isinstance(loc_obj, str):
        return loc_obj
    # Otherwise, treat it as a dictionary
    if not isinstance(loc_obj, dict):
        return str(loc_obj)
    parts = [
        loc_obj.get("SpecificPlace"),
        loc_obj.get("Parish"),
        loc_obj.get("Ward"),
        loc_obj.get("County"),
        loc_obj.get("Country"),
    ]
    return ", ".join(filter(None, parts))


def split_into_sentences(items: List[Dict], key: str) -> List[str]:
    """Consolidate text and split into logical sentences."""
    full_text = " ".join(str(item.get(key, "")) for item in items if item.get(key))
    if not full_text:
        return []

    full_text = " ".join(full_text.split())
    pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z0-9])"
    segments = re.split(pattern, full_text)
    return [segment.strip() for segment in segments if segment.strip()]


def get_name_confidence(surname: str, extracted_entities: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Look up Bayesian probability for a surname from extracted_entities.
    Returns: Probability value (0.0-1.0) or None if not found.
    """
    if not extracted_entities or not surname:
        return None
    
    surnames = extracted_entities.get("surnames", [])
    surname_lower = surname.lower().strip()
    
    for entry in surnames:
        term = entry.get("term", "").lower().strip()
        if term == surname_lower:
            probability = entry.get("probability")
            if probability is not None:
                return float(probability)
    
    return None


def get_place_confidence(place_name: str, extracted_entities: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Look up Bayesian probability for a place name from extracted_entities.
    Returns: Probability value (0.0-1.0) or None if not found.
    """
    if not extracted_entities or not place_name:
        return None
    
    place_names = extracted_entities.get("place_names", [])
    place_lower = place_name.lower().strip()
    
    for entry in place_names:
        term = entry.get("term", "").lower().strip()
        anglicized = entry.get("anglicized", "").lower().strip()
        # Match against both original term and anglicized form
        if term == place_lower or anglicized == place_lower:
            probability = entry.get("probability")
            if probability is not None:
                return float(probability)
    
    return None


def normalize_damages_for_comparison(damages_str: str) -> str:
    """
    Normalize damages/currency strings for comparison.
    
    Handles various currency formats:
    - Currency symbols: £, $, etc.
    - Currency words: pounds, pound, lbs, shillings, pence, etc.
    - Historical abbreviations: l (libra/pounds), s (shillings), d (pence) after numbers
    - Number formats: integers, decimals
    
    Normalizes to format: "NUMBER CURRENCY_UNIT" (e.g., "100 pounds")
    
    Examples:
    - "£100" -> "100 pounds"
    - "100 pounds" -> "100 pounds"
    - "100 pound" -> "100 pounds"
    - "100l" -> "100 pounds" (historical abbreviation)
    - "10s 5d" -> "10 shillings 5 pence"
    
    Args:
        damages_str: String containing damages/currency information
        
    Returns:
        Normalized string for comparison
    """
    if not damages_str:
        return ""
    
    text = damages_str.strip()
    if not text:
        return ""
    
    # Convert to lowercase for processing
    text_lower = text.lower()
    
    # Currency symbol mappings (standalone symbols)
    # £ = pounds sterling
    currency_symbols = {
        '£': 'pounds',
        '$': 'dollars',  # Though in historical context might be pounds
    }
    
    # Historical abbreviations (appear after numbers): l = libra (pounds), s = solidus (shillings), d = denarius (pence)
    historical_abbrevs = {
        'l': 'pounds',
        's': 'shillings',
        'd': 'pence',
    }
    
    # Currency word normalization
    currency_words = {
        'pound': 'pounds',
        'pounds': 'pounds',
        'lbs': 'pounds',
        'lb': 'pounds',
        'libra': 'pounds',
        'shilling': 'shillings',
        'shillings': 'shillings',
        'solidus': 'shillings',
        'pence': 'pence',
        'penny': 'pence',
        'pennies': 'pence',
        'denarius': 'pence',
        'dollar': 'dollars',
        'dollars': 'dollars',
    }
    
    # Check for standalone currency symbols (like £, $)
    found_currency_symbol = None
    for symbol, currency in currency_symbols.items():
        if symbol in text:
            found_currency_symbol = currency
            break
    
    # Check for historical abbreviations after numbers (like "100l", "10s", "5d")
    found_historical_abbrev = None
    # Pattern: number followed immediately by l, s, or d (possibly with spaces)
    historical_pattern = r'\d+\s*([lsd])\b'
    historical_match = re.search(historical_pattern, text_lower)
    if historical_match:
        abbrev_char = historical_match.group(1)
        if abbrev_char in historical_abbrevs:
            found_historical_abbrev = historical_abbrevs[abbrev_char]
    
    # Extract currency words
    found_currency_word = None
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        if word in currency_words:
            found_currency_word = currency_words[word]
            break
    
    # Determine the currency unit (priority: symbol > historical abbrev > word > default)
    currency_unit = found_currency_symbol or found_historical_abbrev or found_currency_word or 'pounds'
    
    # Extract numbers (including decimals)
    # Match patterns like: 100, 100.50, 100,000, etc.
    number_pattern = r'\d+(?:[.,]\d+)*'
    numbers = re.findall(number_pattern, text)
    
    # Build normalized string
    if numbers:
        # Use the first (and typically only) number found
        number_str = numbers[0].replace(',', '')  # Remove thousand separators
        # Handle case where comma is decimal separator (European format)
        if ',' in number_str and '.' not in number_str:
            # Check if there are multiple comma-separated parts
            parts = number_str.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Likely decimal: "100,50" -> "100.50"
                number_str = '.'.join(parts)
            else:
                # Likely thousands separator: "100,000" -> already handled by replace above
                number_str = ''.join(parts)
        
        return f"{number_str} {currency_unit}"
    else:
        # No number found, just normalize currency words
        normalized_words = []
        for word in words:
            if word in currency_words:
                normalized_words.append(currency_words[word])
            else:
                normalized_words.append(word)
        return ' '.join(normalized_words)


def normalize_writ_type_for_comparison(writ_type_str: str) -> str:
    """
    Normalize writ type strings for comparison by removing parenthetical qualifiers.
    
    Handles cases where Ground Truth has qualifiers like "Debt (account)" or "Debt (loan)"
    but AI extraction has just "Debt". This normalization extracts the base type.
    
    Examples:
    - "Debt (account)" -> "Debt"
    - "Debt (loan)" -> "Debt"
    - "Debt" -> "Debt"
    - "Trespass (battery)" -> "Trespass"
    
    Args:
        writ_type_str: Writ type string that may contain parenthetical qualifiers
        
    Returns:
        Normalized writ type with parenthetical qualifiers removed
    """
    if not writ_type_str:
        return ""
    
    text = writ_type_str.strip()
    if not text:
        return ""
    
    # Remove parenthetical qualifiers like "(account)", "(loan)", etc.
    # Pattern matches: ( ... ) with any content inside
    normalized = re.sub(r'\s*\([^)]*\)', '', text)
    
    # Clean up any extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized.strip()


def normalize_date_for_comparison(date_str: str) -> str:
    """
    Normalize date strings for comparison by stripping timestamps and metadata.
    
    Handles cases where dates have timestamps (e.g., "1400-03-10 00:00:00") or
    parenthetical metadata (e.g., "(initial)") that should be ignored for comparison.
    
    Examples:
    - "1400-03-10 00:00:00 (initial)" -> "1400-03-10"
    - "1400-03-10 (initial)" -> "1400-03-10"
    - "1400-03-10 00:00:00" -> "1400-03-10"
    - "1400-03-10" -> "1400-03-10"
    - "1400-03-10; 1400-03-12" -> "1400-03-10; 1400-03-12" (handles multiple dates)
    
    Args:
        date_str: Date string that may contain timestamps and metadata
        
    Returns:
        Normalized date string with timestamps and metadata removed
    """
    if not date_str:
        return ""
    
    text = date_str.strip()
    if not text:
        return ""
    
    # Handle multiple dates separated by semicolons
    date_parts = text.split(';')
    normalized_parts = []
    
    for part in date_parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove parenthetical metadata like "(initial)", "(final)", etc.
        part = re.sub(r'\s*\([^)]*\)', '', part)
        
        # Remove timestamps (patterns like " 00:00:00", " 12:34:56", etc.)
        # Match space followed by HH:MM:SS or HH:MM:SS.microseconds
        part = re.sub(r'\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?', '', part)
        
        # Clean up any extra whitespace
        part = ' '.join(part.split())
        part = part.strip()
        
        if part:
            normalized_parts.append(part)
    
    return '; '.join(normalized_parts)