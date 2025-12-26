"""Prompt construction helpers (content ported from the legacy workflow manager)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types


def _attach_image(
    parts: List[types.Part],
    image_path: str,
    upload_fn,
    use_images: bool,
) -> List[types.Part]:
    """
    Attach an image to a list of prompt parts if image mode is enabled.

    Args:
        parts: Existing list of prompt parts (typically text).
        image_path: Path to the image file to attach.
        upload_fn: Function to upload the image and return a file reference.
        use_images: If False, returns parts unchanged (text-only mode).

    Returns:
        List of parts with image reference appended (if use_images=True and upload succeeds),
        or the original parts list unchanged.
    """
    if not use_images:
        return parts
    file_ref = upload_fn(image_path)
    if file_ref:
        parts.append(types.Part.from_uri(file_uri=file_ref.uri, mime_type=file_ref.mime_type))
    return parts


def build_county_prompt(image_path: str, upload_fn, use_images: bool) -> List[types.Part]:
    """
    Build prompt for extracting county name from marginal annotation.

    Creates a prompt that instructs the AI to identify the county abbreviation
    in the left margin of a CP40 plea roll image and return the full county name.

    Args:
        image_path: Path to the manuscript image.
        upload_fn: Function to upload the image and return a file reference.
        use_images: If True, attaches the image to the prompt.

    Returns:
        List of prompt parts (text + optional image reference).
    """
    prompt = """
**Task:** Identify the COUNTY marginal annotation in this plea roll image.

**Context:** In Court of Common Pleas rolls (CP40), the county is written in the LEFT MARGIN, 
typically abbreviated. It indicates jurisdiction and is CRITICAL for legal identification.

**Common Abbreviations:**
- `Midd` or `Midđ` = Middlesex
- `Kanc` or `Kanc'` = Kent  
- `London` or `Lond` = London
- `Essex` or `Ess'` = Essex
- `Hertf` or `Hertford` = Hertfordshire
- `Suff` or `Suff'` = Suffolk
- `Norf` or `Norff'` = Norfolk
- `Surr` or `Surr'` = Surrey
- `Sussex` or `Suss'` = Sussex
- `Glouc` or `Glouc'` = Gloucestershire
- `Oxon` or `Oxon'` = Oxfordshire
- `Warr` or `Warr'` = Warwickshire
- `Derb` or `Derby` = Derbyshire
- `Leic` or `Leic'` = Leicestershire
- `Northt` or `Northampt` = Northamptonshire

**Instructions:**
1. Look at the LEFT MARGIN of the image (first 10-15% of width)
2. Find the abbreviated county name (usually appears near the top of each case entry)
3. Return the FULL county name in English

**Output:** Return ONLY a JSON object:
```json
{
    "marginal_text": "the exact text visible in margin",
    "county": "Full County Name",
    "confidence": "high|medium|low"
}
```
"""
    parts = [types.Part.from_text(text=prompt)]
    return _attach_image(parts, image_path, upload_fn, use_images)


from typing import Dict, Any

def get_step1_instruction_text() -> str:
    """
    Get the static instruction text for Step 1 diplomatic transcription.
    Updated with STRICT character decomposition rules and explicit bans on precomposed glyphs.
    """
    return """
***

**Role:** Expert 15th-c. English Court of Common Pleas Paleographer.

**Task:** Produce a strictly diplomatic transcription of the manuscript image within the defined regions.

**Context:** 15th-century Court of Common Pleas plea roll (CP40). Use legal context to resolve ambiguities (e.g., distinguishing *u/n*, *c/t*), but never modernize.

---

## Instructions:

### 1. Spatial & Visual Priority
*   Transcribe **only** the visible pixels inside the box defined by `BOX_2D`.
*   If a word is cut off by the box edge, transcribe only the visible letters. Do not autofill.

---

## 2. STRICT Character Set & Forbidden Glyphs

**CRITICAL RULE:** You may **ONLY** use Latin letters (A-Z, a-z) and the specific symbols in the "Allowed List" below.

### **A. Forbidden Characters (Automatic Rejection)**
| Forbidden | Reason | Required Replacement |
| :--- | :--- | :--- |
| **`ñ`** | Precomposed Tilde | **`n`** + **`̅`** (n + combining overline) |
| **`ī`, `ū`, `ē`, `ā`** | Precomposed Macrons | **`i`**+**`̅`**, **`u`**+**`̅`**, etc. |
| **`'`** | Apostrophe | None (Delete) |
| **`-`** or **`=`** | Hyphens/Line Fillers | None (Delete entirely) |
| **`:`** | Colon | `.` or `;` or `·` |
| **`&`** | Ampersand | **`⁊`** (Tironian Et) |

### **B. Decomposition Rule (THE "Ñ" TRAP)**
Middle English scribes write a tilde over an `n` (n̅).
*   **NEVER** use the Spanish `ñ` (U+00F1).
*   **ALWAYS** use `n` followed immediately by the combining overline `̅`.

### **C. Allowed Special Characters**
Use these pre-composed glyphs **only** when the visual stroke physically modifies the letter.

| Glyph | Name | Usage |
| :---: | :--- | :--- |
| **⁊** | Tironian Et | *⁊* (et/and) |
| **ſ** | Long S | *reſpondend* (Start/Middle of word) |
| **ħ** | H with Stroke | *Joħes* (Johannes) |
| **ł** | L with Stroke | *Wiłłm*, *vidłt* |
| **đ** | D with Stroke | *qđ* (quod) |
| **ꝛ** | R Rotunda | *libꝛas* (Only after **o, b, p, d, h**) |
| **ꝝ** | Rum Rotunda | *alioꝝ* (aliorum) |
| **ꝫ** | Et/Us/Que | *omnibꝫ*, *usqꝫ* |
| **ꝑ** | P with Stroke | *ꝑ* (per/par) - Stroke through tail |
| **ꝓ** | P with Flourish | *ꝓ* (pro) - Loop curving left |
| **ꝙ** | Q with Diagonal | *ꝙd* (quod) |
| **ꝰ** | Us Modifier | *huiꝰ* (huius) |
| **ꝭ** | Is Modifier | *scilꝭt* (scilicet) |
| **ı** | Dotless I | *scı̅* (used under overlines) |
| **¶** | Pilcrow | Section start |
| **þ** | Thorn | Names/Places (*þe*) |
| **ȝ** | Yogh | Names (*Knyȝt*) |

### **D. Combining Diacritics**
*   **`̅`** (Overline): Use for all suspended nasals (`om̅is`) and abbreviations (`p̅` for pre).
*   **`ͬ`** (Superscript r): Use for *ur* abbreviations (`tͬ`).

---

## 3. Transcription Rules

### The "R vs. ꝛ" Rule
1.  **R Rotunda (`ꝛ`):** Use ONLY after rounded letters: **o, b, p, d, h**.
2.  **Normal R (`r`):** Use after straight letters: **e, i, u, a, c, t, s**.

### Specific Word Conventions
| Word | Transcription | Note |
| :--- | :--- | :--- |
| *Johannes* | `Joħes` | Use `ħ`. |
| *Willelmus* | `Wiłłm` | Use `ł`. |
| *Placito* | `pli̅to` | Use `i` + `̅`. **NOT** `plīto`. |
| *Stopham-* | `Stopham` | **Delete** the trailing `-` or `=`. |

---

## 4. HTR Error Correction
The input HTR often mistakes capitals. Use visual evidence + context:
*   HTR `B` $\to$ often `O`, `G`, `E`, or `C` in image.
*   HTR `y` $\to$ often `i` in image.
*   HTR `n` $\to$ often `u` or `v` in image.

---

## 5. Final Validation Checklist
Before outputting, verify:
1.  Did I use `ñ`? $\to$ **CHANGE** to `n`+`̅`.
2.  Did I use `=` or `-`? $\to$ **DELETE** it.
3.  Did I use `ī`? $\to$ **CHANGE** to `i`+`̅`.
4.  Are all characters in the list below?

## 6. COMPLETE ALLOWED CHARACTER LIST
Use **ONLY** these characters.
[
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  ".", ",", ";", "¶", "·",
  "⁊", "ſ", "ħ", "ł", "đ", "ꝛ", "ꝝ", "ꝫ", "ꝑ", "ꝓ", "ꝙ", "ꝰ", "ꝭ", "ı",
  "þ", "ȝ", "÷",
  "̅", "ͬ", "ᵃ", "ᵉ", "ⁱ", "ᵒ", "ᵘ"
]
"""


def build_step1_prompt(
    lines: List[Dict[str, Any]],
    image_path: str,
    upload_fn,
    use_images: bool,
) -> List[types.Part]:
    """
    Build comprehensive prompt for diplomatic transcription (Step 1).

    Creates a detailed prompt for AI-assisted paleographic transcription that:
    - Preserves medieval orthography and abbreviations
    - Uses special Unicode characters for medieval glyphs
    - Follows strict diplomatic transcription conventions
    - Includes HTR output as a spatial guide

    Args:
        lines: List of line dictionaries with HTR text and bounding boxes.
        image_path: Path to the manuscript image.
        upload_fn: Function to upload the image and return a file reference.
        use_images: If True, attaches the image to the prompt.

    Returns:
        List of prompt parts (text + optional image reference).
    """
    formatted_htr_list = []
    for idx, line in enumerate(lines, 1):
        entry = {"ID": f"L{idx:02d}", "TEXT": line["htr_text"]}
        if line.get("bbox"):
            entry["BOX_2D"] = line["bbox"]
        formatted_htr_list.append(entry)

    htr_json_str = json.dumps(formatted_htr_list, indent=2, ensure_ascii=False)
    
    # Get static instruction text and combine with variable HTR data
    instruction_text = get_step1_instruction_text()
    variable_text = f"""
    ## Input HTR:

    {htr_json_str}

    """
    prompt = instruction_text + variable_text
    print(prompt)
    parts = [types.Part.from_text(text=prompt)]
    return _attach_image(parts, image_path, upload_fn, use_images)

def build_step2a_prompt(diplomatic_results: Dict[str, Any], group_lines_by_image: Dict[str, List[Dict[str, Any]]] = None) -> List[types.Part]:
    """
    Build prompt for merging diplomatic transcriptions and extracting entities (Step 2a).

    Creates a prompt to:
    1. Stitch together overlapping text fragments from multiple images
    2. Extract all surnames appearing in the text
    3. Extract all place names with original and anglicized forms

    Args:
        diplomatic_results: Dictionary mapping image names to their Step 1 JSON results.
        group_lines_by_image: Optional dictionary mapping image names to HTR line data with bounding boxes.

    Returns:
        List of prompt parts (text only).
    """
    def limit_candidates_to_top3(entity_list):
        """Limit candidates array to top 3 for each entity and filter fields to only text and probability."""
        if not isinstance(entity_list, list):
            return entity_list
        limited_entities = []
        for entity in entity_list:
            if not isinstance(entity, dict):
                limited_entities.append(entity)
                continue
            # Create a copy of the entity, but exclude unwanted fields
            limited_entity = {}
            # Copy only allowed fields from entity (exclude position_in_htr, best_candidate)
            if "original" in entity:
                limited_entity["original"] = entity["original"]
            if "corrected" in entity:
                limited_entity["corrected"] = entity["corrected"]
            # Limit candidates to top 3 and filter to only text and probability
            if "candidates" in entity and isinstance(entity["candidates"], list):
                filtered_candidates = []
                for cand in entity["candidates"][:3]:
                    if isinstance(cand, dict):
                        filtered_cand = {}
                        if "text" in cand:
                            filtered_cand["text"] = cand["text"]
                        if "probability" in cand:
                            filtered_cand["probability"] = cand["probability"]
                        if filtered_cand:  # Only add if it has at least one field
                            filtered_candidates.append(filtered_cand)
                limited_entity["candidates"] = filtered_candidates
            limited_entities.append(limited_entity)
        return limited_entities
    
    try:
        input_block = ""
        sorted_filenames = sorted(diplomatic_results.keys())
        for i, img_name in enumerate(sorted_filenames):
            json_data = diplomatic_results[img_name]
            
            # Handle case where json_data might be a list (shouldn't happen, but defensive)
            if isinstance(json_data, list):
                logger.warning(f"diplomatic_results[{img_name}] is a list instead of dict. Converting...")
                json_data = {"lines": json_data}
                diplomatic_results[img_name] = json_data
            
            # Ensure json_data is a dict
            if not isinstance(json_data, dict):
                logger.error(f"diplomatic_results[{img_name}] is not a dict or list: {type(json_data)}")
                logger.error(f"  Value: {json_data}")
                continue
            
            lines_data = json_data.get("lines", [])
            
            # Validate lines_data is a list
            if not isinstance(lines_data, list):
                logger.error(f"lines_data for {img_name} is not a list: {type(lines_data)}")
                logger.error(f"  Value: {lines_data}")
                continue
            
            # Build enriched line data with bounding boxes if available
            enriched_lines = []
            for idx, line in enumerate(lines_data, 1):
                try:
                    # Handle case where line might be a list or other non-dict type
                    if not isinstance(line, dict):
                        logger.error(f"Line {idx} for {img_name} is not a dict: {type(line)}")
                        logger.error(f"  Line value (first 200 chars): {str(line)[:200]}")
                        logger.error(f"  Line type: {type(line)}")
                        # Try to convert if it's a list
                        if isinstance(line, list):
                            logger.warning(f"  Attempting to convert list to dict (assuming first element is transcription)")
                            if len(line) > 0:
                                line = {"transcription": str(line[0]) if line else ""}
                            else:
                                logger.warning(f"  Empty list, skipping line {idx}")
                                continue
                        else:
                            logger.warning(f"  Skipping line {idx} due to invalid type")
                            continue
                    
                    # Get transcription - handle different possible keys
                    transcription = line.get("transcription", "") or line.get("corrected_text", "") or line.get("text", "")
                    
                    line_entry = {
                        "line_id": f"L{idx:02d}",
                        "transcription": transcription
                    }
                    
                    # Add bounding box information if available from HTR data
                    if group_lines_by_image and img_name in group_lines_by_image:
                        try:
                            htr_lines = group_lines_by_image[img_name]
                            if not isinstance(htr_lines, list):
                                logger.warning(f"group_lines_by_image[{img_name}] is not a list: {type(htr_lines)}")
                            elif idx - 1 < len(htr_lines):
                                htr_line = htr_lines[idx - 1]
                                if not isinstance(htr_line, dict):
                                    logger.warning(f"htr_line at index {idx-1} for {img_name} is not a dict: {type(htr_line)}")
                                elif htr_line.get("bbox"):
                                    # bbox format: [ymin, xmin, ymax, xmax]
                                    line_entry["bbox"] = htr_line["bbox"]
                        except Exception as e:
                            logger.warning(f"Error accessing HTR data for {img_name} line {idx}: {e}")
                            # Continue without bbox
                    
                    # Add entity information with Bayesian correction candidates and probabilities
                    # This allows the LLM to reconcile names across images
                    # Limit to top 3 candidates for each entity to reduce prompt size
                    if line.get("entities"):
                        entities = line["entities"]
                        line_entry["entities"] = {
                            "forenames": limit_candidates_to_top3(entities.get("forenames", [])),
                            "surnames": limit_candidates_to_top3(entities.get("surnames", [])),
                            "placenames": limit_candidates_to_top3(entities.get("placenames", []))
                        }
                    elif line.get("forenames") or line.get("surnames") or line.get("placenames"):
                        # Fallback: entities might be at line level directly
                        line_entry["entities"] = {
                            "forenames": limit_candidates_to_top3(line.get("forenames", [])),
                            "surnames": limit_candidates_to_top3(line.get("surnames", [])),
                            "placenames": limit_candidates_to_top3(line.get("placenames", []))
                        }
                    
                    enriched_lines.append(line_entry)
                except Exception as e:
                    logger.error(f"Error processing line {idx} for {img_name}: {e}")
                    logger.error(f"  Line type: {type(line)}")
                    logger.error(f"  Line value (first 200 chars): {str(line)[:200]}")
                    continue
            
            input_block += f"--- FRAGMENT {i+1} (Source: {img_name}) ---\n"
            input_block += json.dumps(enriched_lines, indent=2, ensure_ascii=False) + "\n\n"

        # Use regular string (not f-string) to avoid issues with JSON curly braces in input_block
        prompt = """System Role:

You are an expert medieval paleographer and archivist. You are correcting and merging raw OCR transcriptions of Latin legal manuscripts.

Input Context:

You are provided with a JSON block containing transcription lines from one or more image files. Each line includes:
- line_id: Line identifier (e.g., L01, L02)
- transcription: The diplomatic transcription text
- bbox (optional): Bounding box coordinates [ymin, xmin, ymax, xmax] indicating the spatial position of the line in the original image
- entities (optional): Bayesian correction results with candidate names and probabilities for:
  - forenames: Array of forename entities with candidates and probabilities
  - surnames: Array of surname entities with candidates and probabilities
  - placenames: Array of placename entities with candidates and probabilities

Scenario A (Sequential): The files represent different parts of one long scroll (e.g., Top/Bottom). They will overlap partially.

Scenario B (Parallel): The files represent the same physical text processed multiple times (nondeterministic OCR). They will cover the exact same text range.

Operational Instructions:

PHASE 1: ANALYZE AND MERGE

Compare Content: Read the transcription text from all input files.

Use Spatial Information: When bounding boxes are available, use them to understand the relative layout WITHIN each fragment:
- IMPORTANT: Absolute Y-coordinates cannot be compared across fragments (different camera angles/reference points)
- Within each fragment: lines are ordered top-to-bottom (increasing Y values = lower on page)
- Use bbox to understand line density and relative positioning within each fragment
- For overlap detection: Compare the LAST lines of Fragment 1 with the FIRST lines of Fragment 2 based on TEXT CONTENT and their relative positions within their respective fragments
- If Fragment 1's bottom lines match Fragment 2's top lines in content, you've found the overlap zone
- The bbox data helps you understand how many lines are in each fragment and their relative spacing, which can inform decisions about where seams are likely to occur

Determine Relationship:

IF the text in File B repeats the text in File A almost entirely (>80% similarity): Treat this as Scenario B. Use the clearest version of each word to correct typos. Do not repeat the text. Output one single, corrected version.

IF the text in File B continues where File A left off (with a small overlap zone): Treat this as Scenario A. Find the unique overlapping phrase (the "seam"), remove the duplicate lines from the start of File B, and stitch them into one continuous stream. Use bbox information to identify which lines are in the overlap zone.

Diplomatic Standards:

Preserve original spelling, abbreviations (e.g., ꝙ, ꝑ, ⁊, p', q', dm̅i), and case.

**CRITICAL: DO NOT EXPAND ABBREVIATIONS IN THIS STEP.** Preserve all abbreviations exactly as they appear in the input transcriptions. Abbreviations should remain in their original form (e.g., "p'", "q'", "ꝑ", "dm̅i", etc.). Do NOT add square brackets or expand abbreviations. Abbreviation expansion will be done in Step 2b.

Ignore JSON Line IDs (L01, etc.) - these are for reference only.

PHASE 2: RECONCILE AND EXTRACT ENTITIES (From the Result of Phase 1)

**IMPORTANT: ENTITY RECONCILIATION USING BAYESIAN CORRECTION DATA**

Each line in the input includes an "entities" field containing Bayesian correction results with:
- **original**: The name as it appeared in the raw HTR text
- **corrected**: The best candidate from Bayesian correction
- **candidates**: Array of candidate names with:
  - **text**: The candidate name
  - **probability**: Probability score (0.0 to 1.0) from Bayesian analysis

**RECONCILIATION RULES:**
1. **Cross-Image Consistency**: When the same person or place appears in multiple images, use the name with the HIGHEST PROBABILITY across all occurrences. This ensures consistency in the stitched result.
2. **Probability-Based Selection**: For each unique entity (same person/place), compare probabilities across all images and select the candidate with the highest combined probability.
3. **Context Awareness**: Consider the context - if multiple high-probability candidates exist for similar names, choose the one that makes the most sense in the legal context.
4. **Confidence Levels**: Prefer 'bayesian_corrected' over 'original' when probabilities are similar, as these have been validated against the database and image evidence.

**EXTRACTION REQUIREMENTS:**

Surnames: Extract family names (e.g., Boſevyle, Colverdone). Exclude Christian names (e.g., Johannes, Willms). Use the reconciled name (highest probability across all images).

Place Names: Extract locations. Return a list of objects with the original Latin spelling and the anglicized modern English equivalent (e.g., 'Ebor' -> 'York'). Use reconciled names based on probability.

Marginal County: **REQUIRED OUTPUT** - This is CRITICAL legal metadata. The county name appears in the LEFT MARGIN of the manuscript, vertically positioned near where the case starts.

**HOW TO FIND MARGINAL COUNTY USING BBOX COORDINATES:**
1. **Left Margin Identification**: Look for lines with LOW xmin values (typically in the leftmost 10-15% of the image width). The bbox format is [ymin, xmin, ymax, xmax], so lines with small xmin values are in the left margin.
2. **Vertical Position**: The marginal county appears near the START of the case entry - typically in the first few lines of the fragment where the case begins (low ymin values within each fragment).
3. **Text Characteristics**: The marginal county is usually a short abbreviation (e.g., 'Midd', 'Kanc', 'Lond', 'Ess') written vertically in the left margin, often appearing before or alongside the first substantive text of the case.
4. **Extraction**: Once identified using bbox coordinates, extract the exact text as it appears (original Latin form) and provide the anglicized English county name (e.g., 'Midd' -> 'Middlesex', 'Kanc' -> 'Kent', 'Lond' -> 'London').

**IMPORTANT**: You MUST identify the marginal county using the spatial information (bbox coordinates) provided. Do not assume it's the first word of the merged text - use the bbox data to find text in the left margin near the case start.

Return this in the marginal_county field of the JSON as {"original": "LatinName", "anglicized": "EnglishName"}. If you cannot identify it from the bbox coordinates, return {"original": "", "anglicized": ""} but note that this is a required field and should be extracted whenever possible.

Output Format:

Return valid JSON only.

{
  "processing_logic_used": "Scenario A (Stitching) OR Scenario B (Consensus)",
  "merged_text": "THE_FULL_CONTINUOUS_DIPLOMATIC_TEXT_STRING",
  "surnames": [ "Name1", "Name2" ],
  "place_names": [
    { "original": "LatinName", "anglicized": "EnglishName" }
  ],
  "marginal_county": {
    "original": "LatinName",
    "anglicized": "EnglishName"
  }
}

Input Data:

""" + input_block
        return [types.Part.from_text(text=prompt)]
    except Exception as e:
        logger.error(f"Error in build_step2a_prompt: {e}", exc_info=True)
        logger.error(f"diplomatic_results keys: {list(diplomatic_results.keys()) if diplomatic_results else 'None'}")
        if diplomatic_results:
            for img_name, data in list(diplomatic_results.items())[:3]:  # First 3 items
                logger.error(f"  {img_name}: type={type(data)}")
                if isinstance(data, dict):
                    logger.error(f"    keys: {list(data.keys())}")
                    if "lines" in data:
                        lines = data["lines"]
                        logger.error(f"    lines type: {type(lines)}")
                        if isinstance(lines, list) and len(lines) > 0:
                            logger.error(f"    first line type: {type(lines[0])}")
                            logger.error(f"    first line value (first 200 chars): {str(lines[0])[:200]}")
        raise


def build_step2b_prompt(merged_text: str) -> List[types.Part]:
    """
    Build prompt for expanding medieval abbreviations (Step 2b).

    Creates a prompt to expand abbreviations in diplomatic Latin text to
    standard legal Latin while preserving the diplomatic structure.

    Args:
        merged_text: The merged diplomatic text from Step 2a.

    Returns:
        List of prompt parts (text only).
    """
    # Static instructions first, dynamic input at the end
    prompt = f"""
        Role: Expert Latin Philologist.
        Task: Expand the abbreviations in the provided diplomatic Latin text to standard legal Latin.

        **Rules:**
        1. Expand standard 15th-century legal abbreviations (e.g., `p'` -> `p[er]`, `q'` -> `q[uod]`, `dm̅i` -> `d[omi]ni`, `ꝑ` -> `p[er]`, `⁊` -> `et`).
        2. Use only standard latin characters (no special unicode latin glyphs) and basic punctuation.
        3. Maintain the original grammatical case.
        4. **CRITICAL: TRANSCRIPTION CONVENTION**: Show expanded abbreviations in square brackets. The format is: [expanded_letters] where the letters in brackets represent what was abbreviated. Examples:
           - "p'" -> "p[er]"
           - "q'" -> "q[uod]"
           - "Joh'is" -> "Joh[ann]is"
           - "d'ni" -> "d[omi]ni"
           - "&c" -> "et c[etera]"
        
        This convention preserves the original abbreviated form while clearly indicating the expansion in brackets. Words without abbreviations should remain unchanged. Apply this pattern consistently to all abbreviations in the text.
        5. **Output:** Return ONLY the expanded Latin text with square brackets showing abbreviations.
        6. **County Name Handling:** The input text may begin with a county name abbreviation from a marginal annotation (e.g., "Midd", "Kanc", "Lond", "Ess"). 
           - If a county abbreviation appears at the start of the text, expand it using the same square bracket convention (e.g., "Midd" -> "Midd[lesex]", "Kanc" -> "Kanc[iam]" or "Cant[ia]").
           - If the county name is already fully written, leave it as is.
           - If no county name appears at the start, that's fine - just expand the abbreviations in the text that is present.
           - Do NOT add a county name if one is not present in the input text.
           - Do NOT change the order of words - preserve the text exactly as it appears, only expanding abbreviations.

        **IMPORTANT: WORKFLOW EFFICIENCY**
        - Make your decisions confidently and move forward. Do not repeatedly second-guess or self-correct the same abbreviations.
        - If you encounter an ambiguous abbreviation, choose the most likely expansion based on context and move on.
        - Focus on producing the final output rather than excessive reasoning about each abbreviation.
        - Once you have determined the expansion for an abbreviation, apply it consistently and proceed to the next one.

        **Input (Diplomatic):**
        {merged_text}
        """
    return [types.Part.from_text(text=prompt)]


def build_step3_prompt(latin_text: str) -> List[types.Part]:
    """
    Build prompt for translating expanded Latin text to English (Step 3).

    Creates a simple prompt to translate the expanded legal Latin text
    into fluent English.

    Args:
        latin_text: The expanded Latin text from Step 2b.

    Returns:
        List of prompt parts (text only).
    """
    prompt = f"Provide a fluent English translation of the latin text. Return only translation:\n{latin_text}"
    return [types.Part.from_text(text=prompt)]


def build_step4_prompt(
    english_text: str,
    latin_text: Optional[str],
    date_info: Optional[Dict[str, Any]],
    county_info: Optional[Dict[str, Any]],
) -> List[types.Part]:
    """
    Build prompt for structured entity extraction (Step 4).
    """
    context_block = ""
    metadata_parts = []
    
    if date_info:
        metadata_parts.append(f"""Court: Court of Common Pleas
            Series: CP 40
            Roll Number: {date_info.get('Roll')}
            Term: {date_info.get('Term')}
            Calendar Year: {date_info.get('Calendar Year')}
            Regnal Year: {date_info.get('Regnal Year')}""")
    
    if county_info:
        county_name = county_info.get('county', 'UNKNOWN')
        county_source = county_info.get('source', 'unknown')
        county_latin = county_info.get('county_original_latin', '')
        
        county_note = f"""County: {county_name}
            County Source: {county_source}"""
        if county_latin:
            county_note += f"""
            County (Original Latin): {county_latin}"""
        
        if county_source == "marginal_annotation":
            county_note += """
            
            NOTE: This county was extracted from the marginal annotation at the start of the text.
            Use this value for the County field in TblReference and TblCase. This is authoritative."""
        elif county_source == "not_found" or county_name == "UNKNOWN":
            county_note += """
            
            **CRITICAL: County Identification Priority**
            Since no marginal county was found, you MUST extract the county from the venue line in the text:
            - Look for: "X summonitus fuit ad respondendum Y de [County]" (X was summoned to answer Y concerning [County])
            - The county in the venue line is the authoritative source when marginal annotation is missing
            - Do NOT default to a common location or assume the county
            - If the venue line specifies a county, use that value for TblReference.County and TblCase.County"""
        
        metadata_parts.append(county_note)
    
    if metadata_parts:
        context_block = f"""
            MANDATORY ARCHIVAL METADATA (Use these values, do not extract from text):
            {chr(10).join(metadata_parts)}
            
            **CRITICAL: You MUST populate the following schema fields from the metadata above:**
            - TblReference.term: Use the "Term" value from metadata (must be one of: "Michaelmas", "Hilary", "Easter", "Trinity")
            - TblReference.County: Use the "County" value from metadata (must match one of the enum values in the schema)
            - TblReference.dateyear: Use the "Calendar Year" value from metadata (as an integer)
            - TblReference.reference: Construct from "Roll Number" and rotulus (e.g., "CP40-562 340")
            - TblCase.County: Use the same "County" value from metadata
            
            If any metadata value is missing or "UNKNOWN", you may extract from the text as a fallback, but metadata takes precedence.
            
            IMPORTANT: Use the Calendar Year and Regnal Year above to convert medieval feast dates in the text to ISO dates.
            For example, if the text says "feast of St. Michael in the 6th year of Henry VI" and the metadata shows
            Regnal Year "6 Henry VI" corresponds to Calendar Year 1427, then convert this to "1427-09-29".
            """
    else:
        # Fallback if no metadata provided
        context_block = """
            NOTE: No archival metadata was provided. You must extract Term, County, and Calendar Year from the text.
            - Look for term references: "Michaelmas", "Hilary", "Easter", "Trinity" in the text
            - Look for county names in the text or marginal annotations
            - Look for regnal years and convert to calendar years
            """

    static_instructions = """
        Role: Medieval Legal Historian and Database Specialist for Court of Common Pleas records.

        PRIMARY TASK: Extract ALL named entities, legal pleadings, and procedural outcomes from this plea roll entry.

        ENTITY EXTRACTION RULES

        A. NAME HANDLING AND ANGLICIZATION
        1. Anglicize Names: Convert Latin names to English (Johannes -> John, Henricus -> Henry). Keep surnames as is unless a standard translation is obvious (e.g., Faber -> Smith).
        
        2. **NAME VALIDATION RULES (CRITICAL):**
           - If you see "Rog'us" or "Rog'i" in Latin → Extract as "Roger" (NOT "Robert")
           - If you see "Rob'us" or "Rob'i" in Latin → Extract as "Robert" (NOT "Roger")
           - If you see "Nic'us" or "Nic'o" in Latin → Extract as "Nicholas" (NOT "Michael")
           - If you see "Mich'lus" in Latin → Extract as "Michael" (NOT "Nicholas")
           - Cross-reference with the Latin text provided to verify names
        
        3. **PALEOGRAPHIC CHARACTER DISAMBIGUATION (CAPITAL LETTERS) - CRITICAL:**
           **The HTR model may confuse similar capital letters in "Court Hand" script. Pay careful attention to:**
           - **C vs. R/G**: Capital C can be misread as R or G (e.g., "Walter Cok" might be misread as "Walter Roke")
           - **G vs. C**: Capital G can be misread as C (e.g., "Richard Goold" might be misread as "Richard Coolde")
           - **K vs. H**: Capital K can be misread as H (e.g., "Robert Kelme" might be misread as "Robert Holme")
           - **Extraction rules**:
             * When extracting surnames, carefully distinguish these similar capital letters
             * Use context clues (known surnames, place names, other mentions) to resolve ambiguities
             * If uncertain, preserve the original transcription but flag for review
             * Cross-reference with the Latin text and other mentions of the same name in the document
        
        4. **SURNAME ACCURACY:**
           - "Sauvage" and "Stannage" are DIFFERENT surnames - extract exactly as written
           - "Mymmes" and "Symmers" are DIFFERENT surnames - extract exactly as written
           - Extract surnames EXACTLY as they appear in the text, do not substitute similar-sounding names
           - If uncertain, use the name from the Latin text (provided in Latin Reference section)
        
        5. Aliases: If text says "[NAME1] alias [NAME2]", record BOTH.
        6. Status/Occupation: Extract EXACTLY as found (e.g., "citizen and mercer", "husbandman"). Always capture "citizen".
        
        7. **HISTORICAL PLACE NAME NORMALIZATION (GAZETTEER INTEGRATION) - CRITICAL:**
           **CRITICAL: Map archaic/medieval spellings to historically accurate modern equivalents**
           - Do NOT simply use phonetically similar modern place names
           - Use historical knowledge and context to map archaic spellings correctly
           - Examples:
             * "Yerdele" → "Yeovil" (NOT "Yardley") - historical spelling of Yeovil, Somerset
             * "Northflete" → "Northfleet" - normalize spelling but keep historical form if appropriate
             * "Wrangle" → "Wrangle" (keep as is if correct historical spelling)
           - **Extraction rules**:
             * When anglicizing place names, use the historically accurate modern equivalent
             * Cross-reference with known historical place names for the region and era
             * If uncertain between multiple candidates, prefer the one that matches the historical context
             * Consider the county and region context when normalizing place names
             * Do NOT default to the most phonetically similar modern town name
        
        **OCCUPATION EXTRACTION (CRITICAL):**
        - **You MUST extract the occupation for EVERY person when it can be determined from the text.**
        - Look carefully for occupational terms in the text associated with each person's name
        - Common occupations to look for:
          * Trades: "mercer", "skinner", "goldsmith", "tailor", "carpenter", "smith", "baker", "brewer", "butcher", "fishmonger", "draper", "grocer", "haberdasher", "ironmonger", "vintner"
          * Agricultural: "husbandman", "yeoman", "farmer"
          * Clerical/Religious: "prior", "dean", "bishop", "clerk", "chaplain", "friar", "monk", "nun"
          * Legal: "attorney", "serjeant", "apprentice"
          * Status: "citizen", "esquire", "knight", "gentleman", "merchant"
        - Extract occupations EXACTLY as written in the text (e.g., "citizen and mercer" → extract both "citizen" and "mercer")
        - If a person is described with multiple occupations/statuses, extract ALL of them
        - The occupation should be placed in TblAgent.Occupation field
        - If the text explicitly mentions an occupation for a person, you MUST extract it - do not leave it blank
        - If no occupation is mentioned in the text for a person, you may leave TblAgent.Occupation as null/empty

        B. PARTY IDENTIFICATION AND AGENT ROLES (CRITICAL - MANDATORY)
        **CRITICAL: Agent roles are MANDATORY. Every agent extracted MUST have a TblAgentRole.role field populated.**
        
        1. Identification: The first party named after the county margin is usually the Defendant. The party "to answer" is the Plaintiff.
        2. Attorneys: Capture attorneys for both sides. Note if a party appears "in their own person" vs "by attorney".
        3. Executors: Extract the Executor AND the Testator (deceased) as separate entities.
        4. Sureties/Pledges: Extract "Pledges of prosecution" and "Sureties for Law" (compurgators).
        
        **AGENT ROLE EXTRACTION (MANDATORY FOR ALL AGENTS):**
        - **EVERY agent in the Agents array MUST have a TblAgentRole.role field. This is NOT optional.**
        - You MUST identify the role for EVERY person mentioned in the case.
        - If you cannot determine a specific role, use "Other" - but you MUST assign a role.
        
        **HOW TO IDENTIFY ROLES:**
        - **Plaintiff**: The party who brings the case (usually "to answer" or "summoned to answer")
        - **Defendant**: The party who must answer the case (usually the first named party after the county margin)
        - **Debtor**: Person who owes money or is bound by a bond/obligation
        - **Creditor**: Person to whom money is owed
        - **Attorney of plaintiff**: Attorney representing the plaintiff (look for "by attorney" or "attorney of [plaintiff name]")
        - **Attorney of defendant**: Attorney representing the defendant (look for "by attorney" or "attorney of [defendant name]")
        - **Attorney of third party**: Attorney representing a third party
        - **Surety for defendant**: Person who guarantees the defendant's appearance (look for "mainpernors", "pledges", "sureties")
        - **Surety of Plaintiff**: Person who guarantees the plaintiff's appearance
        - **Surety of law (compurgator)**: Person who swears an oath in support (look for "wages his law", "compurgation")
        - **Surety other**: Other types of sureties
        - **Executor**: Person who executes a will
        - **Testator**: Person who made the will (usually deceased)
        - **Witness**: Person who witnessed an event or document
        - **Clerk**: Court clerk or administrative clerk
        - **Justice**: Justice of the court
        - **Chief justice**: Chief justice of the court
        - **Juror**: Member of the jury
        - **Essoin of plaintiff**: Person who provides an excuse for the plaintiff's absence
        - **Essoin of defendant**: Person who provides an excuse for the defendant's absence
        - **Accessory**: Person who assisted in a crime or wrong
        - **Administrator**: Person who administers an estate
        - **Arbitrator**: Person who arbitrates a dispute
        - **Auditor**: Person who audits accounts
        - **Intestator**: Person who died without a will
        - **Official**: Court official or other official
        - **Other**: Use only when no other role fits (but you MUST assign a role)
        
        **CRITICAL: Role Inference - Agent Relationships (Attorney Roles)**
        - **When you see "attorney" or "attornatum suum" (his attorney), you MUST link the attorney to their principal**
        - **Relation extraction rules**:
          * Look for patterns like: "per J. Cook attornatum suum" (by J. Cook his attorney)
          * The word "suum" (his) refers back to the subject of the sentence (plaintiff or defendant)
          * If the sentence structure is "[Plaintiff/Defendant] appeared per [Attorney Name] attornatum suum"
            → Extract the attorney with role "Attorney of [plaintiff/defendant]" (matching the principal)
          * If the text says "attorney of [Name]" explicitly, link it to that person
          * If context shows an attorney appearing for a specific party, assign the appropriate role
        - **DO NOT extract "attorney" as a generic occupation without linking it to a role**
        - **Examples**:
          * "John Smith appeared per J. Cook attornatum suum" (John Smith is defendant, J. Cook is Attorney of defendant)
          * "Plaintiff appeared by attorney, J. Cook" → J. Cook is "Attorney of plaintiff"
          * "Defendant appeared by attorney, J. Cook" → J. Cook is "Attorney of defendant"
        
        **EXTRACTION RULES:**
        - Read the ENTIRE text carefully to identify ALL people mentioned
        - For each person, determine their role based on context and relation to other parties
        - **For attorneys**: Always extract their relationship to their principal (plaintiff/defendant) - do NOT use generic "attorney" occupation alone
        - If a person appears multiple times with different roles, create separate agent entries for each role
        - When in doubt about a role, choose the most specific role that fits (e.g., "Debtor" is more specific than "Other")
        - DO NOT leave any agent without a role - this is a critical validation requirement

        C. WRIT TYPE CLASSIFICATION (REQUIRED FIELD)
        **CRITICAL: WritType is a REQUIRED field in the schema. You MUST populate it.**
        **CRITICAL: Schema location: TblCase.WritType (this is a REQUIRED STRING field)**
        
        **IMPORTANT DISTINCTION: Writ vs. Case Type**
        - **WritType**: The form of action (the legal category of the writ). This is what you extract here.
        - **CaseType** (see section G): The specific facts/sub-categories within the writ (the plea/narratio details). 
          For example, a WritType of "Trespass" may have CaseType of "Assault" or "Housebreaking". 
          A WritType of "Debt" may have CaseType of "Loan" or "Bond".
        
        Classify the writ type based on keywords in the text:
        - "Debt": Look for "plea that he render [money/chattels/grain]", "writing obligatory", "bond", "owes", "obligation"
        - "Trespass": Look for "force and arms", "against the peace", "vi et armis"
        - "Account": Look for "render reasonable account", "account", "reckoning"
        - "Detinue": Look for "unjustly detains [specific object/charter]", "detention of goods"
        - "Covenant": Look for "hold to a covenant", "covenant broken"
        - Other common writ types: "Replevin", "Waste", "Dower", etc.
        
        **EXTRACTION RULES:**
        - Read the opening of the case carefully - writ type is often stated in the first few sentences
        - Look for explicit writ type mentions: "writ of debt", "writ of trespass", etc.
        - If writ type is not explicitly stated, infer it from the nature of the claim (e.g., "force and arms" → "Trespass")
        - **SCHEMA REQUIREMENT**: This field MUST be present in the JSON output and cannot be empty
        - **NOTE**: After identifying the WritType, you MUST ALSO extract the specific CaseType (sub-category) in section G below by analyzing the narratio (the facts of the case)

        D. EVENT & DATE EXTRACTION (CRITICAL - MANDATORY)
        **CRITICAL: TblEvents is MANDATORY and must contain at least one entry when events are mentioned in the text.**
        
        **HOW TO FIND EVENTS:**
        Events are the factual occurrences that led to the legal dispute. Look for:
        1. The making of bonds, contracts, or agreements (Date + Place)
        2. Payment due dates or payment events
        3. The date of the alleged wrong/default
        4. Property transfers, sales, or gifts
        5. Any specific historical event mentioned that relates to the case
        
        **COMMON EVENT PATTERNS:**
        - "bond for [amount] made on [Date] at [Place]" → Extract as EventType: "bond", with EventDate and LocationDetails
        - "obligation for [amount] on [Date]" → Extract as EventType: "bond", with EventDate
        - "contract made at [Place] on [Date]" → Extract as EventType: "contract (not service/employment)", with EventDate and LocationDetails
        - "payment due on [Date]" → Extract as EventType: "payment", with EventDate
        - "accounting at [Place] on [Date]" → Extract as EventType: "accounting", with EventDate and LocationDetails
        - "sale of [goods] at [Place] on [Date]" → Extract as EventType: "sale of goods", with EventDate and LocationDetails
        
        **EVENT TYPE EXTRACTION:**
        - Schema location: TblEvents[].EventType (must match schema enum exactly)
        - Common event types (from schema enum):
          * "bond" - Look for: "bond", "obligation", "writing obligatory"
          * "contract (not service/employment)" - Look for: "contract", "agreement", "covenant"
          * "payment" - Look for: "payment", "paid", "render [money]"
          * "accounting" - Look for: "account", "reckoning", "accounting"
          * "sale of goods" - Look for: "sold", "sale", "bought"
          * "loan" - Look for: "lent", "loan", "borrowed"
          * "gift" - Look for: "gave", "granted", "gift"
          * "property transfer" - Look for: "granted", "conveyed", "transferred"
          * "charter" - Look for: "charter", "deed"
          * "will" - Look for: "will", "testament"
        - If multiple event types apply, create separate entries for each
        
        **EXTRACTION RULES:**
        1. Read the ENTIRE English translation carefully to identify ALL events
        2. Look for dates associated with the making of agreements, bonds, or contracts
        3. Extract the location where each event occurred (use LocationDetails)
        4. Extract the date when each event occurred (use EventDate with DateType: "occurred" or "initial")
        5. If an event mentions a value/amount, extract it in EventDetails.ValueAmount
        6. Each distinct event should be a separate entry in the TblEvents array
        7. **CRITICAL**: If the text mentions events like "bond made on [date] at [place]" or "accounting at [place] on [date]", you MUST extract them - do not skip events
        
        **DATE CONVERSION REQUIREMENTS:**
        **CRITICAL: Regnal Year Date Conversion with Accurate Feast Day Calculation**
        
        1. *Convert Regnal Years* (e.g., "6 Henry VI") to calendar years using the provided metadata context.
           - Use the Regnal Year and Calendar Year from the metadata to calculate dates accurately
           - For Henry IV (1399-1413): Year 1 = 1399 (starting Sept 30), Year 2 = 1400, etc.
           - For Henry V (1413-1422): Year 1 = 1413 (starting March 21), Year 2 = 1414, etc.
           - For Henry VI (1422-1461): Year 1 = 1422 (starting Sept 1), Year 2 = 1423, etc.
        
        2. *Convert Medieval Feast Days to ISO Dates (YYYY-MM-DD)*:
           - When you encounter feast dates like "feast of St. Michael" or "in festo sancti Michaelis", convert them to ISO format
           - **Fixed feast dates** (same calendar date every year):
             * St. Michael (Michaelmas): September 29
             * St. John the Baptist: June 24
             * St. Martin: November 11
             * Purification of the Virgin Mary (Candlemas): February 2
             * Annunciation of the Virgin Mary (Lady Day): March 25
             * All Saints: November 1
             * St. Thomas the Apostle: December 21
             * Nativity (Christmas): December 25
             * Epiphany: January 6
           
           - **Moveable feasts** (depend on Easter date for that year):
             * Easter: Variable (calculate based on regnal year calendar)
             * Pentecost (Whitsun): Variable (7 weeks after Easter = 49 days after)
             * Ascension: Variable (40 days after Easter)
             * Trinity Sunday: Variable (first Sunday after Pentecost = 56 days after Easter)
           
           - **CRITICAL**: For moveable feasts, you MUST use the correct Easter date for the specific regnal year
           - **CRITICAL**: Easter dates vary by year - do NOT use a generic approximation
           - If you cannot calculate the exact moveable feast date for the given regnal year, use the approximate date based on:
             * Easter typically falls between March 22 and April 25
             * Pentecost = Easter + 49 days
             * Ascension = Easter + 40 days (Thursday)
             * Trinity = Easter + 56 days (Sunday)
        
        3. When both regnal year and feast are given, combine them to produce a specific ISO date
           - Example: "feast of St. Michael in the 6th year of Henry VI" → "1427-09-29" (if metadata shows Henry VI year 6 = 1427)
           - Example: "Pentecost in the 2nd year of Henry IV" → Calculate Easter 1400, then add 49 days
        
        4. If you cannot determine the exact calendar year, use the format "YYYY-MM-DD" with best estimate or note uncertainty
        
        5. For moveable feasts, if precise calculation is not possible:
           - Calculate based on the regnal year calendar using approximate Easter dates
           - Format: "YYYY-MM-DD" (estimated) or use the date range if uncertain

        E. PLEADING PHASE (The Arguments) - MANDATORY
        **CRITICAL: TblPleadings is MANDATORY and must contain at least one entry.**
        
        **EXTRACTION RULES:**
        1. Read the ENTIRE English translation from start to finish
        2. Identify ALL pleading steps, even if they're implicit
        3. Break down complex sentences into separate entries
        4. If the text says "they say" or "he claims", extract it as a pleading step
        5. If the text mentions "puts himself on the country" or "wages his law", extract as an Issue
        6. DO NOT skip any legal arguments - extract every substantive claim or defense
        
        Do not summarize. Break down the legal arguments into steps. Each entry in TblPleadings must be ONE SENTENCE:
        1. The Count (Narratio): What does the plaintiff claim? (e.g., "bond for 40 marks made on [Date]").
           - Look for: "seeks that", "claims that", "plea that", "writ that"
        2. The Defense (Bar): How does the defendant respond?
           - General Issue: "Non est factum" (Not his deed), "Nil debet" (Owes nothing), "Not guilty".
           - Special Pleas:
             * "Duress/Minas": Defendant claims he was imprisoned or threatened.
             * "Payment": Defendant claims he already paid (look for "Solvit ad diem").
             * "Acquittance": Defendant shows a release form.
           - Look for: "they say", "he says", "defends", "denies"
        3. The Replication: How does the plaintiff respond to the defense?
           - Look for: "plaintiff says", "replication", responses to defense
        4. The Issue: How will this be resolved?
           - "Puts himself on the country" = Jury Trial.
           - "Wages his law" (vadiat legem) = Compurgation (oath taking).
           - Look for: "jury", "country", "wages his law", "compurgation"
        
        **IMPORTANT**: Each PleadingText entry must be a single, complete sentence. If there are multiple arguments or steps, create separate entries, each as one sentence.

        F. POSTEA & PROCESS PHASE (CRITICAL - MANDATORY)
        **CRITICAL: TblPostea is MANDATORY and must contain at least one entry.**
        **CRITICAL: Document Segmentation - Do NOT miss Postea sections**
        
        **HOW TO FIND POSTEA SECTIONS:**
        1. **Look for keywords**: "Afterwards", "Postea", "At which day", "Ad quem diem", "Et vicecomes" (and the sheriff), "Sheriff", "Precept", "Commanded", "It is considered", "Consideratum est"
        2. **Visual breaks**: Postea sections typically appear AFTER the main pleading arguments
           - Look for changes in text format or indentation
           - Postea often starts on a new line or after a paragraph break
        3. **Context indicators**: They describe procedural events: summons, adjournments, defaults, judgments
        4. **CRITICAL**: Read the ENTIRE document from start to finish - do NOT stop at the end of the pleadings
        5. **CRITICAL**: The document often continues beyond the initial pleading section
        6. Each distinct event should be a separate entry
        7. Even if the text doesn't explicitly say "Postea", extract procedural events as Postea entries
        8. **Check for continuation**: If the text appears to end abruptly, look for additional pages or continuation markers
        
        **COMMON POSTEA PATTERNS:**
        - "At which day came both [parties]" → Extract as: "At which day both parties appeared"
        - "Sheriff did not send the writ" → Extract as: "The sheriff did not send the writ"
        - "Therefore, as before, the Sheriff is commanded..." → Extract as: "The sheriff is commanded to cause jurors to come on [date]"
        - "It is considered that [Plaintiff] recover" → Extract as: "It is considered that [Plaintiff] recover [amount]"
        - "[Defendant] is in mercy" → Extract as: "[Defendant] is amerced"
        - "Plaintiff take nothing by his writ" → Extract as: "The plaintiff takes nothing by his writ (dismissed)"
        
        You must extract:
        1. Sheriff's Returns:
           - "Sheriff sent word the writ arrived too late" (tarde).
           - "Sheriff returned that [Name] is dead".
           - "Sheriff returned [Name] has nothing" (nichil).
        2. Adjournments/Continuances:
           - "Day is given here until [Date]".
           - "Nisi Prius": Look for "unless the Justices of Assize come first to [Place] on [Date]".
        3. Defaults:
           - "At which day [Defendant] did not come".
           - "Made default".
        4. Final Judgments:
           - "It is considered that [Plaintiff] recover his debt".
           - "[Defendant] is in mercy" (amerced).
           - "Plaintiff take nothing by his writ" (dismissed).
           - "Defendant to go without day" (acquitted).
        
        **IMPORTANT**: Each PosteaText entry must be a single, complete sentence. If there are multiple postea events, create separate entries, each as one sentence. Always include a Date field when a date is mentioned or can be inferred.

        G. CASE METADATA EXTRACTION (MANDATORY - CRITICAL)
        
        **Case Type (TblCaseType.CaseType) - REQUIRED:**
        **CRITICAL: This field is MANDATORY. You MUST extract at least one case type.**
        **CRITICAL: Legal Taxonomy Alignment - Distinguish Writ (form of action) from Case Type (specific facts/plea)**
        
        - Schema location: TblCaseType.CaseType (this is an ARRAY of strings)
        - **MUST analyze the NARRATIO (the facts/plea section) to identify specific sub-categories**
        - The CaseType represents the SPECIFIC facts of the case, NOT just the writ category
        - Extract ALL case types mentioned in the text based on the narrative details
        
        **CASE TYPE CLASSIFICATION RULES:**
        
        **For Trespass writs** - Analyze the narratio to distinguish sub-categories:
          * "Assault" - Look for: "assault", "beat", "struck", "wounded", "injured", "attacked"
          * "Housebreaking" - Look for: "broke into", "entered [house/property]", "housebreaking", "clausum fregit", "forcibly entered"
          * "Trespass" - Generic trespass (use when specific type cannot be determined)
          * "Imprisonment" - Look for: "imprisoned", "detained", "took and imprisoned", "captured"
          * "Theft" - Look for: "stole", "took", "carried away [goods]"
          * "Trespass (Chattels)" - Look for: "took goods", "carried away chattels", "unlawfully took"
          - **CRITICAL**: If the narratio describes assault, extract "Assault" (NOT just "Trespass")
          - **CRITICAL**: If the narratio describes housebreaking, extract "Housebreaking" (NOT just "Trespass")
          - Example: If text says "force and arms broke into the house and assaulted", extract: ["Assault", "Housebreaking"]
        
        **For Debt writs** - Analyze the narratio to distinguish sub-categories:
          * "Loan" - Look for: "lent", "loan", "borrowed", "money lent", "advanced money"
          * "Bond" - Look for: "bond", "obligation", "writing obligatory", "sealed obligation"
          * "Debt" - Generic debt (use when specific type cannot be determined)
          - **CRITICAL**: If the narratio describes a loan transaction, extract "Loan" (NOT just "Debt")
          - **CRITICAL**: If the narratio describes a bond/obligation, extract "Bond" (or keep as generic "Debt" if bond is already the WritType)
          - Example: If text says "plea that he render money lent", extract: ["Loan"]
          - Example: If text says "writing obligatory for 40 marks", extract: ["Bond"]
        
        **Other case types** (must match schema enum exactly):
          * "Account" - Look for: "render reasonable account", "reckoning", "accounting"
          * "Detinue" - Look for: "unjustly detains", "detention of goods", "detains chattels"
          * "Covenant" - Look for: "hold to a covenant", "covenant broken", "breach of covenant"
          * "Real action  / rents / damage to real estate" - Look for: "land", "tenement", "rent", "real estate", "property damage"
        
        **EXTRACTION RULES:**
        - Read the ENTIRE narratio (the Count/plaintiff's claim section) carefully
        - Identify the SPECIFIC factual allegations, not just the writ category
        - If multiple specific types apply, extract ALL of them into the array
        - Do NOT extract the WritType here - extract only the specific CaseType (sub-category)
        - Example: WritType="Trespass", CaseType=["Assault"] (NOT ["Trespass"])
        - Example: WritType="Debt", CaseType=["Loan"] (NOT ["Debt"])
        
        **Damages Claimed (TblCase.DamClaimed) - REQUIRED FIELD:**
        **CRITICAL: This field is REQUIRED in the schema. You MUST populate it.**
        **CRITICAL: You MUST check for damages. They are almost always stated in legal cases.**
        - Schema location: TblCase.DamClaimed (this is a REQUIRED STRING field)
        - Look CAREFULLY for phrases like:
          * "damages to the value of [amount]"
          * "ad valenciam [amount]"
          * "damages of [amount]"
          * "£X" or "X pounds" or "X marks" or "X shillings"
          * "100s" (means 100 shillings)
          * "40 marks"
          * "£10"
        
        **CRITICAL: Currency Unit Distinction - Marks vs. Shillings vs. Pounds**
        - **Shillings (s. or solidus)**: Look for "s." after numbers (e.g., "100s" = 100 shillings)
        - **Pounds (li. or librae)**: Look for "£", "li.", "lb.", "pounds" (e.g., "£10", "10li.", "10 pounds")
        - **Marks (m. or marc)**: Look for "m." after numbers or word "marks" (e.g., "40m.", "40 marks")
        - **CRITICAL**: Do NOT confuse "100s" (100 shillings) with "100m." (100 marks) or "100 marks"
        - **CRITICAL**: 1 mark = 13s 4d (13 shillings 4 pence), so marks and shillings are DIFFERENT units
        - **Extraction rules**:
          * If you see "100s" → Extract as "100s" (100 shillings) - NOT "100 marks"
          * If you see "100 marks" or "100m." → Extract as "100 marks" - NOT "100s"
          * Always preserve the original unit (s., m., li., £) in your extraction
          * When in doubt, look at the Latin text: "solidus" = shillings, "marc" or "marce" = marks, "libra" = pounds
        
        - Extract the EXACT amount mentioned in the original format with correct unit
        - Format examples: "100s", "£1000", "40 marks", "100 pounds", "£10 5s 3d"
        - If the text explicitly says "no damages" or "without damages", use empty string ""
        - If you cannot find any mention of damages after thorough search, use empty string ""
        - **IMPORTANT**: Damages are typically stated in the Count (plaintiff's claim). Read the entire Count section carefully.
        - **SCHEMA REQUIREMENT**: This field MUST be present in the JSON output (even if empty string "")

        OUTPUT REQUIREMENTS
        Return a JSON object matching the schema.
        
        **MANDATORY SCHEMA FIELDS (Must be populated):**
        - **TblReference**: REQUIRED object with:
          * reference: String (e.g., "CP40-562 340")
          * dateyear: Integer (Calendar Year from metadata or extracted from text)
          * term: String - MUST be one of: "Michaelmas", "Hilary", "Easter", "Trinity" (from metadata or extracted)
          * County: String - MUST match schema enum (from metadata or extracted from text/marginal annotation)
        
        - **Cases[].TblCase**: REQUIRED object with:
          * County: String - Same as TblReference.County
          * DamClaimed: String - REQUIRED (extract from text, use "" if not found after thorough search)
          * WritType: String - REQUIRED (e.g., "Trespass", "Debt", "Account", "Detinue", "Covenant") - MUST be populated
        
        - **Cases[].TblCaseType**: REQUIRED object with:
          * CaseType: Array of strings - REQUIRED (must contain at least one case type from schema enum)
        
        - **TblEvents**: REQUIRED array with at least one entry when events are mentioned in the text. Each event must have EventType (required), and should include EventDate and LocationDetails when available. Extract ALL events such as bonds, contracts, accounting sessions, payments, property transfers, etc.
        
        - **TblPleadings**: REQUIRED array with at least one entry. Each PleadingText must be ONE SENTENCE describing a pleading step (Count, Defense, Replication, or Issue).
        
        - **TblPostea**: REQUIRED array with at least one entry. Each PosteaText must be ONE SENTENCE describing a postea event (Sheriff returns, Adjournments, Defaults, or Judgments). Include Date field when available.
        
        - **Cases[].Agents**: REQUIRED array with at least one entry. Each agent MUST have:
          * TblName: Object with Christian_name, Surname, Suffix
          * **TblAgentRole: REQUIRED object with role field - THIS IS MANDATORY FOR EVERY AGENT**
            - The role field MUST be populated for every agent
            - Must be one of the enum values: "Plaintiff", "Defendant", "Debtor", "Creditor", "Attorney of plaintiff", "Attorney of defendant", "Surety for defendant", "Surety of Plaintiff", "Surety of law (compurgator)", "Executor", "Testator", "Witness", "Clerk", "Justice", "Juror", "Other", etc.
            - DO NOT leave any agent without a role - use "Other" if no specific role fits
          * TblAgent: Object with Occupation, AgentStatus (optional)
            - **Occupation: REQUIRED when mentioned in text** - Extract the occupation for each person if it can be determined from the text. Look for occupational terms like "mercer", "skinner", "husbandman", "prior", "citizen", etc. Extract EXACTLY as written. If multiple occupations/statuses are mentioned (e.g., "citizen and mercer"), extract all of them. If no occupation is mentioned in the text, you may leave this field as null/empty.
          * TblAgentStatus: Object with AgentStatus (optional)
        
        - **DATE FIELDS**: All dates in EventDate[].Date and TblPostea[].Date should be in ISO format (YYYY-MM-DD) when possible.
          Medieval feast dates MUST be converted to their calendar equivalents (e.g., "feast of St. Michael" → "1427-09-29").
        
        **EXTRACTION CHECKLIST - Before returning JSON, verify:**
        1. ✓ Did I populate TblReference.term from metadata (or extracted from text)?
        2. ✓ Did I populate TblReference.County from metadata (or extracted from text/margin)?
        3. ✓ Did I populate TblReference.dateyear from metadata (or extracted from text)?
        4. ✓ Did I populate TblCase.County (same as TblReference.County)?
        5. ✓ Did I populate TblCaseType.CaseType with at least ONE case type from the schema enum?
        6. ✓ Did I check thoroughly for Damages Claimed and populate TblCase.DamClaimed (REQUIRED field - even if empty string)?
        7. ✓ Did I extract and populate TblCase.WritType (REQUIRED field - cannot be empty)?
        8. ✓ **CRITICAL: Did I extract ALL events (bonds, contracts, accounting, payments, etc.) into TblEvents array? (This is MANDATORY when events are mentioned in the text)**
        9. ✓ Did I extract EventType, EventDate, and LocationDetails for each event in TblEvents?
        10. ✓ Did I extract at least ONE Count (what the plaintiff claims)?
        11. ✓ Did I extract at least ONE Defense (how the defendant responds)?
        12. ✓ Did I extract at least ONE Issue (how it will be resolved - jury, compurgation, etc.)?
        13. ✓ Did I extract ALL Postea events (Sheriff returns, adjournments, defaults, judgments)?
        14. ✓ Are all entries ONE SENTENCE each?
        15. ✓ Did I include dates when mentioned in Postea entries?
        16. ✓ Are all agent names correctly anglicized and validated against Latin text?
        17. ✓ **CRITICAL: Did I assign a TblAgentRole.role to EVERY agent in the Agents array? (This is MANDATORY - no agent can be without a role)**
        18. ✓ Did I identify ALL people mentioned in the text and extract them as agents with appropriate roles?
        19. ✓ **CRITICAL: Did I extract the occupation for EVERY person when it can be determined from the text? (Check TblAgent.Occupation field for each agent)**
        
        If any answer is NO, go back and extract the missing information. Pay special attention to items 1-7 (metadata fields), item 8 (events - CRITICAL), items 17-18 (agent roles - CRITICAL), and item 19 (occupations - CRITICAL).
        """

    latin_section = f"\n        Latin Reference (for verification of legal terms like 'non est factum' or 'capiatur'):\n        {latin_text}" if latin_text else ""

    prompt = f"""{static_instructions}
        {context_block}

        English Translation (Source for extraction):
        {english_text}
        {latin_section}
        """
    return [types.Part.from_text(text=prompt)]