# Workflow Manager Improvements - Summary of Changes

This document summarizes all improvements made to the workflow manager based on the 8 suggestions for better extraction accuracy.

## Files Modified

1. `workflow_manager/prompt_builder.py` - Enhanced Step 4 indexing prompt with detailed instructions
2. `workflow_manager/post_correction.py` - Added capital letter disambiguation instructions
3. `workflow_manager/workflow.py` - Changed thinking level from LOW to MEDIUM for Step 4

---

## Improvement 1: Legal Taxonomy Alignment (Case Type Classification)

### Problem
AI was extracting broad writ types (e.g., "Trespass", "Debt") but failing to classify specific case types (e.g., "Assault" vs "Housebreaking" under Trespass, "Loan" vs "Bond" under Debt).

### Changes Made
**File:** `workflow_manager/prompt_builder.py`

Added distinction between WritType and CaseType, with detailed instructions to analyze the narratio:

```python
# Added to build_step4_prompt() around line 790
**IMPORTANT DISTINCTION: Writ vs. Case Type**
- **WritType**: The form of action (the legal category of the writ)
- **CaseType**: The specific facts/sub-categories within the writ (the plea/narratio details)

# Enhanced Case Type classification section (around line 994)
**For Trespass writs** - Analyze the narratio to distinguish sub-categories:
  * "Assault" - Look for: "assault", "beat", "struck", "wounded"
  * "Housebreaking" - Look for: "broke into", "entered [house/property]", "housebreaking"
  - **CRITICAL**: If narratio describes assault, extract "Assault" (NOT just "Trespass")

**For Debt writs** - Analyze the narratio to distinguish sub-categories:
  * "Loan" - Look for: "lent", "loan", "borrowed", "money lent"
  * "Bond" - Look for: "bond", "obligation", "writing obligatory"
  - **CRITICAL**: If narratio describes a loan, extract "Loan" (NOT just "Debt")
```

---

## Improvement 2: Paleographic Character Disambiguation (Capital Letters)

### Problem
HTR model confuses similar capital letters in "Court Hand" script (C vs R/G, G vs C, K vs H), leading to surname errors like "Walter Roke" instead of "Walter Cok".

### Changes Made
**Files:** `workflow_manager/prompt_builder.py` and `workflow_manager/post_correction.py`

Added specific instructions for capital letter disambiguation:

**In prompt_builder.py (around line 688):**
```python
3. **PALEOGRAPHIC CHARACTER DISAMBIGUATION (CAPITAL LETTERS) - CRITICAL:**
   **The HTR model may confuse similar capital letters in "Court Hand" script. Pay careful attention to:**
   - **C vs. R/G**: Capital C can be misread as R or G (e.g., "Walter Cok" might be misread as "Walter Roke")
   - **G vs. C**: Capital G can be misread as C (e.g., "Richard Goold" might be misread as "Richard Coolde")
   - **K vs. H**: Capital K can be misread as H (e.g., "Robert Kelme" might be misread as "Robert Holme")
```

**In post_correction.py (around line 240):**
```python
- **PALEOGRAPHIC CHARACTER DISAMBIGUATION (CAPITAL LETTERS) - CRITICAL:**
  The HTR model may confuse similar capital letters in "Court Hand" script. Pay careful attention when correcting:
  - **C vs. R/G**: Capital C can be misread as R or G (e.g., "Cok" might be misread as "Roke")
  - **G vs. C**: Capital G can be misread as C (e.g., "Goold" might be misread as "Coolde")
  - **K vs. H**: Capital K can be misread as H (e.g., "Kelme" might be misread as "Holme")
```

---

## Improvement 3: Historical Place Name Normalization (Gazetteer Integration)

### Problem
AI correctly transcribed archaic spellings but mapped them to wrong modern locations (e.g., "Yerdele" → "Yardley, Somerset" instead of "Yeovil, Somerset").

### Changes Made
**File:** `workflow_manager/prompt_builder.py` (around line 708)

Added instructions for historical place name normalization:

```python
7. **HISTORICAL PLACE NAME NORMALIZATION (GAZETTEER INTEGRATION) - CRITICAL:**
   **CRITICAL: Map archaic/medieval spellings to historically accurate modern equivalents**
   - Do NOT simply use phonetically similar modern place names
   - Use historical knowledge and context to map archaic spellings correctly
   - Examples:
     * "Yerdele" → "Yeovil" (NOT "Yardley") - historical spelling of Yeovil, Somerset
     * "Northflete" → "Northfleet" - normalize spelling but keep historical form if appropriate
   - **Extraction rules**:
     * When anglicizing place names, use the historically accurate modern equivalent
     * Cross-reference with known historical place names for the region and era
     * Do NOT default to the most phonetically similar modern town name
```

---

## Improvement 4: Currency Unit Distinction (Marks vs. Shillings)

### Problem
AI confused monetary units, specifically between "marks" and "shillings" (e.g., extracting "100 marks" when GT says "100s" meaning 100 shillings).

### Changes Made
**File:** `workflow_manager/prompt_builder.py` (around line 1050)

Added strict currency unit distinction logic:

```python
**CRITICAL: Currency Unit Distinction - Marks vs. Shillings vs. Pounds**
- **Shillings (s. or solidus)**: Look for "s." after numbers (e.g., "100s" = 100 shillings)
- **Pounds (li. or librae)**: Look for "£", "li.", "lb.", "pounds" (e.g., "£10", "10li.")
- **Marks (m. or marc)**: Look for "m." after numbers or word "marks" (e.g., "40m.", "40 marks")
- **CRITICAL**: Do NOT confuse "100s" (100 shillings) with "100m." (100 marks) or "100 marks"
- **CRITICAL**: 1 mark = 13s 4d (13 shillings 4 pence), so marks and shillings are DIFFERENT units
- **Extraction rules**:
  * If you see "100s" → Extract as "100s" (100 shillings) - NOT "100 marks"
  * If you see "100 marks" or "100m." → Extract as "100 marks" - NOT "100s"
  * Always preserve the original unit (s., m., li., £) in your extraction
```

---

## Improvement 5: Regnal Year Date Conversion

### Problem
Date conversions from regnal years to modern calendar dates drifted by several days/weeks, especially for moveable feasts like Pentecost or Trinity.

### Changes Made
**File:** `workflow_manager/prompt_builder.py` (around line 872)

Enhanced date conversion instructions with regnal calendar details:

```python
**DATE CONVERSION REQUIREMENTS:**
**CRITICAL: Regnal Year Date Conversion with Accurate Feast Day Calculation**

1. *Convert Regnal Years* (e.g., "6 Henry VI") to calendar years using the provided metadata context.
   - For Henry IV (1399-1413): Year 1 = 1399 (starting Sept 30), Year 2 = 1400, etc.
   - For Henry V (1413-1422): Year 1 = 1413 (starting March 21), Year 2 = 1414, etc.
   - For Henry VI (1422-1461): Year 1 = 1422 (starting Sept 1), Year 2 = 1423, etc.

2. *Convert Medieval Feast Days to ISO Dates (YYYY-MM-DD)*:
   - **Fixed feast dates** (same calendar date every year): St. Michael (Sept 29), etc.
   - **Moveable feasts** (depend on Easter date for that year):
     * Easter: Variable (calculate based on regnal year calendar)
     * Pentecost (Whitsun): Variable (7 weeks after Easter = 49 days after)
     * Ascension: Variable (40 days after Easter)
     * Trinity Sunday: Variable (first Sunday after Pentecost = 56 days after Easter)
   
   - **CRITICAL**: For moveable feasts, you MUST use the correct Easter date for the specific regnal year
   - **CRITICAL**: Easter dates vary by year - do NOT use a generic approximation
```

---

## Improvement 6: Document Segmentation (Missing "Postea")

### Problem
AI sometimes failed to extract procedural history (Postea) appearing at the end of records, leading to missing data fields.

### Changes Made
**File:** `workflow_manager/prompt_builder.py` (around line 949)

Strengthened Postea extraction instructions:

```python
F. POSTEA & PROCESS PHASE (CRITICAL - MANDATORY)
**CRITICAL: Document Segmentation - Do NOT miss Postea sections**

**HOW TO FIND POSTEA SECTIONS:**
1. **Look for keywords**: "Afterwards", "Postea", "At which day", "Ad quem diem", "Et vicecomes" (and the sheriff), "Sheriff", "Precept", "Commanded", "It is considered", "Consideratum est"
2. **Visual breaks**: Postea sections typically appear AFTER the main pleading arguments
   - Look for changes in text format or indentation
   - Postea often starts on a new line or after a paragraph break
3. **CRITICAL**: Read the ENTIRE document from start to finish - do NOT stop at the end of the pleadings
4. **CRITICAL**: The document often continues beyond the initial pleading section
5. **Check for continuation**: If the text appears to end abruptly, look for additional pages or continuation markers
```

---

## Improvement 7: County Identification Logic (Margin vs. Text)

### Problem
AI misidentified counties, likely by reading wrong cues or defaulting to common locations instead of using marginal annotations.

### Changes Made
**File:** `workflow_manager/prompt_builder.py` (around line 635)

Improved county identification prioritization:

```python
elif county_source == "not_found" or county_name == "UNKNOWN":
    county_note += """
    
    **CRITICAL: County Identification Priority**
    Since no marginal county was found, you MUST extract the county from the venue line in the text:
    - Look for: "X summonitus fuit ad respondendum Y de [County]" (X was summoned to answer Y concerning [County])
    - The county in the venue line is the authoritative source when marginal annotation is missing
    - Do NOT default to a common location or assume the county
    - If the venue line specifies a county, use that value for TblReference.County and TblCase.County"""
```

---

## Improvement 8: Role Inference (Agent Relationships)

### Problem
AI extracted occupation (e.g., "attorney") but failed to link the agent to their principal (e.g., "Attorney for plaintiff" vs "Attorney for defendant").

### Changes Made
**File:** `workflow_manager/prompt_builder.py` (around line 780)

Added relation extraction rules for attorneys:

```python
**CRITICAL: Role Inference - Agent Relationships (Attorney Roles)**
- **When you see "attorney" or "attornatum suum" (his attorney), you MUST link the attorney to their principal**
- **Relation extraction rules**:
  * Look for patterns like: "per J. Cook attornatum suum" (by J. Cook his attorney)
  * The word "suum" (his) refers back to the subject of the sentence (plaintiff or defendant)
  * If the sentence structure is "[Plaintiff/Defendant] appeared per [Attorney Name] attornatum suum"
    → Extract the attorney with role "Attorney of [plaintiff/defendant]" (matching the principal)
  * If the text says "attorney of [Name]" explicitly, link it to that person
- **DO NOT extract "attorney" as a generic occupation without linking it to a role**
- **Examples**:
  * "John Smith appeared per J. Cook attornatum suum" (John Smith is defendant, J. Cook is Attorney of defendant)
  * "Plaintiff appeared by attorney, J. Cook" → J. Cook is "Attorney of plaintiff"
```

---

## Additional Change: Thinking Level Enhancement

### Problem
Complex extraction tasks needed more reasoning capacity.

### Changes Made
**File:** `workflow_manager/workflow.py` (line 3215)

Changed thinking level from LOW to MEDIUM for Step 4 indexing:

```python
# Before:
thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level="LOW")

# After:
thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level="MEDIUM")
```

This allows the AI model to use medium-level thinking for the complex indexing step, which should improve accuracy for all the enhanced extraction tasks.

---

## Summary

All 8 improvements have been successfully integrated into the workflow manager:

1. ✅ Legal Taxonomy Alignment - Enhanced case type classification
2. ✅ Paleographic Character Disambiguation - Capital letter correction instructions
3. ✅ Historical Place Name Normalization - Gazetteer-style mapping instructions
4. ✅ Currency Unit Distinction - Marks vs. Shillings vs. Pounds
5. ✅ Regnal Year Date Conversion - Improved feast day calculations
6. ✅ Document Segmentation - Enhanced Postea extraction
7. ✅ County Identification - Improved margin vs. text prioritization
8. ✅ Role Inference - Attorney relationship extraction

**Bonus:** Thinking level increased from LOW to MEDIUM for Step 4 indexing.

These changes guide the AI model to extract more accurate and structured information from medieval legal documents, addressing the specific issues identified in the validation reports.
