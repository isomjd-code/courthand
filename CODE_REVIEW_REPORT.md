# Code Review Report: Unused and Obsolete Code

**Date:** Generated during code review  
**Project:** Latin-BHO: CP40 Plea Roll Transcription System

## Executive Summary

This report identifies unused, obsolete, and potentially redundant code in the project. The analysis is organized by category with recommendations for cleanup.

---

## 1. Obsolete Standalone Scripts

### 1.1 One-off Test/Debug Scripts

These scripts appear to be diagnostic tools used during development and are likely no longer needed:

**High Priority for Removal:**
- `check_post_correction_files.py` - Hardcoded path check for specific case (CP40-638_305)
- `debug_request.py` - Debug script for web scraper form submission
- `test_pylaia_import.py` - Diagnostic script for Pylaia import issues
- `test_pylaia_cli_import.py` - Another Pylaia import diagnostic
- `test_model_loader_sig.py` - Model loader diagnostic
- `test_greyscale_preprocessing.py` - Preprocessing test script
- `test_inversion_detection.py` - Image inversion detection test
- `test_scraper.py` - Scraper test script
- `inspect_cli_tool.py` - Inspects Pylaia CLI tool (hardcoded path)
- `inspect_cp40_form.py` - CP40 form inspection script
- `inspect_cp40_format.py` - CP40 format inspection script
- `find_crnn.py` - Finds CRNN location in Pylaia/laia
- `find_decode_ctc.py` - Finds decode_ctc module location
- `rotation_checker.py` - Image rotation checker using Tesseract (hardcoded path)

**Recommendation:** Move to `archive/` or `scripts/dev/` directory, or delete if no longer needed.

### 1.2 Check/Validation Scripts

These may be useful for maintenance but appear to be one-off utilities:

- `check_available_images.py` - Check available images
- `check_levenshtein_filtering.py` - Check Levenshtein filtering
- `check_master_record_size.py` - Check master record sizes
- `check_model_loader.py` - Check model loader
- `check_source_material.py` - Check source material
- `check_surname.py` - Check surname functionality
- `check_symbols.py` - Check symbols
- `check_training_status.py` - Check training status

**Recommendation:** Review if these are still needed. If they're one-off diagnostic scripts, consider archiving.

### 1.3 Example/Demo Scripts

- `example_surname_search.py` - Example usage of CP40 surname scraper

**Recommendation:** Move to `examples/` directory or delete if documentation is sufficient.

---

## 2. Duplicate Functionality

### 2.1 Bayesian Surname Decoder

**Files:**
- `bayesian_surname_decoder.py` - Standalone Bayesian decoder class
- `process_htr_with_bayesian_names.py` - Full processing script with Bayesian correction
- `workflow_manager/post_correction.py` - Integrated Bayesian correction (ACTIVE)

**Analysis:**
- `workflow_manager/post_correction.py` contains the active, integrated Bayesian correction functionality
- `bayesian_surname_decoder.py` appears to be an older, standalone implementation
- `process_htr_with_bayesian_names.py` is a standalone script that may have been superseded by the workflow manager

**Recommendation:** 
- **Remove:** `bayesian_surname_decoder.py` (functionality integrated into `post_correction.py`)
- **Review:** `process_htr_with_bayesian_names.py` - Check if it's still used. If not, remove or archive.

### 2.2 String Probability Calculators

**Files:**
- `calculate_string_probability.py` - Full-featured calculator (1157+ lines)
- `calculate_string_probability_simple.py` - Simpler version (271+ lines)
- `loss_for_string.py` - Example/template code for calculating CTC loss

**Analysis:**
- Both calculators appear to do similar things (calculate log probability of strings using Pylaia)
- `loss_for_string.py` is just example/template code

**Recommendation:**
- **Consolidate:** Keep one version (preferably the simpler one if it meets requirements)
- **Remove:** `loss_for_string.py` (example code, not production)

### 2.3 Surname Scrapers

**Files:**
- `cp40_surname_scraper.py` - Full-featured scraper (490+ lines)
- `cp40_surname_scraper_simple.py` - Simpler version (427+ lines)

**Analysis:**
- `cp40_full_scraper.py` imports from `cp40_surname_scraper_simple.py`, so the simple version is actively used
- `cp40_surname_scraper.py` may be an older version

**Recommendation:**
- **Keep:** `cp40_surname_scraper_simple.py` (actively used)
- **Review:** `cp40_surname_scraper.py` - Check if it has features not in the simple version. If not, remove.

### 2.4 Preprocessing Wrappers

**Files:**
- `preprocess_lines.py` - Legacy CLI wrapper for `line_preprocessor`
- `preprocess_lines_greyscale.py` - Legacy CLI wrapper for `line_preprocessor_greyscale`

**Analysis:**
- Both are thin wrappers that just call the module's `main()` function
- The actual functionality is in the `line_preprocessor/` and `line_preprocessor_greyscale/` modules

**Recommendation:**
- **Keep:** These are useful CLI entry points, but consider documenting them as wrappers
- **Alternative:** Could be removed if the modules are always called directly

---

## 3. Potentially Obsolete Scripts

### 3.1 Data Processing Scripts

**Files:**
- `clean_step1_lines.py` - Cleans step1.json files with ground truth surname correction
- `cleanup_checkpoint.py` - Cleanup checkpoint files
- `cleanup_forenames.py` - Cleanup forenames
- `fix_cp40_years.py` - Fix CP40 years in database
- `backfill_baselines.py` - Backfill baselines

**Recommendation:** Review if these are one-off data fixes or ongoing utilities. If one-off, archive after use.

### 3.2 Training-Related Scripts

**Files:**
- `manual_retrain.py` - Manual retraining script
- `retrain.py` - Retraining script
- `resume_training.py` - Resume training script
- `resume_training.sh` - Shell script for resuming training
- `train_model.sh` - Training shell script
- `debug_training.py` - Debug training script

**Analysis:**
- Training scripts may be actively used for model development
- `debug_training.py` is likely a diagnostic tool

**Recommendation:**
- **Keep:** Active training scripts if still in use
- **Remove/Archive:** `debug_training.py` if no longer needed

### 3.3 Utility Scripts

**Files:**
- `get_example_from_master.py` - Gets example from master_record.json (hardcoded path)
- `query_case_638_305.py` - Query specific case (hardcoded)
- `analyze_case_638_305.py` - Analyze specific case (hardcoded)
- `copy_images_from_onedrive.py` - Copy images utility
- `scrape_index.py` - Scrape index utility
- `schema_generator.py` - Schema generator
- `validation_report.py` - Validation report (may be superseded by `report_generator.py`)

**Recommendation:**
- **Remove:** Scripts with hardcoded paths (`get_example_from_master.py`, `query_case_638_305.py`, `analyze_case_638_305.py`)
- **Review:** Others - check if still needed

### 3.4 Report Generation

**Files:**
- `generate_pdf_report.py` - PDF report generator
- `report_generator.py` - Report generator (may be the active one)
- `validation_report.py` - Validation report generator

**Analysis:**
- `report_generator/` directory contains the main report generation module
- Need to check if these standalone scripts are still used

**Recommendation:** Review which report generator is actively used and consolidate.

---

## 4. Unused Imports

### 4.1 workflow_manager/workflow.py

**Import Usage Analysis:**
- `signal` - **USED** - Used for process termination (SIGTERM, SIGKILL) in subprocess management
- `platform` - **USED** - Used for Windows-specific subprocess handling
- `sqlite3` - **USED** - Used for database queries in `_load_list_from_db()` method

**Conclusion:** All imports in `workflow_manager/workflow.py` are actively used.

### 4.2 Other Modules

**Recommendation:** Run a static analysis tool (e.g., `vulture`, `pylint`) to identify unused imports across all modules.

**Note:** Initial review of `workflow_manager/workflow.py` shows all imports are actively used. A comprehensive static analysis would be needed for other modules.

---

## 5. Legacy/Commented Code

### 5.1 Commented-Out Code

**Recommendation:** Search for large blocks of commented-out code and either:
- Remove if obsolete
- Uncomment and fix if still needed
- Document why it's commented if it's kept for reference

### 5.2 Legacy Entry Points

**Files:**
- `workflow_manager.py` - Marked as "Legacy entrypoint preserved for backwards compatibility"

**Analysis:**
- This is a thin wrapper that imports from `workflow_manager` package
- May be kept for backwards compatibility

**Recommendation:** Document the migration path and consider deprecation timeline.

---

## 6. Configuration Files

### 6.1 Training Configuration

**Files:**
- `working_create_config.yaml` - Training config
- `working_train_config.yaml` - Training config

**Recommendation:** Review if these are still active or if they should be in a `configs/` directory.

---

## 7. Summary of Recommendations

### High Priority (Safe to Remove)

1. **Test/Debug Scripts:**
   - All `test_*.py`, `debug_*.py`, `check_*.py`, `inspect_*.py` scripts (unless actively used)
   - `find_crnn.py`, `find_decode_ctc.py`
   - `rotation_checker.py` (hardcoded path)

2. **Duplicate Functionality:**
   - `bayesian_surname_decoder.py` (superseded by `post_correction.py`)
   - `loss_for_string.py` (example code)
   - `cp40_surname_scraper.py` (if `cp40_surname_scraper_simple.py` is sufficient)

3. **Hardcoded One-off Scripts:**
   - `get_example_from_master.py`
   - `query_case_638_305.py`
   - `analyze_case_638_305.py`
   - `check_post_correction_files.py`

4. **Example Scripts:**
   - `example_surname_search.py` (move to examples/ or remove)

### Medium Priority (Review and Consolidate)

1. **String Probability Calculators:**
   - Consolidate `calculate_string_probability.py` and `calculate_string_probability_simple.py`

2. **Report Generators:**
   - Review `generate_pdf_report.py`, `report_generator.py`, `validation_report.py` and consolidate

3. **Training Scripts:**
   - Review training-related scripts and archive obsolete ones

### Low Priority (Keep but Document)

1. **Legacy Wrappers:**
   - `preprocess_lines.py`, `preprocess_lines_greyscale.py` (document as wrappers)

2. **Legacy Entry Point:**
   - `workflow_manager.py` (document migration path)

---

## 8. Action Items

1. **Create `archive/` directory** for obsolete but potentially useful code
2. **Create `scripts/dev/` directory** for development/diagnostic scripts
3. **Create `examples/` directory** for example scripts
4. **Run static analysis** to identify unused imports
5. **Review and consolidate** duplicate functionality
6. **Document** which scripts are actively maintained vs. legacy

---

## 9. Files to Review Before Removal

Before removing any files, verify:
- They're not imported by other modules
- They're not referenced in documentation
- They're not used in CI/CD pipelines
- They don't contain unique functionality not found elsewhere

**Suggested command to check imports:**
```bash
grep -r "import.*filename\|from.*filename" . --include="*.py"
```

---

## 10. Implementation Status

✅ **COMPLETED** - All high priority recommendations have been implemented.

See `CLEANUP_SUMMARY.md` for details of files removed.

**Total files removed:** 30 files
- 15 test/debug scripts
- 8 check scripts
- 3 duplicate functionality files
- 3 hardcoded one-off scripts
- 1 example script

---

## Notes

- This analysis is based on static code review. Some scripts may be used in ways not immediately apparent.
- Some "obsolete" scripts may be kept for reference or historical purposes.
- Consider version control history before removing files to understand their purpose.
- **All high priority deletions have been completed and verified.**

