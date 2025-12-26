# Latin-BHO: CP40 Plea Roll Transcription System

A comprehensive workflow system for transcribing and extracting structured data from 15th-century English legal documents (Court of Common Pleas CP40 plea rolls). This project combines Handwritten Text Recognition (HTR), AI-powered vision models, and paleographic expertise to produce accurate diplomatic transcriptions and structured legal case data.

## Overview

This system processes images of medieval legal manuscripts through a multi-stage pipeline:

1. **HTR Processing**: Uses Kraken for line segmentation and PyLaia for initial text recognition
2. **Post-Correction**: AI-powered correction and named entity extraction using Gemini 3 Flash Preview with Bayesian correction
3. **Stitching**: Merges transcriptions from multiple images per case
4. **Expansion**: Expands medieval abbreviations to full forms
5. **Translation**: Translates Latin text to English
6. **Indexing**: Structured entity extraction (parties, locations, dates, legal details)
7. **Validation**: Cross-references with ground truth databases and validates extracted entities
8. **Report Generation**: Creates comprehensive reports and web-viewable outputs

## Features

- **Multi-Model HTR**: Integration with both Kraken (segmentation) and PyLaia (recognition) for robust text recognition
- **AI-Powered Transcription**: Uses Google Gemini 3 Flash Preview for all vision and text processing tasks
- **Bayesian Correction**: Advanced named entity correction using Bayesian inference with reference databases
- **Diplomatic Accuracy**: Preserves medieval abbreviations, ligatures, and orthographic conventions
- **Structured Extraction**: Extracts parties, locations, dates, writs, and case details
- **Database Validation**: Cross-references surnames and place names against reference databases
- **Confidence Scoring**: Calculates confidence levels for extracted entities based on multiple attestations
- **AI Certainty Reporting**: Reports confidence levels (High, Medium, Low, Very Low) for personal and place names
- **Optimal Matching**: Uses Hungarian algorithm for optimal bipartite matching of agents and events
- **Semantic Similarity**: Advanced matching using Google Gemini embeddings with Levenshtein fallback
- **Batch Processing**: Efficient batch processing of multiple images per case
- **Validation Reports**: Comprehensive PDF reports comparing AI extraction with ground truth
- **Line Image Extraction**: Visual verification with processed line images in validation reports
- **Web Viewer**: HTML-based viewer for reviewing transcriptions and extracted data
- **Ground Truth Integration**: Validates results against known ground truth data with exact matching

## Requirements

### System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for HTR models)
- Linux/Unix environment (for shell scripts and HTR tools)
- Sufficient disk space for models and processing outputs

### Python Environments

This project uses three separate Python environments:

1. **Main Environment**: Core workflow and AI integration
2. **Kraken Environment**: Kraken HTR toolkit
3. **PyLaia Environment**: PyLaia HTR toolkit

### External Dependencies

- **Kraken**: For document segmentation and layout analysis
- **PyLaia**: For handwritten text recognition
- **Google Gemini API**: For vision and text processing (requires paid API key for Gemini 3 Flash Preview)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd latin_bho
```

### 2. Set Up Python Environments

Run the setup script to create all required virtual environments:

```bash
bash setup.sh
```

This will create:
- `venv_main/` - Main workflow environment (requires `requirements.txt`)
- `venv_kraken/` - Kraken HTR environment  
- `venv_pylaia/` - PyLaia HTR environment

**Note**: The setup script expects a `requirements.txt` file for the main environment. If this file doesn't exist, you'll need to create it with the main project dependencies, or install them manually after creating the virtual environment.

### 3. Configure Settings

Edit `workflow_manager/settings.py` to configure:

- **API Keys**: Set `GEMINI_API_KEY` environment variable with a paid API key for Gemini 3 Flash Preview access (recommended). Alternatively, you can set it directly in `settings.py`, but using an environment variable is more secure.
- **Environment Paths**: Update `PYLAIA_ENV` and `KRAKEN_ENV` to match your virtual environment locations (default paths are in `~/.projects/`)
- **Model Paths**: Ensure model files are in the `model_v10/` directory:
  - `epoch=322-lowest_va_cer.ckpt` (PyLaia checkpoint, CER 25.9%)
  - `syms.txt` (PyLaia symbols file)
  - `model` (PyLaia architecture file)
- **Database Paths**: Verify paths to:
  - `cp40_database_new.sqlite` (surname database)
  - `places_data.db` (places database)

### 4. Install HTR Tools

Install Kraken and PyLaia in their respective environments:

```bash
# Activate Kraken environment
source venv_kraken/bin/activate
pip install -r requirements_kraken.txt
deactivate

# Activate PyLaia environment
source venv_pylaia/bin/activate
pip install -r requirements_pylaia.txt
deactivate
```

## Usage

### Basic Workflow

1. **Prepare Images**: Place manuscript images in the `input_images/` directory
   - Images should be named with case identifiers (e.g., `CP40-565_481a.jpg`)
   - Multiple images per case are automatically grouped

2. **Run the Workflow**:

```bash
# Activate main environment
source venv_main/bin/activate

# Run with default settings
python workflow_manager.py

# Force reprocessing (ignore existing results)
python workflow_manager.py --force

# Process without image uploads (text-only mode)
python workflow_manager.py --no-images

# Rerun from post-correction step (keeps Kraken/PyLaia results)
python workflow_manager.py --rerun-from-post-pylaia

# Specify custom input directory
python workflow_manager.py --dir /path/to/images
```

### Workflow Steps

The system processes each case through these stages:

1. **Image Grouping**: Groups images by case identifier
2. **HTR Processing**: 
   - Kraken segmentation (line detection)
   - Line preprocessing
   - PyLaia recognition
3. **Post-Correction**: AI-powered correction using Gemini 3 Flash Preview with Bayesian named entity correction
4. **Step 2a - Stitching**: Merges transcriptions from multiple images into a single diplomatic text
5. **Step 2b - Expansion**: Expands medieval abbreviations to full Latin forms
6. **Step 3 - Translation**: Translates expanded Latin text to English
7. **Step 4 - Indexing**: Generates structured case data with confidence scores (parties, locations, dates, legal details)
8. **Validation**: Cross-references with ground truth and databases
9. **Report Generation**: Creates HTML reports and web viewer files

### Output Structure

Results are saved in `cp40_processing/output/` organized by case:

```
cp40_processing/output/
└── CP40-565_481/
    ├── final_index.json                    # Complete structured case data
    ├── CP40-565 481a.jpg_post_correction.json  # Post-correction results
    ├── CP40-565 481a.jpg_step1.json       # Diplomatic transcription (compatibility)
    ├── step2a_merged.json                  # Merged diplomatic transcription
    ├── step2b_latin_expanded.txt           # Expanded Latin text
    ├── step3_english.txt                   # English translation
    ├── step4_index.json                    # Structured extraction
    └── [htr intermediate files]
```

### Ground Truth Validation

Extract and compare with ground truth data:

```bash
python ground_truth.py
```

### Bootstrap Training

The project includes a bootstrap training workflow that uses Gemini 3 Flash Preview to correct HTR transcriptions, which are then used to retrain PyLaia models:

```bash
python bootstrap_training_runner.py [--force]
```

The `--force` flag reprocesses all images even if they've already been processed.

**Bootstrap Training Features:**
- Automatic checkpointing for resuming after restarts
- Statistics tracking for all processing elements
- Model versioning with separate storage for each retrained model
- Retraining every 1,000 corrected lines
- All data artifacts stored in `bootstrap_training_data/`

See `bootstrap_training/README.md` for detailed documentation.

### CP40 Surname Scraper

The project includes a web scraper for the Medieval Genealogy CP40 Index Order Search:

```bash
# Search for a surname
python cp40_surname_scraper_simple.py Smith

# Search with year range
python cp40_surname_scraper_simple.py Black --year-from 1400 --year-to 1500

# Search with wildcards
python cp40_surname_scraper_simple.py "Bl*" --output black_results.json
```

See `CP40_SCRAPER_README.md` for detailed documentation.

### Report Generation

Generate validation reports comparing AI extraction with ground truth:

```bash
# Generate PDF validation report
python report_generator.py

# Reports are automatically generated during workflow execution
# Output: comparison_report_CP40-{roll}_{rotulus}.tex (LaTeX source)
#         comparison_report_CP40-{roll}_{rotulus}.pdf (compiled PDF)
```

**Validation Report Features:**
- Field-level comparison with similarity scores
- Field-specific similarity thresholds:
  - Agent Name and Event Place: > 95% required for match
  - Case Details: > 78% required for match
  - All other fields: > 90% required for match
- "Names to Check" section listing low-confidence names with visual line images
- AI certainty badges for personal and place names
- Executive summary with accuracy metrics by category
- Detailed field-by-field comparison tables

## Project Structure

```
latin_bho/
├── workflow_manager/          # Core workflow system
│   ├── workflow.py            # Main WorkflowManager class
│   ├── image_grouper.py      # Image grouping logic
│   ├── post_correction.py     # AI post-correction with Bayesian correction
│   ├── prompt_builder.py      # AI prompt construction
│   ├── schemas.py             # JSON schemas for extraction
│   ├── paleography.py         # Paleographic matching utilities
│   ├── settings.py            # Configuration and logging
│   └── utils.py               # Helper functions
├── bootstrap_training/        # Bootstrap training workflow
│   ├── workflow.py            # BootstrapTrainingManager class
│   └── README.md              # Bootstrap training documentation
├── ground_truth/              # Ground truth extraction
│   ├── extractor.py           # Database extraction with exact matching
│   └── query.py               # SQL query utilities
├── report_generator/          # Report generation
│   ├── report.py              # Main report generation
│   ├── sections.py            # LaTeX section builders
│   ├── similarity.py          # Similarity calculation and matching
│   └── text_utils.py          # Text formatting utilities
├── line_preprocessor/         # Line preprocessing utilities
│   ├── processing.py          # Line extraction and processing
│   └── geometry.py            # Geometric transformations
├── line_preprocessor_greyscale/  # Greyscale line preprocessing
├── pylaia_dataset/            # PyLaia dataset generation
│   └── generator.py           # Dataset creation from HTR results
├── input_images/              # Input manuscript images
├── cp40_processing/           # Processing workspace
│   └── output/                # Generated outputs
├── bootstrap_training_data/   # Bootstrap training data artifacts
├── model_v10/                 # HTR model files (current version)
├── webviewer/                 # Web viewer files
├── logs/                      # Application logs
├── cp40_database_new.sqlite   # Surname reference database
├── places_data.db             # Places reference database
├── cp40_surname_scraper_simple.py  # CP40 surname web scraper
├── cp40_full_scraper.py       # Full CP40 scraper
├── requirements.txt           # Main project dependencies
├── requirements_kraken.txt    # Kraken dependencies
├── requirements_pylaia.txt    # PyLaia dependencies
├── requirements_scraper.txt   # Scraper dependencies
└── setup.sh                   # Setup script
```

## Configuration

### Environment Variables

Set these in `workflow_manager/settings.py` or as environment variables:

- `GEMINI_API_KEY`: Google Gemini API key (required, must be a paid key for Gemini 3 Flash Preview)
- `MODEL_VISION`: Vision model name (default: `gemini-3-flash-preview`)
- `MODEL_TEXT`: Text model name (default: `gemini-3-flash-preview`)

### API Configuration

The system uses Google Gemini 3 Flash Preview for all operations:
- Requires a paid API key (set via `GEMINI_API_KEY` environment variable)
- Uses non-batch mode for all API calls
- Includes automatic retry logic with exponential backoff for timeout/504 errors
- File uploads are cached to avoid re-uploading the same images

### Directory Paths

Default paths (configurable in `settings.py`):

- `WORK_DIR`: `cp40_processing/`
- `IMAGE_DIR`: `input_images/`
- `OUTPUT_DIR`: `cp40_processing/output/`
- `LOG_DIR`: `logs/`

## Output Format

### Final Index JSON

The `final_index.json` contains structured case data:

```json
{
  "caseIdentifier": {
    "rollNumber": "565",
    "rotulusNumber": "481",
    "county": "Middlesex",
    "countySource": "marginal_annotation"
  },
  "cases": [{
    "parties": [...],
    "caseDetails": {
      "writ": {...},
      "date": {...},
      "locations": [...]
    }
  }],
  "confidence": {...},
  "validation": {...},
  "extracted_entities": {
    "personal_names": [...],
    "place_names": [...]
  },
  "source_material": [...]
}
```

### Validation Reports

Validation reports are generated as LaTeX files and compiled to PDF:
- **Location**: `cp40_processing/output/CP40-{roll}_{rotulus}/comparison_report_CP40-{roll}_{rotulus}.tex`
- **Sections**:
  - Executive Summary with accuracy metrics
  - Names to Check (low-confidence names with line images)
  - Case Record Comparison
  - Detailed Field Comparison
  - Full Text Reconstruction

Reports include:
- Similarity scores for each field
- Match/mismatch indicators
- AI certainty badges for names
- Visual line images for verification
- Field-specific similarity thresholds

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure `GEMINI_API_KEY` environment variable is set with a paid API key
2. **HTR Failures**: Verify Kraken and PyLaia environments are activated correctly
3. **GPU Issues**: Check CUDA installation if HTR models fail to load
4. **Database Errors**: Ensure SQLite database files exist at configured paths
5. **Path Issues**: Verify all directory paths in `settings.py` are correct

### Logs

Check log files for detailed processing information:
- `logs/debug.log`: Main workflow processing logs
- `logs/report_generator.log`: Report generation logs
- Per-case logs in `cp40_processing/output/{case_id}/`:
  - `step2a_thoughts.log`: AI reasoning for merging transcriptions
  - `step2b_thoughts.log`: AI reasoning for abbreviation expansion
  - `step3_thoughts.log`: AI reasoning for translation
  - `step4_thoughts.log`: AI reasoning for structured extraction

## Recent Improvements

- **Gemini 3 Flash Preview**: Upgraded to Gemini 3 Flash Preview for all vision and text tasks
- **Bayesian Correction**: Implemented Bayesian named entity correction using reference databases
- **7-Step Pipeline**: Expanded workflow to include post-correction, stitching, expansion, translation, and indexing
- **Validation Reports**: Enhanced PDF reports with field-specific similarity thresholds and visual verification
- **Optimal Matching**: Uses Hungarian algorithm for optimal agent/event assignment
- **AI Certainty**: Confidence level reporting (High, Medium, Low, Very Low) for extracted names
- **Line Image Processing**: Integrated line image extraction and processing for visual verification
- **Exact Database Matching**: Improved ground truth extraction with exact SQL matching
- **Semantic Matching**: Enhanced person matching with weighted semantic similarity and Levenshtein fallback
- **Retry Logic**: Automatic retry with exponential backoff for API timeouts and 504 errors
- **Bootstrap Training**: Added automated training workflow using Gemini-corrected transcriptions
- **Code Cleanup**: Removed 30 obsolete files including test scripts, debug tools, and duplicate functionality (see `CODE_REVIEW_REPORT.md` and `CLEANUP_SUMMARY.md`)

## Additional Documentation

- **`CODE_REVIEW_REPORT.md`**: Comprehensive code review identifying unused and obsolete code
- **`CLEANUP_SUMMARY.md`**: Summary of cleanup operations removing 30 obsolete files
- **`CP40_SCRAPER_README.md`**: Documentation for the CP40 surname web scraper
- **`bootstrap_training/README.md`**: Detailed documentation for the bootstrap training workflow
- **`model_architecture.md`**: Information about the HTR model architecture
- **`database_schema_documentation.md`**: Database schema documentation

## Contributing

This project processes historical legal documents. When contributing:

- Maintain accuracy in paleographic conventions
- Preserve diplomatic transcription standards
- Test with sample images before submitting changes
- Document any schema or workflow modifications
- Follow field-specific similarity thresholds in validation reports
- Avoid creating duplicate functionality (check existing modules first)
- Remove or archive obsolete test/debug scripts after use

## License

Copyright Joshua David Isom, 2025. All rights reserved.

## Acknowledgments

- Uses [Kraken](https://github.com/mittagessen/kraken) for document segmentation
- Uses [PyLaia](https://github.com/jpuigcerver/PyLaia) for handwritten text recognition
- Powered by Google Gemini AI models
- Processes Court of Common Pleas (CP40) plea rolls from 15th-century England

## Citation

If you use this system in your research, please cite appropriately.
