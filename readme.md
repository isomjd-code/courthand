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
- `venv/` - Main workflow environment (install dependencies manually)
- `venv_kraken/` - Kraken HTR environment  
- `venv_pylaia/` - PyLaia HTR environment

**Note**: The main environment dependencies should be installed manually. Install required packages such as `google-generativeai`, `python-dotenv`, and other core dependencies. See the individual requirements files for each environment.

### 3. Configure API Keys

**Important**: The system requires a paid Google Gemini API key for Gemini 3 Flash Preview access.

#### Option A: Using .env file (Recommended)

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << 'EOF'
GEMINI_API_KEY=your_actual_api_key_here
EOF
```

Install `python-dotenv` to enable automatic loading:

```bash
pip install python-dotenv
```

The `workflow_manager/settings.py` file automatically loads `.env` files if `python-dotenv` is installed.

#### Option B: Environment Variable

Set the environment variable directly:

```bash
# In WSL/Linux/Mac
export GEMINI_API_KEY=your_actual_api_key_here

# Or add to ~/.bashrc for persistence
echo 'export GEMINI_API_KEY=your_actual_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

See `API_KEYS_SETUP.md` for detailed setup instructions and security best practices.

### 4. Configure Settings

Edit `workflow_manager/settings.py` to configure:

- **Environment Paths**: Update `PYLAIA_ENV` and `KRAKEN_ENV` to match your virtual environment locations (default paths are in `~/.projects/`)
- **Model Paths**: The system uses models from `workflow_active_model/` directory (active model) or `model_v10/` (fallback):
  - `workflow_active_model/` - Active model directory (updated by bootstrap training)
  - `model_v10/` - Fallback model directory with:
    - `epoch=322-lowest_va_cer.ckpt` (PyLaia checkpoint, CER 25.9%)
    - `syms.txt` (PyLaia symbols file)
    - `model` (PyLaia architecture file)
- **Database Paths**: Verify paths to:
  - `cp40_database_new.sqlite` (surname database)
  - `places_data.db` (places database)

### 5. Install HTR Tools

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
│   ├── geometry.py            # Geometric transformations
│   ├── parser.py              # Configuration parsing
│   ├── runner.py              # Preprocessing runner
│   └── config.py              # Preprocessing configuration (dimensions, binarization, etc.)
├── line_preprocessor_greyscale/  # Greyscale line preprocessing
│   ├── processing.py          # Greyscale processing
│   ├── runner.py              # Greyscale runner
│   └── config.py              # Greyscale configuration
├── pylaia_dataset/            # PyLaia dataset generation
│   └── generator.py           # Dataset creation from HTR results
├── input_images/              # Input manuscript images
├── cp40_processing/           # Processing workspace
│   └── output/                # Generated outputs
├── bootstrap_training_data/   # Bootstrap training data artifacts
├── model_v10/                 # HTR model files (fallback version)
├── workflow_active_model/     # Active HTR model directory (updated by bootstrap training)
├── webviewer/                 # Web viewer files
├── logs/                      # Application logs
├── cp40_database_new.sqlite   # Surname reference database
├── places_data.db             # Places reference database
├── cp40_surname_scraper_simple.py  # CP40 surname web scraper
├── cp40_full_scraper.py       # Full CP40 scraper
├── requirements_kraken.txt    # Kraken dependencies
├── requirements_pylaia.txt    # PyLaia dependencies
├── requirements_scraper.txt   # Scraper dependencies
├── setup.sh                   # Setup script
├── .env                       # API keys (not in git, see API_KEYS_SETUP.md)
└── API_KEYS_SETUP.md          # API key setup guide
```

## Configuration

### Environment Variables

Set these in `workflow_manager/settings.py` or as environment variables:

- `GEMINI_API_KEY`: Google Gemini API key (required, must be a paid key for Gemini 3 Flash Preview)
- `MODEL_VISION`: Vision model name (default: `gemini-3-flash-preview`)
- `MODEL_TEXT`: Text model name (default: `gemini-3-flash-preview`)

### API Configuration

The system uses Google Gemini 3 Flash Preview for all operations:
- **Requires a paid API key** (set via `GEMINI_API_KEY` environment variable or `.env` file)
- Uses non-batch mode for all API calls
- **Automatic retry logic**: Exponential backoff for timeout/504 errors (up to 5 retries)
- **API Timeout**: 30 minutes (1,800,000 ms) for API calls and file uploads
- **File upload caching**: Uploaded files are cached to avoid re-uploading the same images
- **Cost**: $0.25 per million input tokens, $1.50 per million output tokens
- **Environment variable loading**: Automatically loads `.env` file if `python-dotenv` is installed

### Directory Paths

Default paths (configurable in `settings.py`):

- `WORK_DIR`: `cp40_processing/`
- `IMAGE_DIR`: `input_images/`
- `OUTPUT_DIR`: `cp40_processing/output/`
- `LOG_DIR`: `logs/`
- `MODEL_DIR`: `model_v10/` (fallback)
- `ACTIVE_MODEL_DIR`: `workflow_active_model/` (active model used by workflow)
- `SURNAME_DB_PATH`: `cp40_database_new.sqlite`
- `PLACE_DB_PATH`: `places_data.db`

### Line Preprocessing Configuration

Line preprocessing parameters are configured in `line_preprocessor/config.py`:

- **Image Dimensions**: Minimum line size (8x8), target height (128px), max width (6000px)
- **Binarization**: Sauvola algorithm with configurable window size and sensitivity
- **Deslanting**: Angle range for text deslanting correction
- **Morphological Operations**: Dilation kernel size and iterations for text thickening
- **Baseline Positioning**: Target Y position for baseline in normalized images
- **Bounding Box Extension**: Left extension for additional context (150px default)

See `line_preprocessor/config.py` for all configurable parameters.

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

1. **API Key Errors**: 
   - Ensure `GEMINI_API_KEY` environment variable is set with a paid API key
   - Check that `.env` file exists and contains the key (if using python-dotenv)
   - Verify the key is a paid key (required for Gemini 3 Flash Preview)
   - See `API_KEYS_SETUP.md` for detailed troubleshooting

2. **HTR Failures**: 
   - Verify Kraken and PyLaia environments are activated correctly
   - Check that virtual environment paths in `settings.py` match your setup
   - Ensure model files exist in `workflow_active_model/` or `model_v10/`

3. **GPU Issues**: 
   - Check CUDA installation if HTR models fail to load
   - Verify GPU drivers are up to date
   - Models can run on CPU but will be slower

4. **Database Errors**: 
   - Ensure SQLite database files exist at configured paths
   - Verify `cp40_database_new.sqlite` and `places_data.db` are in the project root
   - Check file permissions if database access fails

5. **Path Issues**: 
   - Verify all directory paths in `settings.py` are correct
   - Ensure `input_images/` directory exists
   - Check that `cp40_processing/output/` is writable

6. **API Timeout Errors**:
   - The system automatically retries up to 5 times with exponential backoff
   - Timeout is set to 30 minutes for large file uploads
   - Check network connectivity if timeouts persist

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

### Core System Updates
- **Gemini 3 Flash Preview**: Upgraded to Gemini 3 Flash Preview for all vision and text tasks
- **Bayesian Correction**: Implemented Bayesian named entity correction using reference databases
- **7-Step Pipeline**: Expanded workflow to include post-correction, stitching, expansion, translation, and indexing
- **Environment Variable Support**: Added `.env` file support with automatic loading via `python-dotenv`
- **Active Model Directory**: Introduced `workflow_active_model/` for bootstrap-trained models
- **Retry Logic**: Automatic retry with exponential backoff (up to 5 retries) for API timeouts and 504 errors
- **Extended API Timeout**: Increased to 30 minutes for large file uploads and processing

### Extraction Accuracy Improvements
- **Legal Taxonomy Alignment**: Enhanced case type classification (distinguishes WritType vs CaseType, sub-categories like "Assault" vs "Housebreaking")
- **Paleographic Character Disambiguation**: Capital letter correction instructions for Court Hand script (C vs R/G, G vs C, K vs H)
- **Historical Place Name Normalization**: Gazetteer-style mapping instructions for medieval place names
- **Currency Unit Distinction**: Improved extraction of Marks vs. Shillings vs. Pounds
- **Regnal Year Date Conversion**: Enhanced feast day calculations for date conversion
- **Document Segmentation**: Improved Postea extraction and case boundary detection
- **County Identification**: Enhanced margin vs. text prioritization for county extraction
- **Role Inference**: Attorney relationship extraction improvements
- **Thinking Level**: Increased from LOW to MEDIUM for Step 4 indexing

### Validation & Reporting
- **Validation Reports**: Enhanced PDF reports with field-specific similarity thresholds and visual verification
- **Optimal Matching**: Uses Hungarian algorithm for optimal agent/event assignment
- **AI Certainty**: Confidence level reporting (High, Medium, Low, Very Low) for extracted names
- **Line Image Processing**: Integrated line image extraction and processing for visual verification
- **Exact Database Matching**: Improved ground truth extraction with exact SQL matching
- **Semantic Matching**: Enhanced person matching with weighted semantic similarity and Levenshtein fallback

### Training & Infrastructure
- **Bootstrap Training**: Added automated training workflow using Gemini-corrected transcriptions
- **Code Cleanup**: Removed 30 obsolete files including test scripts, debug tools, and duplicate functionality (see `CODE_REVIEW_REPORT.md` and `CLEANUP_SUMMARY.md`)

See `WORKFLOW_IMPROVEMENTS_DIFF.md` for detailed documentation of all 8 extraction accuracy improvements.

## Additional Documentation

- **`API_KEYS_SETUP.md`**: Comprehensive guide for setting up API keys securely
- **`CODE_REVIEW_REPORT.md`**: Comprehensive code review identifying unused and obsolete code
- **`CLEANUP_SUMMARY.md`**: Summary of cleanup operations removing 30 obsolete files
- **`CP40_SCRAPER_README.md`**: Documentation for the CP40 surname web scraper
- **`bootstrap_training/README.md`**: Detailed documentation for the bootstrap training workflow
- **`WORKFLOW_IMPROVEMENTS_DIFF.md`**: Detailed documentation of all 8 extraction accuracy improvements
- **`model_architecture.md`**: Information about the HTR model architecture
- **`database_schema_documentation.md`**: Database schema documentation
- **`REGENERATE_AND_TRAIN.md`**: Guide for regenerating datasets and retraining models

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
