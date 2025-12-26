# Bootstrap Training Workflow

This module implements a bootstrap training workflow for Pylaia models where Gemini 3 Flash Preview acts as the teacher, correcting HTR transcriptions that are then used to train improved Pylaia models.

## Overview

The workflow processes images through the following pipeline:

1. **Rotation Detection**: Uses Gemini 2.5 Pro to detect the clockwise rotation angle needed for normal reading orientation (currently disabled)
2. **Image Rotation**: Rotates images on disk if necessary (currently disabled)
3. **HTR Processing**: Runs Kraken segmentation and Pylaia HTR using the current model version
4. **Database Lookup**: Queries `cp40_database_new.sqlite` for index data matching the image's roll and rotulus numbers
5. **Gemini Correction**: Sends imperfect HTR transcript, image, bounding boxes, and database index data to Gemini 3 Flash Preview for correction (uses batch API when processing multiple images)
6. **Storage**: Saves corrected transcriptions tied to original Kraken segmentation polygons
7. **Retraining**: Every 1,000 corrected lines, retrains the Pylaia model using all accumulated corrected data

## Key Features

- **Checkpointing**: All intermediate results are checkpointed for resuming after restarts
- **Statistics**: Tracks statistics on each element (images processed, rotations, API calls, etc.)
- **Model Versioning**: Each retrained model is versioned and stored separately
- **Architecture Reuse**: After the first retrain, subsequent models reuse the architecture from the previous version
- **Data Artifacts**: All data artifacts are stored in `bootstrap_training_data/`

## Usage

```bash
python bootstrap_training_runner.py [--force]
```

The `--force` flag reprocesses all images even if they've already been processed.

## Directory Structure

```
bootstrap_training_data/
├── checkpoint.json              # State for resuming
├── statistics.json              # Processing statistics
├── corrected_lines/             # Corrected line data by image
│   └── <image_basename>/
│       └── metadata.json
├── htr_work/                    # Intermediate HTR processing files
│   └── <image_basename>/
│       ├── kraken.json
│       ├── htr.txt
│       └── lines/
├── datasets/                    # Generated training datasets
│   └── dataset_v<N>/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       └── images/
└── pylaia_models/               # Trained model versions
    └── model_v<N>/
        ├── model                # Architecture file
        ├── syms.txt
        ├── train_config.yaml
        └── experiment/          # Training checkpoints
```

## Model Training

- **First Retrain (v1)**: Uses architecture from `models/model` (epoch=220 model)
- **Subsequent Retrains (v2+)**: Reuses architecture from previous model version
- **Training Config**: Based on `working_train_config.yaml` with dataset-specific paths
- **Architecture Config**: Based on `working_create_config.yaml` (only for v1)

## Correction Format

Gemini 3 Flash Preview corrections follow these rules:
- Use single straight apostrophe (') to indicate abbreviated letters
- Only characters: A-Z, a-z, &, and pilcrow (¶)
- Do NOT expand abbreviations
- Preserve original Kraken segmentation polygons
- Corrections are returned as JSON mapping line keys to corrected text

## Resuming

The workflow automatically resumes from the last checkpoint. To restart from scratch, delete `bootstrap_training_data/checkpoint.json`.

## Statistics

Statistics are tracked in `bootstrap_training_data/statistics.json`:
- `images_processed`: Total images processed
- `images_rotated`: Images that required rotation
- `total_lines_processed`: Total lines from HTR
- `total_lines_corrected`: Total lines corrected by Gemini
- `gemini_rotation_calls`: Number of rotation detection API calls
- `gemini_correction_calls`: Number of correction API calls
- `model_retrains`: Number of model retraining cycles

