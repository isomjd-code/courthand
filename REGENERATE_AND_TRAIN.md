# How to Regenerate Dataset and Start Training

## Step-by-Step Instructions

### Step 1: Backfill Baselines (Optional but Recommended)
First, backfill baseline information from existing Kraken JSON files:

```bash
# Dry run to see what will be updated
python backfill_baselines.py --dry-run

# Actually update the metadata files
python backfill_baselines.py --verbose
```

This will update all metadata.json files with baseline information from Kraken JSON files.

### Step 2: Regenerate Dataset
Regenerate the dataset with the fixed format and baselines:

```bash
python manual_retrain.py --regenerate-dataset --model-version 2
```

This will:
- Regenerate `dataset_v2` with correct format (relative paths + tokenized text)
- Use Kraken baselines where available
- Include all required symbols (`<ctc>`, `<space>`, `<unk>`)
- Apply 95% similarity filtering

### Step 3: Start Training
Start the training process:

```bash
python manual_retrain.py --retrain
```

This will:
- Use the regenerated dataset
- Start training model v2
- Save checkpoints as training progresses

### Alternative: Do Everything at Once

You can also do steps 2 and 3 together:

```bash
python manual_retrain.py --regenerate-dataset --retrain
```

## What to Expect

### During Dataset Regeneration:
- You'll see progress as lines are processed
- Filtering statistics (e.g., "Filtered out X lines with similarity < 95%")
- Format verification

### During Training:
- Model architecture creation (if needed)
- Training progress with epochs
- Validation CER (Character Error Rate) monitoring
- Checkpoints saved automatically

## Monitoring Training

Training logs are saved to:
- `bootstrap_training_data/pylaia_models/model_v2/experiment/train-crnn.log`
- Application logs: `logs/debug.log`

You can monitor progress:
```bash
tail -f bootstrap_training_data/pylaia_models/model_v2/experiment/train-crnn.log
```

## Troubleshooting

If training fails:
1. Check the error log: `logs/train_pylaia_model_error.log`
2. Verify dataset format: `head bootstrap_training_data/datasets/dataset_v2/train.txt`
3. Check symbols file: `head bootstrap_training_data/datasets/dataset_v2/syms.txt`

