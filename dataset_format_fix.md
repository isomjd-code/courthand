# Dataset Format Fix

## Problem
Training was failing with error:
```
WARNING laia.data.text_image_from_text_table_dataset] No text found for image ID '...', ignoring example...
ValueError: not enough values to unpack (expected 2, got 0)
```

This happened because the validation set had **no valid examples** after filtering - all images were missing their text entries.

## Root Cause
The dataset generator was creating files in the wrong format:

**What it was creating:**
- `train.txt` - Just image paths (absolute paths)
- `train_text.txt` - Just text labels (separate file)

**What Pylaia expects:**
- `train.txt` - Combined format: `{relative_image_path} {tokenized_text}`

Pylaia couldn't match image paths with text because they were in separate files, and the format didn't match what the loader expected.

## Solution
Updated `bootstrap_training/dataset_generator.py` to:

1. **Tokenize text properly**: Convert text to space-separated characters with `<space>` tokens
   - Example: `"hello world"` → `"h e l l o <space> w o r l d"`

2. **Use relative image paths**: Convert absolute paths to relative paths from the images directory
   - Example: `/path/to/images/val/abc123.png` → `val/abc123.png`

3. **Write combined format**: Write `{image_id} {tokenized_text}` in the main `.txt` file
   - Example: `val/abc123.png h e l l o <space> w o r l d`

## Next Steps

**You need to regenerate the dataset** for the fix to take effect:

```bash
# Regenerate dataset for a specific model version
python manual_retrain.py --regenerate-dataset --model-version 2
```

Or if you're using the bootstrap training workflow, it will automatically regenerate when retraining is triggered.

You can also use the Python API directly:
```python
from bootstrap_training.dataset_generator import generate_training_dataset

generate_training_dataset(
    corrected_lines_dir="bootstrap_training_data",
    output_dir="bootstrap_training_data/datasets/dataset_v1",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    image_height=128,
    max_levenshtein_distance=10,  # Or whatever you were using
)
```

## Verification

After regenerating, check the format:
```bash
head -3 bootstrap_training_data/datasets/dataset_v1/val.txt
```

Should show:
```
val/abc123.png t o k e n i z e d <space> t e x t
val/def456.png m o r e <space> t e x t
...
```

Not:
```
/home/absolute/path/to/image.png
/home/another/path.png
```

