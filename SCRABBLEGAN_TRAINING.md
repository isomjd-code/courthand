# ScrabbleGAN Training Guide for Latin BHO Dataset

This guide explains how to train a ScrabbleGAN model using your existing Latin BHO dataset.

## Overview

ScrabbleGAN is a semi-supervised GAN for generating handwritten text. This setup converts your PyLaia-format dataset to LMDB format (required by ScrabbleGAN) and trains the model.

## Prerequisites

1. **Python packages**: Install required packages:
   ```bash
   pip install lmdb torch torchvision pillow tqdm
   ```

2. **Data**: Ensure you have your dataset in `bootstrap_training_data/datasets/dataset_v22/` with:
   - `train.txt`, `val.txt`, `test.txt` (PyLaia format: `image_path tokenized_text`)
   - `images/train/`, `images/val/`, `images/test/` directories with PNG images

## Quick Start

### Option 1: Automated Setup and Training

```bash
# Convert data and train in one go
python train_scrabblegan.py --convert-data --train
```

### Option 2: Step by Step

#### Step 1: Convert Data to LMDB Format

Convert your PyLaia format data to LMDB format:

```bash
# Convert all splits
python convert_to_lmdb.py --input_dir bootstrap_training_data/datasets/dataset_v22 --output_dir Datasets/LatinBHO --split train
python convert_to_lmdb.py --input_dir bootstrap_training_data/datasets/dataset_v22 --output_dir Datasets/LatinBHO --split val
python convert_to_lmdb.py --input_dir bootstrap_training_data/datasets/dataset_v22 --output_dir Datasets/LatinBHO --split test
```

Or use the setup script:
```bash
bash setup_scrabblegan.sh
```

This will create LMDB files in `Datasets/LatinBHO/train/`, `Datasets/LatinBHO/val/`, and `Datasets/LatinBHO/test/`.

#### Step 2: Train the Model

```bash
cd scrabblegan
python train.py --dataname LatinBHOtrH32 --name_prefix latin_bho --dataset_mode text --model ScrabbleGAN --input_nc 1 --resolution 32 --labeled
```

Or use the convenience script:
```bash
python train_scrabblegan.py --train
```

## Dataset Configuration

The dataset has been configured in:
- `scrabblegan/data/dataset_catalog.py` - Maps dataset names to paths
- `scrabblegan/data/alphabets.py` - Defines the character alphabet

Available dataset names:
- `LatinBHOtrH32` - Training set
- `LatinBHOvalH32` - Validation set  
- `LatinBHOteH32` - Test set

## Training Options

Key training parameters (see `scrabblegan/options/train_options.py` for full list):

- `--dataname`: Dataset name (e.g., `LatinBHOtrH32`)
- `--name_prefix`: Experiment name prefix
- `--niter`: Number of epochs (default: 100)
- `--niter_decay`: Number of epochs to decay learning rate (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.0002)
- `--beta1`: Adam beta1 (default: 0.5)
- `--gb_alpha`: Balance between recognizer and discriminator loss (default: 0.1)
- `--seed`: Random seed

Example with custom parameters:
```bash
cd scrabblegan
python train.py --dataname LatinBHOtrH32 --name_prefix latin_bho --niter 200 --batch_size 32 --lr 0.0001
```

## Data Format

### Input Format (PyLaia)
- Text files: `{relative_image_path} {tokenized_text}`
- Example: `train/im000000.png P o u n t f r e i t <space> c l ̅ i c i`
- Images: PNG format in `images/train/`, `images/val/`, `images/test/`
- Default dataset location: `bootstrap_training_data/datasets/dataset_v22/`

### Output Format (LMDB)
- LMDB database with keys:
  - `image-000000001`, `image-000000002`, ... (TIFF image bytes)
  - `label-000000001`, `label-000000002`, ... (text labels)
  - `num-samples` (total number of samples)

## Model Outputs

Training outputs are saved in:
- `scrabblegan/checkpoints/{name_prefix}/` - Model checkpoints
- `scrabblegan/results/{name_prefix}/` - Generated images and visualizations
- Training logs and loss plots

## Semi-Supervised Training

For semi-supervised training (using unlabeled data), see `scrabblegan/train_semi_supervised.py`:

```bash
cd scrabblegan
python train_semi_supervised.py --dataname LatinBHOtrH32 --unlabeled_dataname LatinBHOteH32 --disjoint
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the correct directory:
```bash
cd scrabblegan
python train.py ...
```

### LMDB Creation Fails
- Check that image paths in your text files are correct
- Ensure images exist at the specified paths
- Check disk space (LMDB files can be large)

### Out of Memory
- Reduce `--batch_size`
- Reduce image resolution (though 32 is already quite small)
- Use fewer workers in data loading

### Training Doesn't Start
- Verify LMDB files were created successfully
- Check that `Datasets/LatinBHO/train/` contains the LMDB database
- Ensure the dataset name matches what's in `dataset_catalog.py`

## Monitoring Training

Training progress is logged and visualized. Check:
- Console output for loss values
- `scrabblegan/results/{name_prefix}/` for generated images
- Loss plots in the results directory

## Next Steps

After training:
1. Use the trained model to generate handwritten text
2. Fine-tune hyperparameters based on results
3. Consider semi-supervised training for better performance
4. Generate synthetic data for augmentation

## References

- ScrabbleGAN paper: [CVPR 2020](https://www.amazon.science/publications/scrabblegan-semi-supervised-varying-length-handwritten-text-generation)
- Original repository: https://github.com/amzn/convolutional-handwriting-gan

