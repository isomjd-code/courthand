#!/bin/bash
# Setup script for ScrabbleGAN training

set -e

echo "=========================================="
echo "Setting up ScrabbleGAN for Latin BHO dataset"
echo "=========================================="

# Check if we're in the right directory
if [ ! -d "scrabblegan" ]; then
    echo "Error: scrabblegan directory not found. Please run this from the project root."
    exit 1
fi

# Check if data directory exists
if [ ! -d "bootstrap_training_data/datasets/dataset_v22" ]; then
    echo "Error: bootstrap_training_data/datasets/dataset_v22 directory not found."
    exit 1
fi

# Install required packages if needed
echo ""
echo "Checking Python packages..."
python3 -c "import lmdb" 2>/dev/null || {
    echo "Installing lmdb..."
    pip install lmdb
}

python3 -c "import torch" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install torch torchvision
}

# Convert data to LMDB format
echo ""
echo "Converting data to LMDB format..."
python3 convert_to_lmdb.py --input_dir bootstrap_training_data/datasets/dataset_v22 --output_dir Datasets/LatinBHO --split train
python3 convert_to_lmdb.py --input_dir bootstrap_training_data/datasets/dataset_v22 --output_dir Datasets/LatinBHO --split val
python3 convert_to_lmdb.py --input_dir bootstrap_training_data/datasets/dataset_v22 --output_dir Datasets/LatinBHO --split test

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To train the model, run:"
echo "  cd scrabblegan"
echo "  python train.py --dataname LatinBHOtrH32 --name_prefix latin_bho"
echo ""
echo "Or use the convenience script:"
echo "  python train_scrabblegan.py --train"

