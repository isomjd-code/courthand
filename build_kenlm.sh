#!/bin/bash
# Build KenLM from source to get command-line tools (lmplz, build_binary)

set -e

echo "=========================================="
echo "Building KenLM from source"
echo "=========================================="

# Install dependencies
echo ""
echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev

# Clone or update KenLM repository
KENLM_DIR="$HOME/kenlm"
if [ -d "$KENLM_DIR" ]; then
    echo ""
    echo "KenLM directory exists, updating..."
    cd "$KENLM_DIR"
    git pull
else
    echo ""
    echo "Cloning KenLM repository..."
    git clone https://github.com/kpu/kenlm.git "$KENLM_DIR"
    cd "$KENLM_DIR"
fi

# Build KenLM
echo ""
echo "Building KenLM..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# Check if binaries were created
if [ -f "bin/lmplz" ] && [ -f "bin/build_binary" ]; then
    echo ""
    echo "=========================================="
    echo "KenLM built successfully!"
    echo "=========================================="
    echo ""
    echo "Binaries are located at:"
    echo "  $KENLM_DIR/build/bin/lmplz"
    echo "  $KENLM_DIR/build/bin/build_binary"
    echo ""
    echo "To use these tools, add to your PATH:"
    echo "  export PATH=\$PATH:$KENLM_DIR/build/bin"
    echo ""
    echo "Or add to ~/.bashrc for persistence:"
    echo "  echo 'export PATH=\$PATH:$KENLM_DIR/build/bin' >> ~/.bashrc"
    echo ""
else
    echo ""
    echo "Error: Binaries not found after build"
    exit 1
fi

