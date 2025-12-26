#!/bin/bash
# Quick setup script for API keys in WSL
# Usage: ./setup_api_key_wsl.sh

set -e

echo "=== WSL API Key Setup ==="
echo ""

# Check if .env already exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Edit .env manually if needed."
        exit 0
    fi
fi

# Get API key from user
echo "Enter your Gemini API key:"
read -s GEMINI_KEY
echo ""

if [ -z "$GEMINI_KEY" ]; then
    echo "❌ Error: API key cannot be empty"
    exit 1
fi

# Ask about free keys
echo "Do you have free API keys for load distribution? (optional, comma-separated)"
read -p "Enter free keys (or press Enter to skip): " FREE_KEYS

# Create .env file
cat > .env << EOF
# Primary Gemini API key (paid key required for Gemini 3 Flash Preview)
GEMINI_API_KEY=$GEMINI_KEY
EOF

if [ ! -z "$FREE_KEYS" ]; then
    echo "GEMINI_FREE_API_KEYS=$FREE_KEYS" >> .env
fi

echo ""
echo "✅ .env file created successfully!"
echo ""

# Check if python-dotenv is installed
if python3 -c "import dotenv" 2>/dev/null; then
    echo "✅ python-dotenv is already installed"
else
    echo "📦 Installing python-dotenv..."
    pip install python-dotenv
    echo "✅ python-dotenv installed"
fi

echo ""
echo "🔍 Verifying setup..."
if python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); key = os.environ.get('GEMINI_API_KEY'); print('✅ API key loaded successfully!' if key else '❌ API key not found!')" 2>/dev/null; then
    echo ""
    echo "🎉 Setup complete! Your API key is configured."
else
    echo ""
    echo "⚠️  Warning: Could not verify API key. Please check manually."
fi

echo ""
echo "To use in a new terminal session, the .env file will be automatically loaded"
echo "if python-dotenv is installed (which it now is)."

