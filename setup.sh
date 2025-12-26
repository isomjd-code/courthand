#!/bin/bash

# 1. Create Main Environment
echo "Setting up Main Environment..."
python3 -m venv venv_main
source venv_main/bin/activate
pip install -r requirements.txt
deactivate

# 2. Create Kraken Environment
echo "Setting up Kraken Environment..."
python3 -m venv venv_kraken
source venv_kraken/bin/activate
pip install -r requirements_kraken.txt
deactivate

# 3. Create PyLaia Environment
echo "Setting up PyLaia Environment..."
python3 -m venv venv_pylaia
source venv_pylaia/bin/activate
pip install -r requirements_pylaia.txt
deactivate

echo "Setup Complete. Run with: source venv_main/bin/activate && python workflow_manager.py"