#!/bin/bash
# Script to complete the repository migration

# Stage and commit .gitignore changes
git add .gitignore
git commit -m "Update .gitignore to track only .py, .sh, .html, .md files everywhere and .txt files in root only"

# Push to the new repository
git push -u origin main

echo "Repository migration complete!"
echo "Remote is now set to: https://github.com/isomjd-code/courthand-transcription.git"

