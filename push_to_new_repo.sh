#!/bin/bash
# Script to pull remote changes and push to the new repository

# Fetch and see what's on remote
git fetch origin

# Pull with rebase to integrate remote changes
git pull origin main --rebase

# If rebase has conflicts, resolve them, then:
# git rebase --continue

# Push to the new repository
git push -u origin main

echo "Repository migration complete!"

