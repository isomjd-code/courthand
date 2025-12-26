#!/bin/bash
# Script to add workflow_active_model to version control with Git LFS

# Initialize Git LFS if not already done
git lfs install

# Add .ckpt files to Git LFS tracking
git lfs track "*.ckpt"

# Add the workflow_active_model directory
git add workflow_active_model/

# Check status
git status

echo "Files added. Review the status above and commit when ready."

