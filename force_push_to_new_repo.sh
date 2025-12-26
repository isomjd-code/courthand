#!/bin/bash
# Script to force push (overwrites remote - use with caution!)

# Force push to overwrite remote repository
# WARNING: This will overwrite any content on the remote
git push -u origin main --force

echo "Force push complete! Remote repository has been overwritten."

