# API Keys Setup Guide

## Security Best Practices

This project follows security best practices for storing API keys:

### ✅ DO:
- Store API keys in environment variables
- Use a `.env` file for local development (never commit it!)
- Use `.env.example` as a template (commit this, but without real keys)
- Set environment variables in production using secure methods (CI/CD secrets, cloud provider secrets managers, etc.)
- Rotate API keys regularly
- Use different keys for different environments (dev, staging, production)

### ❌ DON'T:
- Hardcode API keys in source code
- Commit `.env` files to version control
- Share API keys in chat, email, or documentation
- Use the same key across multiple projects
- Leave keys in git history (if you did, rotate them immediately!)

## Setup Instructions

### 1. Create a `.env` file

Create a `.env` file in the project root (it's already in `.gitignore`):

```bash
# Primary Gemini API key (paid key required for Gemini 3 Flash Preview)
GEMINI_API_KEY=your_actual_api_key_here

# Optional: Free API keys for load distribution (comma-separated)
# Used by cp40_place_latinizer.py and cp40_forename_latinizer.py
GEMINI_FREE_API_KEYS=key1,key2,key3,key4
```

### 2. Load environment variables

**Note:** The `workflow_manager/settings.py` file now automatically loads `.env` files if `python-dotenv` is installed. For other methods, see below.

#### Option A: Use python-dotenv (Recommended)

Install python-dotenv:
```bash
pip install python-dotenv
```

Then add this at the top of your main scripts (before importing settings):
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file
```

#### Option B: Export in WSL/Linux/Mac (Current Session Only)

For the current terminal session:
```bash
export GEMINI_API_KEY=your_actual_api_key_here
```

To make it persistent in WSL, add to your `~/.bashrc` or `~/.zshrc`:
```bash
# Add this line to ~/.bashrc (or ~/.zshrc if using zsh)
echo 'export GEMINI_API_KEY=your_actual_api_key_here' >> ~/.bashrc
source ~/.bashrc  # Reload the file
```

**WSL Quick Setup:**
```bash
# Navigate to your project
cd ~/projects/latin_bho

# Create .env file
cat > .env << 'EOF'
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_FREE_API_KEYS=key1,key2,key3,key4
EOF

# Install python-dotenv (recommended)
pip install python-dotenv
```

#### Option C: Set in PowerShell (Windows Host)
```powershell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```
Note: This won't work in WSL - use Option B instead.

#### Option D: Set in system environment variables
- **Windows Host**: System Properties → Environment Variables (won't be available in WSL)
- **WSL/Linux**: Add to `~/.bashrc` or `~/.zshrc` (see Option B)

### 3. Verify setup

#### Quick Test in WSL Terminal:
```bash
# Check if environment variable is set
echo $GEMINI_API_KEY

# Or test with Python
python3 -c "import os; key = os.environ.get('GEMINI_API_KEY'); print(f'✓ API key loaded (length: {len(key)})') if key else print('✗ API key not found!')"
```

#### Test in Python Script:
```python
import os
key = os.environ.get('GEMINI_API_KEY')
if key:
    print(f"✓ API key loaded (length: {len(key)})")
else:
    print("✗ API key not found!")
```

#### Test with Your Project:
```bash
# Try importing settings (will error if key is missing)
cd ~/projects/latin_bho
python3 -c "from workflow_manager import settings; print('✓ Settings loaded successfully!')"
```

## Files Updated

The following files have been updated to use environment variables:

1. **workflow_manager/settings.py** - Main settings file
2. **cp40_place_latinizer.py** - Place latinizer with free key pool support
3. **cp40_forename_latinizer.py** - Forename latinizer with free key pool support
4. **report_generator/config.py** - Report generator configuration

## Production Deployment

For production environments, use secure secret management:

- **AWS**: AWS Secrets Manager or Parameter Store
- **Azure**: Azure Key Vault
- **GCP**: Secret Manager
- **Docker**: Docker secrets or environment variables
- **Kubernetes**: Kubernetes secrets
- **CI/CD**: GitHub Secrets, GitLab CI/CD variables, etc.

Example for Docker:
```dockerfile
# Don't hardcode in Dockerfile!
# Use docker-compose or runtime environment
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
```

## If You've Already Committed Keys

If API keys were previously committed to git:

1. **Rotate the keys immediately** - Generate new keys from your API provider
2. **Remove from git history** (if the repo is private):
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch workflow_manager/settings.py" \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** (only if repo is private and you've coordinated with team)
4. **Update all environments** with new keys

## Additional Resources

- [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [12 Factor App: Config](https://12factor.net/config)
- [Python-dotenv Documentation](https://pypi.org/project/python-dotenv/)

