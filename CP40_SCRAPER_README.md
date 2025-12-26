# CP40 Surname Scraper

Web scraper for the Medieval Genealogy CP40 Index Order Search:
https://www.medievalgenealogy.org.uk/aalt/cp40_search.php

## Quick Start

### Run in WSL/Linux Terminal (not PowerShell!)

Make sure you're in your WSL terminal with the pylaia-env activated:

```bash
# If not already activated
source venv/bin/activate
# or
conda activate pylaia-env
```

### 1. Test the Scraper

```bash
python test_scraper.py
```

### 2. Search for a Surname

```bash
# Basic search
python cp40_surname_scraper_simple.py Smith

# Search with year range
python cp40_surname_scraper_simple.py Black --year-from 1400 --year-to 1500

# Search with wildcards
python cp40_surname_scraper_simple.py "Bl*" --output black_results.json

# With county filter
python cp40_surname_scraper_simple.py Smith --county Yorkshire
```

## Troubleshooting

### If you get 0 results:

1. **Run the debug script** to see what the website is actually returning:

```bash
python debug_request.py
```

This will create several `debug_*.html` files showing:
- The actual form structure (`debug_form.html`)
- What the server returns for different field names (`debug_response_*.html`)

2. **Check the HTML files** to see:
   - Are there actually results in the HTML?
   - What are the correct form field names?
   - Does the website return an error message?

3. **Inspect the form manually**:

```bash
python inspect_cp40_form.py
```

This will save `cp40_form_inspection.html` with the form structure.

### Common Issues

1. **"No results found"** - Could mean:
   - The surname doesn't exist in the database for that time period
   - Form field names are incorrect (run `debug_request.py`)
   - The website structure changed

2. **Connection errors** - Check:
   - Internet connection
   - Website is accessible: https://www.medievalgenealogy.org.uk/aalt/cp40_search.php
   - Firewall settings

3. **Wrong Python/Environment** - Make sure you're running in WSL with the correct Python:
   ```bash
   which python  # Should show /path/to/venv/bin/python or similar
   python --version  # Should be 3.x
   ```

## Files

- `cp40_surname_scraper_simple.py` - Main scraper (recommended)
- `cp40_surname_scraper.py` - Alternative with more features
- `test_scraper.py` - Quick test with "Smith" surname
- `debug_request.py` - Debug what the website returns
- `inspect_cp40_form.py` - Inspect the form structure
- `example_surname_search.py` - Usage examples

## Command Line Options

```bash
python cp40_surname_scraper_simple.py --help
```

Available options:
- `surname` - Required: surname to search (supports wildcards `*` and `?`)
- `--forename` - Optional: forename filter
- `--place` - Optional: place filter
- `--year-from` - Optional: starting year (e.g., 1400)
- `--year-to` - Optional: ending year (e.g., 1500)
- `--county` - Optional: county filter (default: All)
- `--soundex` - Optional: use Soundex matching
- `--output` - Optional: output file (default: cp40_results.json)
- `--delay` - Optional: delay between requests in seconds (default: 1.0)

## Output Format

Results are saved as JSON with this structure:

```json
[
  {
    "raw_text": "Full text of the entry...",
    "roll_reference": "CP 40/559/55",
    "links": [
      {
        "url": "https://...",
        "text": "Index at AALT"
      }
    ],
    "county": "Yorkshire",
    "persons": ["John Smith", "William Black"],
    "places": ["London", "Westminster"],
    "year": "1450"
  }
]
```

## Next Steps After Debugging

If the debug script shows the website returns results but the scraper doesn't find them:

1. Open `debug_response_1.html` in a browser
2. Look at the HTML structure of the results
3. Update the `_parse_results()` method in the scraper to match the actual HTML structure
4. Test again

## Dependencies

All dependencies should already be in your environment:
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing

Both are already in your `requirements_pylaia.txt`.

