# Webviewer

A web-based viewer for CP40 plea roll cases with transcription editing capabilities.

## Features

- Dropdown menu to select from available cases
- Automatic image loading from `input_images` directory
- Interactive line editing with polygon overlays
- Legal index and full text editing
- Zoom and pan functionality

## Usage

1. Start the server:
   ```bash
   python webviewer/server.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000/webviewer.html
   ```

3. Select a case from the dropdown menu. The dropdown will show:
   - Case group ID
   - County (if available)
   - Roll number (if available)

4. The viewer will automatically:
   - Load the `master_record.json` from the selected case directory
   - Find and load corresponding images from the `input_images` directory
   - Display all available pages/images for the case

## Configuration

The server paths are configured in `server.py`:
- `OUTPUT_DIR`: Directory containing case subdirectories with `master_record.json`
- `INPUT_IMAGES_DIR`: Directory containing source images
- `PORT`: Server port (default: 8000)

## How It Works

1. The server scans `cp40_processing/output` for subdirectories containing `master_record.json`
2. When a case is selected, the server loads the JSON file
3. Images are automatically found in `input_images` based on filenames in the `source_material` section
4. The viewer displays images with interactive line overlays and allows editing

## Image Matching

The server uses flexible image matching to handle:
- Case variations (e.g., "file.jpg" vs "FILE.JPG")
- Spacing variations (e.g., "CP 40-559 055-a.jpg" vs "CP-40-559-055-a.jpg")

