"""Legacy entrypoint preserved for backwards compatibility."""

import argparse

from workflow_manager import ImageGrouper, WorkflowManager
from workflow_manager.settings import IMAGE_DIR, OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CP40 Plea Roll Transcription Workflow Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Sequence:
  1. Kraken (line segmentation)
  2. PyLaia (HTR recognition)
  3. Post-correction and named entity extraction (Gemini 3 Flash Preview + Bayesian)
  4. Stitching (merge transcriptions from multiple images)
  5. Expansion (expand abbreviations)
  6. Translation (Latin to English)
  7. Indexing (structured entity extraction)

Examples:
  # Process all images in input_images/
  python workflow_manager.py

  # Force reprocess everything
  python workflow_manager.py --force

  # Rerun from post-correction step (keeps Kraken/PyLaia results)
  python workflow_manager.py --rerun-from-post-pylaia

  # Process specific directory
  python workflow_manager.py --dir /path/to/images
        """
    )
    parser.add_argument("--dir", default=IMAGE_DIR, 
                        help="Directory containing input images")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocess all steps even if results exist")
    parser.add_argument("--no-images", action="store_true",
                        help="Run in text-only mode (no image uploads to vision API)")
    parser.add_argument("--rerun-from-post-pylaia", action="store_true",
                        help="Rerun everything from post-correction step onwards, "
                             "keeping existing Kraken and PyLaia results")
    args = parser.parse_args()

    manager = WorkflowManager(
        force=args.force, 
        use_images=not args.no_images,
        rerun_from_post_pylaia=args.rerun_from_post_pylaia
    )
    groups = ImageGrouper(args.dir, output_directory=OUTPUT_DIR).scan()
    manager.execute(groups)


if __name__ == "__main__":
    main()

