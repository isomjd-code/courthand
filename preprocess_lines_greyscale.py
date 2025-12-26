"""Legacy CLI wrapper for the greyscale line preprocessor."""

import sys

from line_preprocessor_greyscale import main


def run() -> None:
    if len(sys.argv) != 5:
        print("Usage: python3 preprocess_lines_greyscale.py <image_path> <json_path> <output_dir> <output_list_path>")
        print()
        print("Arguments:")
        print("  image_path       Path to the source page image file (e.g., page.jpg)")
        print("  json_path        Path to the Kraken JSON segmentation file")
        print("  output_dir       Directory where processed line images will be saved")
        print("  output_list_path Path to write the list of processed image paths")
        print()
        print("This script produces normalized greyscale line images instead of binarized images.")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    run()
