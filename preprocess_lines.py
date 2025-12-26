"""Legacy CLI wrapper for the line preprocessor."""

import sys

from line_preprocessor import main


def run() -> None:
    if len(sys.argv) != 5:
        print("Usage: python3 preprocess_lines.py <image_path> <json_path> <output_dir> <pylaia_list_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    run()

