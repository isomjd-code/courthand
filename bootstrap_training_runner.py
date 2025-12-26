#!/usr/bin/env python3
"""
Main entry point for bootstrap training workflow.

Usage:
    python bootstrap_training_runner.py [--force]
"""

import argparse
import sys

from bootstrap_training import BootstrapTrainingManager
from workflow_manager.settings import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bootstrap Pylaia training with Gemini 3 Flash Preview as teacher"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all images even if already processed"
    )
    
    args = parser.parse_args()
    
    try:
        manager = BootstrapTrainingManager(force=args.force)
        manager.run()
        logger.info("Bootstrap training completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Bootstrap training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Bootstrap training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

