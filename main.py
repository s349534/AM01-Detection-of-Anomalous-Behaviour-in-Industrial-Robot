#!/usr/bin/env python3
"""Main script to run the anomaly detection pipeline.

Usage:
    python main.py [--config CONFIG_PATH]
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Main entry point."""
    print("AM01 Anomaly Detection Pipeline")
    print("=" * 40)
    print("This is a placeholder for the main execution script.")
    print("Run notebooks/01_data_exploration.ipynb to start.")


if __name__ == "__main__":
    main()