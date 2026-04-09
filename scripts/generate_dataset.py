#!/usr/bin/env python
"""
Generate training dataset from FARSITE outputs
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.dataset_pipeline import DatasetPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate training dataset from FARSITE outputs'
    )

    parser.add_argument(
        '--data-config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to data configuration file'
    )

    parser.add_argument(
        '--crew-config',
        type=str,
        default='configs/crew_positions.yaml',
        help='Path to crew positions configuration'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--prediction-horizon',
        type=int,
        default=10,
        help='Prediction horizon in minutes'
    )

    parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Training data fraction'
    )

    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation data fraction'
    )

    args = parser.parse_args()

    logger.info("Starting dataset generation...")
    logger.info(f"Data config: {args.data_config}")
    logger.info(f"Crew config: {args.crew_config}")
    logger.info(f"Output dir: {args.output_dir}")

    # Initialize pipeline
    pipeline = DatasetPipeline(args.data_config)

    # Load crew positions
    crew_positions = pipeline.load_crew_positions(args.crew_config)
    logger.info(f"Loaded {len(crew_positions)} crew positions")

    # Generate dataset
    saved_paths = pipeline.generate_full_dataset(
        crew_positions=crew_positions,
        output_dir=args.output_dir,
        prediction_horizon=args.prediction_horizon,
        train_split=args.train_split,
        val_split=args.val_split
    )

    logger.info("Dataset generation complete!")
    logger.info("Saved files:")
    for split, paths in saved_paths.items():
        logger.info(f"  {split}: X={paths['X']}, y={paths['y']}")


if __name__ == '__main__':
    main()
