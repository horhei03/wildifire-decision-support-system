#!/usr/bin/env python
"""
Train ConvLSTM model for wildfire danger prediction
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import train_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train ConvLSTM wildfire danger prediction model'
    )

    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to model configuration file'
    )

    parser.add_argument(
        '--training-config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with preprocessed training data'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/trained',
        help='Directory to save trained model'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Optional experiment name for output organization'
    )

    args = parser.parse_args()

    logger.info("Starting model training...")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Training config: {args.training_config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Train model
    model, history = train_model(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )

    logger.info("Training complete!")

    # Print final metrics
    final_metrics = {
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'train_accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
    }

    logger.info("Final metrics:")
    for metric, value in final_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()
