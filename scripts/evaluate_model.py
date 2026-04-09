#!/usr/bin/env python
"""
Evaluate trained wildfire prediction model
"""

import argparse
import logging
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import evaluate_model
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained wildfire prediction model'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model weights (.h5)'
    )

    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to model configuration file'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with preprocessed test data'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/metrics',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--prediction-horizon',
        type=int,
        default=10,
        help='Prediction horizon in minutes'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'val'],
        help='Data split to evaluate'
    )

    args = parser.parse_args()

    logger.info("Starting model evaluation...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Split: {args.split}")

    # Load model config
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    # Evaluate
    metrics = evaluate_model(
        model_path=args.model_path,
        model_config=model_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prediction_horizon=args.prediction_horizon,
        split=args.split
    )

    logger.info("Evaluation complete!")

    # Print summary
    logger.info("Performance Summary:")
    logger.info(f"  Accuracy: {metrics['overall']['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['overall']['precision']:.4f}")
    logger.info(f"  Recall: {metrics['overall']['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['overall']['f1_score']:.4f}")

    if metrics['overall'].get('auc_roc'):
        logger.info(f"  AUC-ROC: {metrics['overall']['auc_roc']:.4f}")

    logger.info("\nSafety Metrics:")
    logger.info(f"  False Negative Rate: {metrics['safety']['false_negative_rate']:.4f}")
    logger.info(f"  Recall (Safety Critical): {metrics['safety']['recall_safety_critical']:.4f}")


if __name__ == '__main__':
    main()
