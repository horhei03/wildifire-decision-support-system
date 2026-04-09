"""
Model evaluation orchestration
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import logging

from ..models.convlstm_model import ConvLSTMModel
from .metrics import calculate_metrics, calculate_lead_time_metrics, calculate_safety_metrics

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    model_config: Dict,
    data_dir: str,
    output_dir: str,
    prediction_horizon: int = 10,
    split: str = 'test'
) -> Dict:
    """
    Complete model evaluation pipeline.

    Args:
        model_path: Path to trained model weights
        model_config: Model architecture configuration
        data_dir: Directory with preprocessed data
        output_dir: Directory to save evaluation results
        prediction_horizon: Prediction horizon in minutes
        split: Data split to evaluate ('test' or 'val')

    Returns:
        Dictionary of evaluation metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test = np.load(Path(data_dir) / f"X_{split}.npy")
    y_test = np.load(Path(data_dir) / f"y_{split}.npy")

    logger.info(f"Loaded {split} data: X={X_test.shape}, y={y_test.shape}")

    # Get model parameters
    input_shape = X_test.shape[1:]
    num_crews = y_test.shape[1]

    # Initialize and load model
    model = ConvLSTMModel(
        input_shape=input_shape,
        num_crews=num_crews,
        convlstm_filters=model_config.get('convlstm_filters', [64, 32, 16]),
        kernel_size=tuple(model_config.get('kernel_size', [3, 3])),
        dense_units=model_config.get('dense_units', [128, 64]),
        dropout_rate=model_config.get('dropout_rate', 0.3)
    )

    model.load(model_path)

    # Make predictions
    logger.info("Generating predictions...")
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate metrics
    all_metrics = {}

    # Overall metrics
    logger.info("Calculating overall metrics...")
    overall_metrics = calculate_metrics(
        y_test.flatten(),
        y_pred.flatten(),
        y_prob.flatten()
    )
    all_metrics['overall'] = overall_metrics

    # Lead time metrics
    logger.info("Calculating lead time metrics...")
    lead_time_metrics = calculate_lead_time_metrics(
        y_test.flatten(),
        y_pred.flatten(),
        prediction_horizon
    )
    all_metrics['lead_time'] = lead_time_metrics

    # Safety metrics
    logger.info("Calculating safety metrics...")
    safety_metrics = calculate_safety_metrics(
        y_test.flatten(),
        y_pred.flatten()
    )
    all_metrics['safety'] = safety_metrics

    # Per-crew metrics
    logger.info("Calculating per-crew metrics...")
    per_crew_metrics = {}
    for i in range(num_crews):
        crew_metrics = calculate_metrics(
            y_test[:, i],
            y_pred[:, i],
            y_prob[:, i],
            average='binary'
        )
        per_crew_metrics[f'crew_{i}'] = crew_metrics

    all_metrics['per_crew'] = per_crew_metrics

    # Save metrics
    metrics_file = output_path / f"evaluation_metrics_{split}.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Evaluation complete. Metrics saved to {metrics_file}")

    # Save predictions
    predictions_file = output_path / f"predictions_{split}.npz"
    np.savez(
        predictions_file,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob
    )

    logger.info(f"Predictions saved to {predictions_file}")

    return all_metrics


def compare_models(
    model_paths: Dict[str, str],
    model_config: Dict,
    data_dir: str,
    output_dir: str
) -> Dict:
    """
    Compare multiple models on the same test set.

    Args:
        model_paths: Dictionary mapping model names to paths
        model_config: Model architecture configuration
        data_dir: Directory with preprocessed data
        output_dir: Directory to save comparison results

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}

    for model_name, model_path in model_paths.items():
        logger.info(f"Evaluating model: {model_name}")

        metrics = evaluate_model(
            model_path,
            model_config,
            data_dir,
            output_dir,
            split='test'
        )

        comparison[model_name] = {
            'accuracy': metrics['overall']['accuracy'],
            'precision': metrics['overall']['precision'],
            'recall': metrics['overall']['recall'],
            'f1_score': metrics['overall']['f1_score'],
            'auc_roc': metrics['overall'].get('auc_roc'),
            'false_negative_rate': metrics['safety']['false_negative_rate']
        }

    # Save comparison
    comparison_file = Path(output_dir) / "model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Model comparison saved to {comparison_file}")

    return comparison
