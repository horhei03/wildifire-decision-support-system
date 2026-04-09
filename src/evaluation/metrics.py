"""
Evaluation metrics for wildfire danger prediction
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'macro'
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        y_prob: Prediction probabilities (for AUC)
        average: Averaging strategy for multi-class

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    # AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob, average=average))
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics['auc_roc'] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Per-class metrics if multi-output
    if y_true.ndim > 1:
        num_classes = y_true.shape[1]
        metrics['per_class'] = {}

        for i in range(num_classes):
            class_metrics = calculate_metrics(
                y_true[:, i],
                y_pred[:, i],
                y_prob[:, i] if y_prob is not None else None,
                average='binary'
            )
            metrics['per_class'][f'crew_{i}'] = class_metrics

    logger.info(
        f"Metrics - Acc: {metrics['accuracy']:.4f}, "
        f"Prec: {metrics['precision']:.4f}, "
        f"Rec: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1_score']:.4f}"
    )

    return metrics


def calculate_lead_time_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prediction_horizon: int
) -> Dict:
    """
    Calculate metrics specific to early warning performance.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        prediction_horizon: Prediction horizon in minutes

    Returns:
        Dictionary with lead time metrics
    """
    metrics = {}

    # True Positives (correctly predicted danger)
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # False Negatives (missed dangers)
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # False Positives (false alarms)
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # True Negatives (correctly predicted safe)
    tn = np.sum((y_true == 0) & (y_pred == 0))

    metrics['true_positives'] = int(tp)
    metrics['false_negatives'] = int(fn)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)

    # Warning system specific metrics
    total_dangers = tp + fn
    if total_dangers > 0:
        metrics['warning_rate'] = float(tp / total_dangers)
        metrics['miss_rate'] = float(fn / total_dangers)
    else:
        metrics['warning_rate'] = None
        metrics['miss_rate'] = None

    total_predictions = tp + fp
    if total_predictions > 0:
        metrics['false_alarm_rate'] = float(fp / total_predictions)
    else:
        metrics['false_alarm_rate'] = None

    metrics['prediction_horizon_minutes'] = prediction_horizon

    logger.info(
        f"Lead time metrics - TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}"
    )

    return metrics


def calculate_safety_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Calculate safety-critical metrics with emphasis on avoiding false negatives.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with safety metrics
    """
    metrics = {}

    # Recall is critical - we must not miss actual dangers
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_safety_critical'] = float(recall)

    # False negative rate (most dangerous errors)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    total_dangers = np.sum(y_true == 1)

    if total_dangers > 0:
        metrics['false_negative_rate'] = float(fn / total_dangers)
    else:
        metrics['false_negative_rate'] = None

    # Specificity (true negative rate)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    total_safe = np.sum(y_true == 0)

    if total_safe > 0:
        metrics['specificity'] = float(tn / total_safe)
    else:
        metrics['specificity'] = None

    # Cost-weighted score (penalize FN more than FP)
    # Assume FN cost = 10x FP cost
    fp = np.sum((y_true == 0) & (y_pred == 1))
    cost = 10 * fn + fp
    metrics['weighted_error_cost'] = int(cost)

    logger.info(f"Safety metrics - Recall: {recall:.4f}, FN rate: {metrics['false_negative_rate']}")

    return metrics
