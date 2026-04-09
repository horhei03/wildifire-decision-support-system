"""
Evaluate the trained ConvLSTM model on the held-out test set.

Produces metrics and figures for thesis:
- Overall metrics (AUC, accuracy, precision, recall, F1)
- Per-crew metrics
- Confusion matrix
- ROC curve
- Training history curves

Run via: python scripts/evaluate_test_set.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import logging
import csv

from src.models.convlstm_model import ConvLSTMModel
from src.training.train import create_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate():
    # --- Paths ---
    data_dir = 'data/processed'
    model_dir = 'outputs/convlstm_v3_6crews'
    checkpoint_path = f'{model_dir}/checkpoints/best_model.weights.h5'
    results_dir = Path('results')
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 4

    # --- Load test data ---
    logger.info("Loading test dataset...")
    test_ds, n_test, input_shape, num_crews, test_steps = create_dataset(
        data_dir, 'test', batch_size, shuffle=False
    )
    logger.info(f"Test set: {n_test} samples, input_shape={input_shape}, crews={num_crews}")

    # --- Rebuild model and load best weights ---
    logger.info(f"Loading model from {checkpoint_path}...")
    model = ConvLSTMModel(
        input_shape=input_shape,
        num_crews=num_crews,
        convlstm_filters=[32, 16, 8],
        kernel_size=(3, 3),
        dense_units=[128, 64],
        dropout_rate=0.3,
        learning_rate=1e-4
    )
    model.load(checkpoint_path)
    logger.info("Model loaded successfully")

    # --- Evaluate with Keras metrics ---
    logger.info("Evaluating on test set...")
    results = model.model.evaluate(test_ds, steps=test_steps, verbose=1)
    metric_names = model.model.metrics_names
    keras_metrics = dict(zip(metric_names, results))
    logger.info(f"Keras metrics: {keras_metrics}")

    # --- Get predictions for detailed analysis ---
    logger.info("Generating predictions...")
    y_true_list = []
    y_pred_list = []
    for i, (x_batch, y_batch) in enumerate(test_ds):
        if i >= test_steps:
            break
        preds = model.model.predict(x_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list, axis=0)[:n_test]
    y_prob = np.concatenate(y_pred_list, axis=0)[:n_test]
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Compute detailed metrics ---
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve
    )

    # Overall (flatten all crews together)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    y_prob_flat = y_prob.flatten()

    overall_metrics = {
        'accuracy': float(accuracy_score(y_true_flat, y_pred_flat)),
        'precision': float(precision_score(y_true_flat, y_pred_flat, zero_division=0)),
        'recall': float(recall_score(y_true_flat, y_pred_flat, zero_division=0)),
        'f1_score': float(f1_score(y_true_flat, y_pred_flat, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true_flat, y_prob_flat)),
        'num_samples': int(n_test),
        'num_crews': int(num_crews),
    }

    # Per-crew metrics (positions in 320x320 downsampled grid)
    crew_names = [
        'Alpha (60,100)', 'Bravo (100,200)', 'Charlie (160,100)',
        'Delta (160,220)', 'Echo (220,140)', 'Foxtrot (260,260)',
    ]
    per_crew = {}
    for c in range(num_crews):
        name = crew_names[c] if c < len(crew_names) else f'Crew {c+1}'
        per_crew[name] = {
            'accuracy': float(accuracy_score(y_true[:, c], y_pred[:, c])),
            'precision': float(precision_score(y_true[:, c], y_pred[:, c], zero_division=0)),
            'recall': float(recall_score(y_true[:, c], y_pred[:, c], zero_division=0)),
            'f1_score': float(f1_score(y_true[:, c], y_pred[:, c], zero_division=0)),
            'auc_roc': float(roc_auc_score(y_true[:, c], y_prob[:, c])),
            'danger_rate': float(y_true[:, c].mean()),
        }

    all_results = {
        'overall': overall_metrics,
        'per_crew': per_crew,
        'keras_metrics': {k: float(v) for k, v in keras_metrics.items()},
    }

    # Save metrics JSON
    metrics_path = results_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # --- Print results ---
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nSamples: {n_test} | Crews: {num_crews}")
    print(f"\n{'Metric':<15} {'Value':>10}")
    print("-" * 27)
    for k, v in overall_metrics.items():
        if k not in ('num_samples', 'num_crews'):
            print(f"{k:<15} {v:>10.4f}")

    print(f"\n{'Per-Crew Results':}")
    print("-" * 65)
    print(f"{'Crew':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 65)
    for name, m in per_crew.items():
        print(f"{name:<25} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1_score']:>7.4f} {m['auc_roc']:>7.4f}")

    # --- Generate figures ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. Confusion matrix (overall)
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (All Crews)', fontsize=14)
    plt.colorbar(im, ax=ax)
    labels = ['Safe (0)', 'Danger (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=16)
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    logger.info("Saved confusion_matrix.png")

    # 2. ROC curves (per crew + overall)
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr_all, tpr_all, _ = roc_curve(y_true_flat, y_prob_flat)
    ax.plot(fpr_all, tpr_all, 'k-', linewidth=2,
            label=f'Overall (AUC={overall_metrics["auc_roc"]:.4f})')
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for c in range(num_crews):
        name = crew_names[c] if c < len(crew_names) else f'Crew {c+1}'
        fpr, tpr, _ = roc_curve(y_true[:, c], y_prob[:, c])
        ax.plot(fpr, tpr, color=colors[c], linewidth=1.5,
                label=f'{name} (AUC={per_crew[name]["auc_roc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(figures_dir / 'roc_curves.png', dpi=150)
    plt.close()
    logger.info("Saved roc_curves.png")

    # 3. Per-crew confusion matrices
    n_rows = (num_crews + 2) // 3  # ceil division for 3 columns
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows))
    axes = axes.flatten() if num_crews > 3 else [axes] if num_crews == 1 else axes
    for c in range(num_crews):
        name = crew_names[c] if c < len(crew_names) else f'Crew {c+1}'
        cm_c = confusion_matrix(y_true[:, c], y_pred[:, c])
        ax = axes[c]
        im = ax.imshow(cm_c, interpolation='nearest', cmap='Blues')
        ax.set_title(name, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Safe', 'Danger'])
        ax.set_yticklabels(['Safe', 'Danger'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        for i in range(2):
            for j in range(2):
                color = 'white' if cm_c[i, j] > cm_c.max() / 2 else 'black'
                ax.text(j, i, str(cm_c[i, j]), ha='center', va='center', color=color, fontsize=14)
    plt.suptitle('Confusion Matrices per Crew Position', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrices_per_crew.png', dpi=150)
    plt.close()
    logger.info("Saved confusion_matrices_per_crew.png")

    # 4. Training history curves
    training_log = Path(model_dir) / 'logs' / 'training_log.csv'
    if training_log.exists():
        epochs_data = []
        with open(training_log) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs_data.append({k: float(v) for k, v in row.items()})

        if epochs_data:
            ep = [d['epoch'] + 1 for d in epochs_data]

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            # Loss
            axes[0, 0].plot(ep, [d['loss'] for d in epochs_data], 'b-', label='Train')
            axes[0, 0].plot(ep, [d['val_loss'] for d in epochs_data], 'r-', label='Validation')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # AUC
            axes[0, 1].plot(ep, [d['auc'] for d in epochs_data], 'b-', label='Train')
            axes[0, 1].plot(ep, [d['val_auc'] for d in epochs_data], 'r-', label='Validation')
            axes[0, 1].set_title('AUC')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Precision
            axes[1, 0].plot(ep, [d['precision'] for d in epochs_data], 'b-', label='Train')
            axes[1, 0].plot(ep, [d['val_precision'] for d in epochs_data], 'r-', label='Validation')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Recall
            axes[1, 1].plot(ep, [d['recall'] for d in epochs_data], 'b-', label='Train')
            axes[1, 1].plot(ep, [d['val_recall'] for d in epochs_data], 'r-', label='Validation')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle('Training History', fontsize=14)
            plt.tight_layout()
            plt.savefig(figures_dir / 'training_history.png', dpi=150)
            plt.close()
            logger.info("Saved training_history.png")

    # 5. Metrics bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / num_crews
    for c in range(num_crews):
        name = crew_names[c] if c < len(crew_names) else f'Crew {c+1}'
        vals = [per_crew[name][m] for m in metrics_to_plot]
        ax.bar(x + c * width, vals, width, label=name, color=colors[c % len(colors)], alpha=0.85)
    ax.set_xticks(x + width * (num_crews - 1) / 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'])
    ax.set_ylim([0, 1.05])
    ax.set_title('Test Set Metrics per Crew Position', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(figures_dir / 'metrics_per_crew.png', dpi=150)
    plt.close()
    logger.info("Saved metrics_per_crew.png")

    print(f"\nAll figures saved to: {figures_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    evaluate()
