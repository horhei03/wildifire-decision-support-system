"""
Training callbacks for model checkpointing and monitoring
"""

from tensorflow import keras
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    monitor: str = 'val_loss'
):
    """
    Create list of training callbacks.

    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        monitor: Metric to monitor

    Returns:
        List of Keras callbacks
    """
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Model checkpointing
    checkpoint_path = str(Path(checkpoint_dir) / 'best_model.weights.h5')
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_cb)

    # Early stopping
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=early_stopping_patience,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    callbacks.append(early_stop_cb)

    # Learning rate reduction
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    callbacks.append(reduce_lr_cb)

    # TensorBoard logging
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_cb)

    # CSV logger
    csv_path = str(Path(log_dir) / 'training_log.csv')
    csv_cb = keras.callbacks.CSVLogger(
        filename=csv_path,
        separator=',',
        append=False
    )
    callbacks.append(csv_cb)

    logger.info(f"Configured {len(callbacks)} callbacks")

    return callbacks
