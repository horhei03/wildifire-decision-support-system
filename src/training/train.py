"""
Training orchestration
"""

import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging

from ..models.convlstm_model import ConvLSTMModel
from .callbacks import get_callbacks

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _read_npy_header(path):
    """Read .npy file header to get shape, dtype, and data offset without loading data."""
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        if version[0] == 1:
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    return shape, dtype, offset


def create_dataset(data_dir: str, split: str, batch_size: int, shuffle: bool = False):
    """
    Create a tf.data.Dataset that loads data in batches from .npy files.
    Uses file-seeking instead of memory-mapping to avoid Windows OOM errors.

    Args:
        data_dir: Directory containing .npy files
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        Tuple of (tf.data.Dataset, num_samples, input_shape, num_crews, steps_per_epoch)
    """
    data_path = Path(data_dir)

    X_path = str(data_path / f"X_{split}.npy")
    y_path = str(data_path / f"y_{split}.npy")

    # Read headers to get shapes/dtypes without loading data into memory
    X_shape, X_dtype, X_offset = _read_npy_header(X_path)
    y_shape, y_dtype, y_offset = _read_npy_header(y_path)

    num_samples = X_shape[0]
    input_shape = tuple(X_shape[1:])
    num_crews = y_shape[1]

    X_sample_bytes = int(np.prod(input_shape)) * X_dtype.itemsize
    y_sample_bytes = num_crews * y_dtype.itemsize

    # Validate file sizes match expected data
    import os
    X_file_size = os.path.getsize(X_path)
    X_expected = X_offset + num_samples * X_sample_bytes
    if X_file_size < X_expected:
        raise ValueError(
            f"{X_path} is truncated: {X_file_size:,} bytes on disk, "
            f"expected {X_expected:,} bytes for {num_samples} samples. "
            f"File may not have fully synced (OneDrive). "
            f"Try regenerating the dataset."
        )

    logger.info(f"Creating {split} dataset: {num_samples} samples, shape={input_shape}")

    # Generator reads individual samples by seeking into the .npy files
    def data_generator():
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)
        with open(X_path, 'rb') as fx, open(y_path, 'rb') as fy:
            for i in indices:
                fx.seek(X_offset + int(i) * X_sample_bytes)
                x = np.frombuffer(fx.read(X_sample_bytes), dtype=X_dtype).reshape(input_shape).astype(np.float32)

                fy.seek(y_offset + int(i) * y_sample_bytes)
                y_val = np.frombuffer(fy.read(y_sample_bytes), dtype=y_dtype).reshape((num_crews,)).astype(np.float32)

                yield x, y_val

    # Build tf.data.Dataset from generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(num_crews,), dtype=tf.float32)
        )
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat indefinitely so Keras can run multiple epochs
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = int(np.ceil(num_samples / batch_size))

    return dataset, num_samples, input_shape, num_crews, steps_per_epoch


def train_model(
    model_config_path: str,
    training_config_path: str,
    data_dir: str,
    output_dir: str,
    experiment_name: Optional[str] = None
):
    """
    Complete training pipeline.

    Args:
        model_config_path: Path to model_config.yaml
        training_config_path: Path to training_config.yaml
        data_dir: Directory with preprocessed data
        output_dir: Directory to save outputs
        experiment_name: Optional experiment identifier
    """
    # Load configs
    model_cfg = load_config(model_config_path)
    train_cfg = load_config(training_config_path)

    # Create output directory
    output_path = Path(output_dir)
    if experiment_name:
        output_path = output_path / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = train_cfg.get('batch_size', 4)

    # Create data pipelines (loads batches on demand, not all at once)
    train_ds, n_train, input_shape, num_crews, train_steps = create_dataset(
        data_dir, 'train', batch_size, shuffle=True
    )
    val_ds, n_val, _, _, val_steps = create_dataset(
        data_dir, 'val', batch_size, shuffle=False
    )

    logger.info(f"Input shape: {input_shape}, Number of crews: {num_crews}")
    logger.info(f"Train: {n_train} samples, Val: {n_val} samples")

    # Initialize model
    model = ConvLSTMModel(
        input_shape=input_shape,
        num_crews=num_crews,
        convlstm_filters=model_cfg.get('convlstm_filters', [32, 16, 8]),
        kernel_size=tuple(model_cfg.get('kernel_size', [3, 3])),
        dense_units=model_cfg.get('dense_units', [128, 64]),
        dropout_rate=model_cfg.get('dropout_rate', 0.3),
        learning_rate=train_cfg.get('learning_rate', 1e-4)
    )

    model.summary()

    # Setup callbacks
    callbacks = get_callbacks(
        checkpoint_dir=str(output_path / 'checkpoints'),
        log_dir=str(output_path / 'logs'),
        early_stopping_patience=train_cfg.get('early_stopping_patience', 10),
        reduce_lr_patience=train_cfg.get('reduce_lr_patience', 5)
    )

    # Train using tf.data.Dataset (memory-efficient)
    epochs = train_cfg.get('epochs', 50)
    logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}")

    history = model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = output_path / 'final_model.weights.h5'
    model.save(str(final_model_path))

    logger.info(f"Training complete. Model saved to {final_model_path}")

    return model, history
