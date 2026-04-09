"""
Hyperparameter tuning for ConvLSTM wildfire model using Keras Tuner.

Uses Bayesian Optimization to efficiently search the hyperparameter space.
Run via: python scripts/tune_hyperparameters.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from pathlib import Path
import logging
import json

from src.training.train import create_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable model builder
# ---------------------------------------------------------------------------

def build_model(hp, input_shape, num_crews):
    """Build a ConvLSTM model with tunable hyperparameters."""

    # --- Hyperparameters to search ---
    learning_rate = hp.Float(
        'learning_rate', min_value=1e-5, max_value=1e-3, sampling='log'
    )
    dropout_rate = hp.Float(
        'dropout_rate', min_value=0.1, max_value=0.5, step=0.1
    )
    num_convlstm_layers = hp.Int(
        'num_convlstm_layers', min_value=2, max_value=4
    )
    initial_filters = hp.Choice(
        'initial_filters', values=[16, 32, 64]
    )
    dense_1_units = hp.Choice(
        'dense_1_units', values=[64, 128, 256]
    )
    dense_2_units = hp.Choice(
        'dense_2_units', values=[32, 64, 128]
    )

    # --- Build architecture ---
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Spatial downsampling (same as baseline)
    x = layers.TimeDistributed(
        layers.AveragePooling2D(pool_size=(5, 5)),
        name='spatial_downsample'
    )(x)

    # ConvLSTM layers with halving filters
    for i in range(num_convlstm_layers):
        filters = max(initial_filters // (2 ** i), 8)
        return_sequences = i < num_convlstm_layers - 1

        x = layers.ConvLSTM2D(
            filters=filters,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=return_sequences,
            activation='relu',
            name=f'convlstm_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_convlstm_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'dropout_convlstm_{i+1}')(x)

    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Dense layers
    x = layers.Dense(dense_1_units, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_dense_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense_1')(x)

    x = layers.Dense(dense_2_units, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_dense_2')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense_2')(x)

    outputs = layers.Dense(num_crews, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model


# ---------------------------------------------------------------------------
# Main tuning routine
# ---------------------------------------------------------------------------

def main():
    # --- Settings ---
    data_dir = 'data/processed'
    tuner_dir = 'outputs/tuning'
    project_name = 'convlstm_bayesian'
    max_trials = 20          # total HP combinations to try
    epochs_per_trial = 15    # each trial trains for this many epochs (early stopping may cut short)
    batch_size = 4           # keep small for memory

    Path(tuner_dir).mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logger.info("Loading datasets...")
    train_ds, n_train, input_shape, num_crews, steps_per_epoch = create_dataset(
        data_dir, 'train', batch_size, shuffle=True
    )
    val_ds, n_val, _, _, validation_steps = create_dataset(
        data_dir, 'val', batch_size, shuffle=False
    )

    logger.info(f"Train: {n_train} samples ({steps_per_epoch} steps/epoch) | Val: {n_val} samples ({validation_steps} steps)")
    logger.info(f"Input shape: {input_shape} | Crews: {num_crews}")

    # --- Tuner ---
    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_model(hp, input_shape, num_crews),
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=max_trials,
        seed=42,
        directory=tuner_dir,
        project_name=project_name,
        overwrite=False  # set True to restart search from scratch
    )

    tuner.search_space_summary()

    # Callbacks for each trial
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]

    # --- Run search ---
    logger.info(f"Starting Bayesian search: {max_trials} trials, {epochs_per_trial} epochs each")
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_per_trial,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # --- Results ---
    tuner.results_summary(num_trials=5)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_values = best_hps.values
    logger.info(f"Best hyperparameters: {best_values}")

    # Save best hyperparameters to JSON for easy reference
    results_path = Path(tuner_dir) / 'best_hyperparameters.json'
    with open(results_path, 'w') as f:
        json.dump(best_values, f, indent=2)
    logger.info(f"Best hyperparameters saved to {results_path}")

    # Print how to retrain with the best config
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"\nBest hyperparameters saved to: {results_path}")
    print(f"\nBest values found:")
    for k, v in best_values.items():
        print(f"  {k}: {v}")
    print("\nNext step: update configs/model_config.yaml and")
    print("configs/training_config.yaml with these values,")
    print("then retrain with: python scripts/verify_and_train.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
