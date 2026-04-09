"""
ConvLSTM model for spatiotemporal wildfire prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


class ConvLSTMModel:
    """
    Convolutional LSTM architecture for wildfire danger prediction.
    """

    def __init__(
        self,
        input_shape: tuple,
        num_crews: int,
        convlstm_filters: list = [64, 32, 16],
        kernel_size: tuple = (3, 3),
        dense_units: list = [128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-4
    ):
        """
        Initialize ConvLSTM model.

        Args:
            input_shape: (timesteps, height, width, channels)
            num_crews: Number of crew positions to predict for
            convlstm_filters: List of filter counts for ConvLSTM layers
            kernel_size: Convolutional kernel size
            dense_units: List of units for dense layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
        """
        self.input_shape = input_shape
        self.num_crews = num_crews
        self.convlstm_filters = convlstm_filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = self._build_model()

        logger.info(
            f"Initialized ConvLSTM model: input={input_shape}, "
            f"crews={num_crews}, filters={convlstm_filters}"
        )

    def _build_model(self) -> keras.Model:
        """
        Build the ConvLSTM architecture.

        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape)

        x = inputs

        # Spatial downsampling to reduce memory (320×320 → 64×64)
        x = layers.TimeDistributed(
            layers.AveragePooling2D(pool_size=(5, 5)),
            name='spatial_downsample'
        )(x)

        # ConvLSTM layers with decreasing filters
        for i, filters in enumerate(self.convlstm_filters):
            return_sequences = i < len(self.convlstm_filters) - 1

            x = layers.ConvLSTM2D(
                filters=filters,
                kernel_size=self.kernel_size,
                padding='same',
                return_sequences=return_sequences,
                activation='relu',
                name=f'convlstm_{i+1}'
            )(x)

            x = layers.BatchNormalization(name=f'bn_convlstm_{i+1}')(x)

            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_convlstm_{i+1}')(x)

        # Global pooling to reduce spatial dimensions
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)

        # Dense layers for crew-specific predictions
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)

            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_dense_{i+1}')(x)

        # Output layer: binary classification for each crew
        outputs = layers.Dense(
            self.num_crews,
            activation='sigmoid',
            name='output'
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='ConvLSTM_WildfireDSS')

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        return model

    def summary(self):
        """Print model architecture summary."""
        self.model.summary()

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int = 50,
        batch_size: int = 16,
        callbacks: list = None
    ):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks

        Returns:
            Training history
        """
        logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            verbose=1
        )

        logger.info("Training completed")

        return history

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        return self.model.predict(X)

    def save(self, path: str):
        """
        Save model weights.

        Args:
            path: Path to save .h5 file
        """
        self.model.save_weights(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load model weights.

        Args:
            path: Path to .h5 file
        """
        self.model.load_weights(path)
        logger.info(f"Model loaded from {path}")
