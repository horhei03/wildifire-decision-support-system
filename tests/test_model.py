"""
Unit tests for ConvLSTM model
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.convlstm_model import ConvLSTMModel


class TestConvLSTMModel:
    """Test cases for ConvLSTMModel."""

    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        return {
            'input_shape': (10, 100, 100, 2),
            'num_crews': 3,
            'convlstm_filters': [16, 8],
            'kernel_size': (3, 3),
            'dense_units': [32, 16],
            'dropout_rate': 0.3,
            'learning_rate': 1e-4
        }

    def test_model_initialization(self, model_config):
        """Test model initialization."""
        model = ConvLSTMModel(**model_config)

        assert model.input_shape == model_config['input_shape']
        assert model.num_crews == model_config['num_crews']
        assert model.model is not None

    def test_model_output_shape(self, model_config):
        """Test model output shape."""
        model = ConvLSTMModel(**model_config)

        # Create dummy input
        batch_size = 4
        X = np.random.rand(batch_size, *model_config['input_shape']).astype(np.float32)

        predictions = model.predict(X)

        assert predictions.shape == (batch_size, model_config['num_crews'])

    def test_model_output_range(self, model_config):
        """Test model outputs are in [0, 1] range (sigmoid)."""
        model = ConvLSTMModel(**model_config)

        X = np.random.rand(2, *model_config['input_shape']).astype(np.float32)
        predictions = model.predict(X)

        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_model_summary(self, model_config, capsys):
        """Test model summary generation."""
        model = ConvLSTMModel(**model_config)
        model.summary()

        captured = capsys.readouterr()
        assert "Model:" in captured.out or "ConvLSTM" in captured.out

    def test_model_save_load(self, model_config, tmp_path):
        """Test model save and load."""
        model = ConvLSTMModel(**model_config)

        # Save model
        save_path = tmp_path / "test_model.h5"
        model.save(str(save_path))

        assert save_path.exists()

        # Create new model and load weights
        model2 = ConvLSTMModel(**model_config)
        model2.load(str(save_path))

        # Test both models produce same output
        X = np.random.rand(1, *model_config['input_shape']).astype(np.float32)
        pred1 = model.predict(X)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)
