"""
Unit tests for label generator
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.label_generator import LabelGenerator


class TestLabelGenerator:
    """Test cases for LabelGenerator."""

    @pytest.fixture
    def label_gen(self):
        """Create a LabelGenerator instance."""
        return LabelGenerator(
            buffer_distance=200.0,
            flame_threshold=4.0,
            resolution=5.0
        )

    def test_initialization(self, label_gen):
        """Test label generator initialization."""
        assert label_gen.buffer_distance == 200.0
        assert label_gen.flame_threshold == 4.0
        assert label_gen.resolution == 5.0
        assert label_gen.buffer_cells == 40  # 200 / 5

    def test_compute_fire_distance(self, label_gen):
        """Test distance computation."""
        # Create simple fire mask
        fire_mask = np.zeros((100, 100))
        fire_mask[50, 50] = 1  # Single fire cell

        distances = label_gen.compute_fire_distance(fire_mask)

        assert distances.shape == (100, 100)
        assert distances[50, 50] == 0  # Distance to itself
        assert distances[50, 60] > 0  # Distance to nearby cell

    def test_generate_danger_labels_shape(self, label_gen):
        """Test danger label generation output shape."""
        T, H, W = 20, 100, 100
        flame_sequence = np.random.rand(T, H, W) * 5
        crew_positions = [(25, 25), (50, 50), (75, 75)]
        horizon = 5

        labels = label_gen.generate_danger_labels(
            flame_sequence,
            crew_positions,
            prediction_horizon=horizon
        )

        expected_samples = T - horizon
        expected_crews = len(crew_positions)

        assert labels.shape == (expected_samples, expected_crews)
        assert labels.dtype == np.int32

    def test_generate_danger_labels_values(self, label_gen):
        """Test danger label values are binary."""
        T, H, W = 15, 50, 50
        flame_sequence = np.random.rand(T, H, W) * 5
        crew_positions = [(25, 25)]

        labels = label_gen.generate_danger_labels(
            flame_sequence,
            crew_positions,
            prediction_horizon=5
        )

        # Check all labels are 0 or 1
        assert np.all((labels == 0) | (labels == 1))

    def test_multi_horizon_labels(self, label_gen):
        """Test multi-horizon label generation."""
        T, H, W = 30, 50, 50
        flame_sequence = np.random.rand(T, H, W) * 5
        crew_positions = [(25, 25)]
        horizons = [5, 10, 15]

        labels_dict = label_gen.generate_multi_horizon_labels(
            flame_sequence,
            crew_positions,
            horizons=horizons
        )

        assert len(labels_dict) == len(horizons)
        for horizon in horizons:
            assert horizon in labels_dict
            assert labels_dict[horizon].shape[0] == T - horizon
