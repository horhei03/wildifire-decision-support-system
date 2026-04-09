"""
Generate danger labels for crew positions based on fire proximity
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.ndimage import distance_transform_edt
import logging

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generate binary danger labels for crew positions based on:
    - Distance to fire front
    - Flame length intensity
    - Prediction horizon
    """

    def __init__(
        self,
        buffer_distance: float = 200.0,
        flame_threshold: float = 4.0,
        resolution: float = 5.0
    ):
        """
        Initialize label generator.

        Args:
            buffer_distance: Safety buffer distance in meters
            flame_threshold: Flame length threshold for extreme danger (meters)
            resolution: Spatial resolution in meters per cell
        """
        self.buffer_distance = buffer_distance
        self.flame_threshold = flame_threshold
        self.resolution = resolution
        self.buffer_cells = int(buffer_distance / resolution)

        logger.info(
            f"Initialized LabelGenerator: buffer={buffer_distance}m, "
            f"flame_threshold={flame_threshold}m, resolution={resolution}m"
        )

    def compute_fire_distance(self, fire_mask: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance to nearest fire cell.

        Args:
            fire_mask: Binary mask of fire presence (1=fire, 0=no fire)

        Returns:
            Distance transform array in cells
        """
        # Invert mask for distance transform (EDT computes distance to nearest 0)
        inverted = 1 - fire_mask
        distance_cells = distance_transform_edt(inverted)

        # Convert to meters
        distance_meters = distance_cells * self.resolution

        return distance_meters

    def generate_danger_labels(
        self,
        flame_length_sequence: np.ndarray,
        crew_positions: List[Tuple[int, int]],
        prediction_horizon: int = 10
    ) -> np.ndarray:
        """
        Generate binary danger labels for crew positions.

        Args:
            flame_length_sequence: Array of shape (T, H, W) with flame lengths
            crew_positions: List of (row, col) crew positions
            prediction_horizon: Number of timesteps ahead to check (minutes)

        Returns:
            Binary label array of shape (T-horizon, num_crews)
            1 = danger, 0 = safe
        """
        T, H, W = flame_length_sequence.shape
        num_crews = len(crew_positions)

        # Number of valid timesteps (excluding last 'horizon' steps)
        valid_steps = T - prediction_horizon

        if valid_steps <= 0:
            raise ValueError(
                f"Sequence too short: {T} timesteps with horizon {prediction_horizon}"
            )

        labels = np.zeros((valid_steps, num_crews), dtype=np.int32)

        for t in range(valid_steps):
            # Check future state at t + horizon
            future_flame = flame_length_sequence[t + prediction_horizon]

            # Create danger zone mask
            danger_mask = future_flame > self.flame_threshold

            # Compute distance to danger
            distance_to_danger = self.compute_fire_distance(danger_mask)

            # Check each crew position
            for crew_idx, (row, col) in enumerate(crew_positions):
                if 0 <= row < H and 0 <= col < W:
                    dist = distance_to_danger[row, col]

                    # Label as danger if within buffer distance
                    if dist < self.buffer_distance:
                        labels[t, crew_idx] = 1

        logger.info(
            f"Generated {labels.shape[0]} labels for {num_crews} crews "
            f"with {prediction_horizon}min horizon. Danger rate: {labels.mean():.2%}"
        )

        return labels

    def generate_multi_horizon_labels(
        self,
        flame_length_sequence: np.ndarray,
        crew_positions: List[Tuple[int, int]],
        horizons: List[int] = [5, 10, 20]
    ) -> Dict[int, np.ndarray]:
        """
        Generate labels for multiple prediction horizons.

        Args:
            flame_length_sequence: Flame length time series
            crew_positions: Crew position coordinates
            horizons: List of prediction horizons in minutes

        Returns:
            Dictionary mapping horizon -> label array
        """
        labels_dict = {}

        for horizon in horizons:
            labels = self.generate_danger_labels(
                flame_length_sequence,
                crew_positions,
                prediction_horizon=horizon
            )
            labels_dict[horizon] = labels

        return labels_dict
