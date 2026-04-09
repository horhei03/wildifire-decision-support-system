"""
Build training tensors from parsed FARSITE data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TensorBuilder:
    """
    Build spatiotemporal tensors for ConvLSTM training from FARSITE outputs.
    """

    def __init__(
        self,
        input_timesteps: int = 10,
        spatial_size: Tuple[int, int] = (1600, 1600),
        normalize: bool = True
    ):
        """
        Initialize tensor builder.

        Args:
            input_timesteps: Number of historical timesteps for input
            spatial_size: (height, width) of spatial grid
            normalize: Whether to normalize features
        """
        self.input_timesteps = input_timesteps
        self.spatial_size = spatial_size
        self.normalize = normalize

        # Normalization statistics (to be computed from training data)
        self.feature_stats = None

        logger.info(
            f"Initialized TensorBuilder: input_steps={input_timesteps}, "
            f"spatial={spatial_size}, normalize={normalize}"
        )

    def compute_normalization_stats(
        self,
        flame_length_data: List[np.ndarray],
        spread_rate_data: List[np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean and std for normalization from training data.

        Args:
            flame_length_data: List of flame length arrays
            spread_rate_data: List of spread rate arrays

        Returns:
            Dictionary with mean/std for each feature
        """
        # Stack all data
        all_flame = np.concatenate([arr.flatten() for arr in flame_length_data])
        all_spread = np.concatenate([arr.flatten() for arr in spread_rate_data])

        # Remove nodata values (assuming -1 or similar)
        all_flame = all_flame[all_flame >= 0]
        all_spread = all_spread[all_spread >= 0]

        stats = {
            'flame_length': {
                'mean': float(np.mean(all_flame)),
                'std': float(np.std(all_flame))
            },
            'spread_rate': {
                'mean': float(np.mean(all_spread)),
                'std': float(np.std(all_spread))
            }
        }

        self.feature_stats = stats
        logger.info(f"Computed normalization stats: {stats}")

        return stats

    def normalize_features(self, data: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Normalize features using precomputed statistics.

        Args:
            data: Input data array
            feature_name: Name of feature ('flame_length' or 'spread_rate')

        Returns:
            Normalized data
        """
        if self.feature_stats is None:
            raise ValueError("Normalization stats not computed. Call compute_normalization_stats first.")

        stats = self.feature_stats[feature_name]
        normalized = (data - stats['mean']) / (stats['std'] + 1e-8)

        return normalized

    def build_sequences(
        self,
        flame_length: np.ndarray,
        spread_rate: np.ndarray,
        labels: np.ndarray,
        crew_positions: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build input-output sequences for training.

        Args:
            flame_length: Array of shape (T, H, W)
            spread_rate: Array of shape (T, H, W)
            labels: Binary labels of shape (T_valid, num_crews)
            crew_positions: List of crew positions

        Returns:
            Tuple of (X, y) where:
            - X: shape (num_samples, input_timesteps, H, W, 2)
            - y: shape (num_samples, num_crews)
        """
        T, H, W = flame_length.shape
        num_crews = len(crew_positions)

        # Normalize if requested
        if self.normalize:
            flame_length = self.normalize_features(flame_length, 'flame_length')
            spread_rate = self.normalize_features(spread_rate, 'spread_rate')

        # Stack features into channels
        features = np.stack([flame_length, spread_rate], axis=-1)  # (T, H, W, 2)

        # Create sliding windows
        num_samples = labels.shape[0]
        X = np.zeros((num_samples, self.input_timesteps, H, W, 2), dtype=np.float32)
        y = labels.astype(np.float32)

        for i in range(num_samples):
            X[i] = features[i:i+self.input_timesteps]

        logger.info(
            f"Built {num_samples} sequences: X={X.shape}, y={y.shape}"
        )

        return X, y

    def create_crew_position_mask(
        self,
        crew_positions: List[Tuple[int, int]],
        spatial_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Create a spatial mask indicating crew positions.

        Args:
            crew_positions: List of (row, col) positions
            spatial_size: Optional override for spatial dimensions

        Returns:
            Binary mask of shape (H, W) with 1 at crew positions
        """
        if spatial_size is None:
            spatial_size = self.spatial_size

        mask = np.zeros(spatial_size, dtype=np.float32)

        for row, col in crew_positions:
            if 0 <= row < spatial_size[0] and 0 <= col < spatial_size[1]:
                mask[row, col] = 1.0

        return mask
