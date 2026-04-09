"""
Complete dataset pipeline from FARSITE to training tensors
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from .farsite_parser import FARSITEParser
from .label_generator import LabelGenerator
from .tensor_builder import TensorBuilder

logger = logging.getLogger(__name__)


class DatasetPipeline:
    """
    End-to-end pipeline for dataset generation.
    """

    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to data_config.yaml
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_root = Path(self.config['data_root'])

        danger_cfg = self.config['danger_criteria']
        self.label_gen = LabelGenerator(
            buffer_distance=danger_cfg['buffer_distance'],
            flame_threshold=danger_cfg['flame_length_threshold'],
            resolution=self.config['spatial']['resolution']
        )

        self.tensor_builder = TensorBuilder(
            spatial_size=tuple(self.config['spatial']['patch_size']),
            normalize=False  # Disable normalization - will normalize during training
        )

        logger.info("Initialized DatasetPipeline")

    def load_crew_positions(self, config_path: str) -> List[Tuple[int, int]]:
        """
        Load crew positions from configuration.

        Args:
            config_path: Path to crew_positions.yaml

        Returns:
            List of (row, col) positions
        """
        with open(config_path, 'r') as f:
            crew_config = yaml.safe_load(f)

        positions = [
            (pos['row'], pos['col'])
            for pos in crew_config['positions']
        ]

        return positions

    def convert_to_time_series(
        self,
        arrival_time: np.ndarray,
        flame_length: np.ndarray,
        rate_of_spread: np.ndarray,
        timestep_minutes: int = 1,
        downsample_factor: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert FARSITE static outputs to time series.

        Args:
            arrival_time: (H, W) array of arrival times in minutes
            flame_length: (H, W) array of max flame lengths
            rate_of_spread: (H, W) array of max spread rates
            timestep_minutes: Time resolution in minutes
            downsample_factor: Factor to reduce spatial resolution (e.g., 5 → 1600→320)

        Returns:
            Tuple of (flame_length_series, spread_rate_series)
            Both with shape (T, H_downsampled, W_downsampled)
        """
        # Downsample spatial resolution first to reduce memory
        if downsample_factor > 1:
            H, W = arrival_time.shape
            new_H = H // downsample_factor
            new_W = W // downsample_factor

            # Use max pooling for flame length and spread rate to preserve peak values
            arrival_time = arrival_time[:new_H*downsample_factor, :new_W*downsample_factor]
            arrival_time = arrival_time.reshape(new_H, downsample_factor, new_W, downsample_factor)
            arrival_time = np.nanmin(arrival_time, axis=(1, 3))  # Min arrival time

            flame_length = flame_length[:new_H*downsample_factor, :new_W*downsample_factor]
            flame_length = flame_length.reshape(new_H, downsample_factor, new_W, downsample_factor)
            flame_length = np.nanmax(flame_length, axis=(1, 3))  # Max flame length

            rate_of_spread = rate_of_spread[:new_H*downsample_factor, :new_W*downsample_factor]
            rate_of_spread = rate_of_spread.reshape(new_H, downsample_factor, new_W, downsample_factor)
            rate_of_spread = np.nanmax(rate_of_spread, axis=(1, 3))  # Max spread rate

        # Get unique timesteps
        valid_times = arrival_time[~np.isnan(arrival_time)]
        if len(valid_times) == 0:
            raise ValueError("No valid arrival times found")

        max_time = int(np.ceil(valid_times.max()))
        num_timesteps = max_time // timestep_minutes + 1

        H, W = arrival_time.shape
        flame_series = np.zeros((num_timesteps, H, W), dtype=np.float32)
        spread_series = np.zeros((num_timesteps, H, W), dtype=np.float32)

        # For each timestep, show which cells have burned
        for t in range(num_timesteps):
            current_time = t * timestep_minutes
            # Cells that have burned by this time
            burned_mask = (arrival_time <= current_time) & ~np.isnan(arrival_time)

            # Apply flame length and spread rate where burned
            flame_series[t][burned_mask] = flame_length[burned_mask]
            spread_series[t][burned_mask] = rate_of_spread[burned_mask]

        return flame_series, spread_series

    def process_single_simulation(
        self,
        patch_id: str,
        scenario: str,
        crew_positions: List[Tuple[int, int]],
        prediction_horizon: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single simulation into training data.

        Args:
            patch_id: Patch identifier
            scenario: Scenario name
            crew_positions: List of crew positions
            prediction_horizon: Prediction horizon in minutes

        Returns:
            Tuple of (X, y) tensors
        """
        logger.info(f"Processing {patch_id}/{scenario}")

        # Find patch folder name
        patch_folder = None
        for patch in self.config['patches']:
            if patch['id'] == patch_id:
                patch_folder = patch['folder']
                break

        if patch_folder is None:
            raise ValueError(f"Unknown patch_id: {patch_id}")

        # Build path to scenario
        scenario_path = self.data_root / patch_folder / "Outputs" / scenario

        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario path does not exist: {scenario_path}")

        # Create parser for this scenario
        parser = FARSITEParser(str(scenario_path))

        # Load all data
        data = parser.get_all_data()

        # Extract arrays
        arrival_time_array, at_meta = data['arrival_time']
        flame_length_array, fl_meta = data['flame_length']
        rate_of_spread_array, ros_meta = data['rate_of_spread']

        # Convert static FARSITE outputs to time series
        # Use 5-minute timesteps to reduce temporal resolution and memory
        flame_series, spread_series = self.convert_to_time_series(
            arrival_time_array,
            flame_length_array,
            rate_of_spread_array,
            timestep_minutes=5,  # Sample every 5 minutes instead of 1
            downsample_factor=5  # 1600×1600 → 320×320
        )

        logger.info(f"  Generated time series: {flame_series.shape[0]} timesteps")

        # Scale crew positions to match downsampled grid
        downsample_factor = 5
        scaled_positions = [
            (row // downsample_factor, col // downsample_factor)
            for row, col in crew_positions
        ]
        logger.info(f"  Scaled crew positions: {crew_positions} -> {scaled_positions}")

        # Generate labels
        labels = self.label_gen.generate_danger_labels(
            flame_series,
            scaled_positions,
            prediction_horizon=prediction_horizon
        )

        # Build tensors
        X, y = self.tensor_builder.build_sequences(
            flame_series,
            spread_series,
            labels,
            scaled_positions
        )

        return X, y

    def generate_full_dataset(
        self,
        crew_positions: List[Tuple[int, int]],
        output_dir: str,
        prediction_horizon: int = 10,
        train_split: float = 0.7,
        val_split: float = 0.15
    ) -> Dict[str, str]:
        """
        Generate complete training/validation/test dataset.

        Uses incremental saving to avoid memory issues with large datasets.

        Args:
            crew_positions: Crew position coordinates
            output_dir: Directory to save processed data
            prediction_horizon: Prediction horizon
            train_split: Fraction for training
            val_split: Fraction for validation

        Returns:
            Dictionary with paths to saved datasets
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for simulation chunks
        temp_dir = output_path / "temp_chunks"
        temp_dir.mkdir(exist_ok=True)

        chunk_files = []
        total_samples = 0

        # Process all patches and scenarios, saving each to disk
        logger.info("Processing simulations and saving to temporary files...")
        for patch in self.config['patches']:
            for scenario in self.config['scenarios']:
                try:
                    X, y = self.process_single_simulation(
                        patch['id'],
                        scenario,
                        crew_positions,
                        prediction_horizon
                    )

                    # Save this simulation's data to temporary file
                    chunk_idx = len(chunk_files)
                    chunk_path = temp_dir / f"chunk_{chunk_idx:03d}.npz"
                    np.savez_compressed(chunk_path, X=X, y=y)

                    chunk_files.append(chunk_path)
                    total_samples += X.shape[0]

                    logger.info(f"  Saved chunk {chunk_idx}: {X.shape[0]} samples")

                    # Free memory
                    del X, y

                except Exception as e:
                    logger.warning(f"Failed to process {patch['id']}/{scenario}: {e}")

        if len(chunk_files) == 0:
            raise ValueError("No simulations were successfully processed!")

        logger.info(f"Successfully processed {len(chunk_files)} simulations with {total_samples} total samples")

        # Determine split sizes and create shuffled indices
        logger.info("Determining train/val/test splits...")
        n_train = int(total_samples * train_split)
        n_val = int(total_samples * val_split)

        # Create shuffled indices for the entire dataset
        all_indices = np.random.permutation(total_samples)
        train_idx = set(all_indices[:n_train].tolist())
        val_idx = set(all_indices[n_train:n_train+n_val].tolist())
        test_idx = set(all_indices[n_train+n_val:].tolist())

        logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # Determine array shapes by reading first chunk
        logger.info("Determining array shapes...")
        first_chunk = np.load(chunk_files[0])
        sample_shape = first_chunk['X'].shape[1:]  # (timesteps, H, W, channels)
        label_shape = first_chunk['y'].shape[1:]    # (num_crews,)
        del first_chunk

        logger.info(f"Sample shape: {sample_shape}, Label shape: {label_shape}")

        # Process each split separately to minimize memory usage
        saved_paths = {}

        for split_name, split_indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            n_split = len(split_indices)
            logger.info(f"Processing {split_name} split ({n_split} samples)...")

            X_path = output_path / f"X_{split_name}.npy"
            y_path = output_path / f"y_{split_name}.npy"

            X_full_shape = (n_split,) + sample_shape
            y_full_shape = (n_split,) + label_shape

            # Write .npy headers (data will be streamed after)
            for path, shape in [(X_path, X_full_shape), (y_path, y_full_shape)]:
                with open(path, 'wb') as f:
                    np.lib.format.write_array_header_2_0(f, {
                        'descr': np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                        'fortran_order': False,
                        'shape': shape,
                    })

            # Stream data to files chunk by chunk (no large RAM allocation)
            split_pos = 0
            global_idx = 0
            with open(X_path, 'ab') as fx, open(y_path, 'ab') as fy:
                for chunk_path in chunk_files:
                    data = np.load(chunk_path)
                    X_chunk = data['X']
                    y_chunk = data['y']

                    for i in range(X_chunk.shape[0]):
                        if global_idx in split_indices:
                            fx.write(np.ascontiguousarray(X_chunk[i], dtype=np.float32).tobytes())
                            fy.write(np.ascontiguousarray(y_chunk[i], dtype=np.float32).tobytes())
                            split_pos += 1
                        global_idx += 1

                    del X_chunk, y_chunk, data

            saved_paths[split_name] = {
                'X': str(X_path),
                'y': str(y_path)
            }

            logger.info(f"  Saved {split_name}: X={X_full_shape}, y={y_full_shape}")

        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Could not remove temp dir {temp_dir}: {e}")

        return saved_paths
