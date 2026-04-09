"""
Inference predictor for real-time danger assessment
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from ..models.convlstm_model import ConvLSTMModel

logger = logging.getLogger(__name__)


class WildfirePredictor:
    """
    Real-time wildfire danger predictor for crew positions.
    """

    def __init__(
        self,
        model_path: str,
        model_config: Dict,
        num_crews: int,
        threshold: float = 0.5
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model weights
            model_config: Model architecture configuration
            num_crews: Number of crew positions
            threshold: Classification threshold for danger
        """
        self.model_path = model_path
        self.threshold = threshold
        self.num_crews = num_crews

        # Initialize and load model
        input_shape = tuple(model_config['input_shape'])
        self.model = ConvLSTMModel(
            input_shape=input_shape,
            num_crews=num_crews,
            convlstm_filters=model_config.get('convlstm_filters', [64, 32, 16]),
            kernel_size=tuple(model_config.get('kernel_size', [3, 3])),
            dense_units=model_config.get('dense_units', [128, 64]),
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )

        self.model.load(model_path)

        logger.info(f"Loaded predictor from {model_path}")

    def predict_danger(
        self,
        flame_length_sequence: np.ndarray,
        spread_rate_sequence: np.ndarray,
        crew_positions: List[Tuple[int, int]]
    ) -> Dict:
        """
        Predict danger for crew positions.

        Args:
            flame_length_sequence: Recent flame length data (T, H, W)
            spread_rate_sequence: Recent spread rate data (T, H, W)
            crew_positions: List of crew positions

        Returns:
            Dictionary with predictions and metadata
        """
        # Stack features
        features = np.stack([flame_length_sequence, spread_rate_sequence], axis=-1)

        # Add batch dimension
        X = np.expand_dims(features, axis=0)  # (1, T, H, W, 2)

        # Predict
        probabilities = self.model.predict(X)[0]  # (num_crews,)

        # Apply threshold
        predictions = (probabilities > self.threshold).astype(int)

        # Build result
        result = {
            'timestamp': 'current',  # Replace with actual timestamp
            'predictions': []
        }

        for i, (row, col) in enumerate(crew_positions):
            result['predictions'].append({
                'crew_id': i,
                'position': {'row': int(row), 'col': int(col)},
                'danger_probability': float(probabilities[i]),
                'danger_alert': bool(predictions[i]),
                'severity': self._get_severity_level(probabilities[i])
            })

        logger.info(
            f"Predicted danger for {len(crew_positions)} crews: "
            f"{predictions.sum()} alerts"
        )

        return result

    def _get_severity_level(self, probability: float) -> str:
        """
        Map probability to severity level.

        Args:
            probability: Danger probability

        Returns:
            Severity level string
        """
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.6:
            return 'MODERATE'
        elif probability < 0.8:
            return 'HIGH'
        else:
            return 'CRITICAL'

    def predict_batch(
        self,
        sequences: List[np.ndarray],
        crew_positions: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        Predict for multiple sequences in batch.

        Args:
            sequences: List of feature sequences
            crew_positions: Crew positions

        Returns:
            List of prediction dictionaries
        """
        results = []

        for seq in sequences:
            # Assume seq is (T, H, W, 2) already stacked
            X = np.expand_dims(seq, axis=0)
            probabilities = self.model.predict(X)[0]
            predictions = (probabilities > self.threshold).astype(int)

            results.append({
                'probabilities': probabilities.tolist(),
                'predictions': predictions.tolist(),
                'num_alerts': int(predictions.sum())
            })

        return results
