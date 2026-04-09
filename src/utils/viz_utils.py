"""
Visualization utilities for wildfire data and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VizUtils:
    """
    Visualization utilities for wildfire simulations and predictions.
    """

    @staticmethod
    def create_fire_colormap():
        """
        Create a fire-themed colormap.

        Returns:
            Matplotlib colormap
        """
        colors = ['white', 'yellow', 'orange', 'red', 'darkred']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('fire', colors, N=n_bins)
        return cmap

    @staticmethod
    def plot_flame_length(
        flame_length: np.ndarray,
        crew_positions: Optional[List[Tuple[int, int]]] = None,
        title: str = "Flame Length",
        save_path: Optional[str] = None
    ):
        """
        Plot flame length raster with crew positions.

        Args:
            flame_length: Flame length array (H, W)
            crew_positions: Optional list of crew positions
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot flame length
        im = ax.imshow(
            flame_length,
            cmap=VizUtils.create_fire_colormap(),
            interpolation='nearest'
        )

        # Add crew positions
        if crew_positions:
            rows, cols = zip(*crew_positions)
            ax.scatter(cols, rows, c='blue', s=100, marker='o',
                      edgecolors='white', linewidths=2, label='Crew Positions')
            ax.legend()

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Flame Length (m)', rotation=270, labelpad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        plt.show()

    @staticmethod
    def plot_danger_prediction(
        flame_length: np.ndarray,
        predictions: np.ndarray,
        crew_positions: List[Tuple[int, int]],
        title: str = "Danger Prediction",
        save_path: Optional[str] = None
    ):
        """
        Plot flame length with danger predictions overlay.

        Args:
            flame_length: Flame length array
            predictions: Binary predictions per crew (num_crews,)
            crew_positions: Crew position coordinates
            title: Plot title
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=(12, 9))

        # Plot flame length
        im = ax.imshow(
            flame_length,
            cmap=VizUtils.create_fire_colormap(),
            interpolation='nearest'
        )

        # Plot crew positions with color-coded danger
        for i, (row, col) in enumerate(crew_positions):
            color = 'red' if predictions[i] == 1 else 'green'
            marker = 'X' if predictions[i] == 1 else 'o'
            size = 200 if predictions[i] == 1 else 150

            ax.scatter(col, row, c=color, s=size, marker=marker,
                      edgecolors='white', linewidths=2,
                      label=f"Crew {i} ({'DANGER' if predictions[i] == 1 else 'Safe'})")

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Flame Length (m)', rotation=270, labelpad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        plt.show()

    @staticmethod
    def create_animation(
        flame_sequence: np.ndarray,
        crew_positions: List[Tuple[int, int]],
        predictions: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        fps: int = 5
    ):
        """
        Create animation of fire progression.

        Args:
            flame_sequence: Sequence of flame length arrays (T, H, W)
            crew_positions: Crew positions
            predictions: Optional time series of predictions (T, num_crews)
            save_path: Optional path to save animation
            fps: Frames per second
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        T = flame_sequence.shape[0]
        cmap = VizUtils.create_fire_colormap()

        def update(frame):
            ax.clear()

            # Plot current frame
            im = ax.imshow(
                flame_sequence[frame],
                cmap=cmap,
                interpolation='nearest'
            )

            # Plot crews
            if predictions is not None:
                for i, (row, col) in enumerate(crew_positions):
                    color = 'red' if predictions[frame, i] == 1 else 'green'
                    marker = 'X' if predictions[frame, i] == 1 else 'o'
                    ax.scatter(col, row, c=color, s=150, marker=marker,
                             edgecolors='white', linewidths=2)
            else:
                rows, cols = zip(*crew_positions)
                ax.scatter(cols, rows, c='blue', s=100, marker='o',
                         edgecolors='white', linewidths=2)

            ax.set_title(f"Time Step: {frame+1}/{T}", fontsize=14, fontweight='bold')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')

            return [im]

        anim = animation.FuncAnimation(
            fig, update, frames=T, interval=1000/fps, blit=False
        )

        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            logger.info(f"Saved animation to {save_path}")

        plt.show()

    @staticmethod
    def plot_training_history(
        history: dict,
        save_path: Optional[str] = None
    ):
        """
        Plot training history curves.

        Args:
            history: Keras training history dictionary
            save_path: Optional save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['loss', 'accuracy', 'precision', 'recall']
        titles = ['Loss', 'Accuracy', 'Precision', 'Recall']

        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            if metric in history:
                ax.plot(history[metric], label='Train')
                if f'val_{metric}' in history:
                    ax.plot(history[f'val_{metric}'], label='Validation')

                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history to {save_path}")

        plt.show()
