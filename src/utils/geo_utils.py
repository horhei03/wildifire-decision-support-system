"""
Geospatial utility functions
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GeoUtils:
    """
    Geospatial processing utilities for wildfire data.
    """

    @staticmethod
    def pixel_to_coords(
        row: int,
        col: int,
        transform: rasterio.Affine
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.

        Args:
            row: Row index
            col: Column index
            transform: Affine transformation

        Returns:
            Tuple of (x, y) geographic coordinates
        """
        x, y = transform * (col, row)
        return x, y

    @staticmethod
    def coords_to_pixel(
        x: float,
        y: float,
        transform: rasterio.Affine
    ) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            transform: Affine transformation

        Returns:
            Tuple of (row, col) pixel indices
        """
        inv_transform = ~transform
        col, row = inv_transform * (x, y)
        return int(row), int(col)

    @staticmethod
    def calculate_distance_matrix(
        shape: Tuple[int, int],
        center: Tuple[int, int],
        resolution: float = 5.0
    ) -> np.ndarray:
        """
        Calculate distance from each cell to a center point.

        Args:
            shape: (height, width) of grid
            center: (row, col) of center point
            resolution: Cell resolution in meters

        Returns:
            Distance matrix in meters
        """
        rows, cols = np.indices(shape)
        center_row, center_col = center

        distances = np.sqrt(
            ((rows - center_row) ** 2 + (cols - center_col) ** 2)
        ) * resolution

        return distances

    @staticmethod
    def create_buffer_mask(
        shape: Tuple[int, int],
        point: Tuple[int, int],
        buffer_distance: float,
        resolution: float = 5.0
    ) -> np.ndarray:
        """
        Create a circular buffer mask around a point.

        Args:
            shape: Grid shape
            point: Center point (row, col)
            buffer_distance: Buffer radius in meters
            resolution: Cell resolution in meters

        Returns:
            Binary mask (1 inside buffer, 0 outside)
        """
        distances = GeoUtils.calculate_distance_matrix(shape, point, resolution)
        mask = (distances <= buffer_distance).astype(np.float32)

        return mask

    @staticmethod
    def reproject_raster(
        src_array: np.ndarray,
        src_crs: str,
        src_transform: rasterio.Affine,
        dst_crs: str,
        resolution: Optional[float] = None
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Reproject raster to different CRS.

        Args:
            src_array: Source array
            src_crs: Source CRS
            src_transform: Source transform
            dst_crs: Destination CRS
            resolution: Optional output resolution

        Returns:
            Tuple of (reprojected array, new transform)
        """
        src_height, src_width = src_array.shape

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_width, src_height,
            *rasterio.transform.array_bounds(src_height, src_width, src_transform),
            resolution=resolution
        )

        dst_array = np.zeros((dst_height, dst_width), dtype=src_array.dtype)

        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

        logger.info(f"Reprojected from {src_crs} to {dst_crs}")

        return dst_array, dst_transform

    @staticmethod
    def calculate_aspect_slope(elevation: np.ndarray, resolution: float = 5.0):
        """
        Calculate aspect and slope from elevation DEM.

        Args:
            elevation: Elevation array
            resolution: Cell resolution in meters

        Returns:
            Tuple of (aspect, slope) in degrees
        """
        # Calculate gradients
        dy, dx = np.gradient(elevation, resolution)

        # Slope in degrees
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi

        # Aspect in degrees (0-360, clockwise from North)
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect = (aspect + 360) % 360

        return aspect, slope
