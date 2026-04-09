"""
FARSITE output parser for wildfire simulation data

Reads exported FARSITE simulation outputs including arrival time, flame length,
rate of spread, spread direction, and perimeter shapefiles.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

import numpy as np
import rasterio
from rasterio.crs import CRS
import geopandas as gpd

logger = logging.getLogger('farsite_parser')


def parse_scenario_name(scenario_name: str) -> Dict[str, any]:
    """
    Parse scenario name to extract wind speed, direction, and moisture condition.

    Args:
        scenario_name: Scenario folder name (e.g., "ws12_wd270_extreme")

    Returns:
        Dictionary with keys: wind_speed, wind_direction, moisture_condition

    Examples:
        >>> parse_scenario_name("ws12_wd90_dry")
        {'wind_speed': 12, 'wind_direction': 90, 'moisture_condition': 'dry'}

        >>> parse_scenario_name("ws25_wd270_extreme")
        {'wind_speed': 25, 'wind_direction': 270, 'moisture_condition': 'extreme'}
    """
    # Pattern: ws{speed}_wd{direction}_{moisture}
    # Also handles ws_25 format (with extra underscore)
    pattern = r'ws_?(\d+)_wd(\d+)_(dry|extreme)'
    match = re.search(pattern, scenario_name.lower())

    if not match:
        logger.warning(f"Could not parse scenario name: {scenario_name}")
        return {
            'wind_speed': None,
            'wind_direction': None,
            'moisture_condition': None
        }

    return {
        'wind_speed': int(match.group(1)),
        'wind_direction': int(match.group(2)),
        'moisture_condition': match.group(3)
    }


class FARSITEParser:
    """
    Parse FARSITE simulation outputs for a single scenario.

    Args:
        scenario_dir: Path to scenario folder (e.g.,
            "C:/Users/horhe/OneDrive/Documents/DSS/malaga_files/P1_darkforest/Outputs/ws12_wd90_dry")

    Attributes:
        scenario_dir: Path to scenario directory
        patch_id: Patch identifier (e.g., "P1")
        patch_name: Patch name (e.g., "darkforest")
        scenario_name: Scenario name (e.g., "ws12_wd90_dry")
        wind_speed: Wind speed in km/h
        wind_direction: Wind direction in degrees
        moisture_condition: Moisture condition ("dry" or "extreme")

    Examples:
        >>> parser = FARSITEParser(
        ...     "C:/Users/horhe/OneDrive/Documents/DSS/malaga_files/P1_darkforest/Outputs/ws12_wd90_dry"
        ... )
        >>> print(parser)
        FARSITEParser(P1_darkforest, ws12_wd90_dry: WS=12km/h, WD=270°, dry)

        >>> arrival = parser.load_arrival_time()
        >>> print(arrival.shape)
        (1600, 1600)

        >>> metadata = parser.get_metadata()
        >>> print(metadata['crs'])
        EPSG:25830
    """

    def __init__(self, scenario_dir: str):
        """
        Initialize FARSITE parser for a scenario.

        Args:
            scenario_dir: Path to scenario output folder
        """
        self.scenario_dir = Path(scenario_dir)

        if not self.scenario_dir.exists():
            raise FileNotFoundError(
                f"Scenario directory not found: {self.scenario_dir}\n"
                f"Please check the path and ensure FARSITE outputs exist."
            )

        # Parse path to extract metadata
        self._parse_path_metadata()

        # Cache for loaded data
        self._cache = {}

        logger.info(f"Initialized FARSITEParser: {self}")

    def _parse_path_metadata(self):
        """Parse patch and scenario information from directory path."""
        # Get scenario name (last directory)
        self.scenario_name = self.scenario_dir.name

        # Parse scenario parameters
        scenario_params = parse_scenario_name(self.scenario_name)
        self.wind_speed = scenario_params['wind_speed']
        self.wind_direction = scenario_params['wind_direction']
        self.moisture_condition = scenario_params['moisture_condition']

        # Get patch information (2 levels up, then parent name)
        # Path structure: .../P1_darkforest/Outputs/ws12_wd90_dry/
        patch_folder = self.scenario_dir.parent.parent.name

        # Extract patch ID and name
        # Expected format: "P1_darkforest"
        match = re.match(r'(P\d+)_(.+)', patch_folder)
        if match:
            self.patch_id = match.group(1)
            self.patch_name = match.group(2)
        else:
            # Fallback
            self.patch_id = patch_folder
            self.patch_name = patch_folder
            logger.warning(
                f"Could not parse patch info from folder name: {patch_folder}"
            )

    def _find_file(self, pattern: str) -> Optional[Path]:
        """
        Find file matching pattern in scenario directory.

        Tries multiple strategies to handle missing or irregular extensions.

        Args:
            pattern: Base filename pattern (e.g., "ArrivalTime")

        Returns:
            Path to found file, or None if not found
        """
        # Strategy 1: Exact match with .tif
        exact_tif = self.scenario_dir / f"{pattern}.tif"
        if exact_tif.exists():
            return exact_tif

        # Strategy 2: Exact match without extension
        exact_no_ext = self.scenario_dir / pattern
        if exact_no_ext.exists():
            return exact_no_ext

        # Strategy 3: Glob for any extension
        matches = list(self.scenario_dir.glob(f"{pattern}*"))

        # Filter out shapefiles and auxiliary files
        valid_extensions = {'.tif', '.tiff', ''}  # '' for no extension
        matches = [
            m for m in matches
            if m.suffix.lower() in valid_extensions or m.suffix == ''
        ]

        if matches:
            if len(matches) > 1:
                logger.warning(
                    f"Multiple files match pattern '{pattern}': {[m.name for m in matches]}. "
                    f"Using first: {matches[0].name}"
                )
            return matches[0]

        return None

    def _load_raster(self, pattern: str) -> Tuple[np.ndarray, Dict]:
        """
        Load raster file and return data with metadata.

        Args:
            pattern: Filename pattern (e.g., "FlameLength")

        Returns:
            Tuple of (data array, metadata dict)

        Raises:
            FileNotFoundError: If file matching pattern is not found
        """
        filepath = self._find_file(pattern)

        if filepath is None:
            raise FileNotFoundError(
                f"Could not find file matching pattern '{pattern}' in {self.scenario_dir}\n"
                f"Expected one of:\n"
                f"  - {pattern}.tif\n"
                f"  - {pattern}\n"
                f"  - {pattern}* (with any extension)\n"
                f"Available files: {[f.name for f in self.scenario_dir.iterdir() if f.is_file()]}"
            )

        logger.debug(f"Loading raster: {filepath.name}")

        try:
            with rasterio.open(filepath) as src:
                # Read first band
                data = src.read(1)

                # Get nodata value
                nodata = src.nodata
                if nodata is None:
                    # Try to detect common nodata values
                    nodata = -9999.0
                    logger.warning(
                        f"No nodata value set in {filepath.name}, assuming {nodata}"
                    )

                # Mask nodata values
                data_masked = np.where(data == nodata, np.nan, data)

                # Extract metadata
                metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'nodata': nodata,
                    'width': src.width,
                    'height': src.height,
                    'resolution': (src.transform.a, abs(src.transform.e)),
                    'filepath': str(filepath)
                }

                # Check CRS
                if metadata['crs'] and metadata['crs'] != CRS.from_epsg(25830):
                    warnings.warn(
                        f"CRS {metadata['crs']} differs from expected EPSG:25830 "
                        f"in file {filepath.name}",
                        UserWarning
                    )

                return data_masked, metadata

        except Exception as e:
            raise IOError(
                f"Error reading raster file {filepath}: {str(e)}\n"
                f"Please verify the file is a valid GeoTIFF."
            ) from e

    def load_arrival_time(self) -> Tuple[np.ndarray, Dict]:
        """
        Load fire arrival time raster.

        Returns:
            Tuple of (arrival time array in minutes, metadata dict)
            NoData values are masked as NaN.

        Raises:
            FileNotFoundError: If ArrivalTime file not found
        """
        if 'arrival_time' not in self._cache:
            self._cache['arrival_time'] = self._load_raster("ArrivalTime")
        return self._cache['arrival_time']

    def load_flame_length(self) -> Tuple[np.ndarray, Dict]:
        """
        Load flame length raster.

        Returns:
            Tuple of (flame length array in meters, metadata dict)
            NoData values are masked as NaN.

        Raises:
            FileNotFoundError: If FlameLength file not found
        """
        if 'flame_length' not in self._cache:
            self._cache['flame_length'] = self._load_raster("FlameLength")
        return self._cache['flame_length']

    def load_rate_of_spread(self) -> Tuple[np.ndarray, Dict]:
        """
        Load rate of spread raster.

        Returns:
            Tuple of (spread rate array in m/min, metadata dict)
            NoData values are masked as NaN.

        Raises:
            FileNotFoundError: If RateOfSpread file not found
        """
        if 'rate_of_spread' not in self._cache:
            self._cache['rate_of_spread'] = self._load_raster("RateOfSpread")
        return self._cache['rate_of_spread']

    def load_spread_direction(self) -> Tuple[np.ndarray, Dict]:
        """
        Load spread direction raster.

        Returns:
            Tuple of (direction array in degrees, metadata dict)
            NoData values are masked as NaN.

        Raises:
            FileNotFoundError: If SpreadDirection file not found
        """
        if 'spread_direction' not in self._cache:
            self._cache['spread_direction'] = self._load_raster("SpreadDirection")
        return self._cache['spread_direction']

    def load_perimeters(self) -> Optional[gpd.GeoDataFrame]:
        """
        Load fire perimeters shapefile.

        Returns:
            GeoDataFrame with perimeter polygons and timestep information,
            or None if file not found.
        """
        if 'perimeters' in self._cache:
            return self._cache['perimeters']

        # Look for Perimeters.shp
        shp_file = self.scenario_dir / "Perimeters.shp"

        if not shp_file.exists():
            logger.warning(
                f"Perimeters.shp not found in {self.scenario_dir}. "
                f"Perimeter data will not be available."
            )
            self._cache['perimeters'] = None
            return None

        try:
            gdf = gpd.read_file(shp_file)

            # Ensure CRS is set
            if gdf.crs is None:
                logger.warning(
                    "Perimeters.shp has no CRS defined. Assuming EPSG:25830."
                )
                gdf.set_crs(epsg=25830, inplace=True)
            elif gdf.crs != CRS.from_epsg(25830):
                warnings.warn(
                    f"Perimeters CRS {gdf.crs} differs from expected EPSG:25830",
                    UserWarning
                )

            logger.info(f"Loaded {len(gdf)} fire perimeters")
            self._cache['perimeters'] = gdf
            return gdf

        except Exception as e:
            logger.error(f"Error reading Perimeters.shp: {e}")
            self._cache['perimeters'] = None
            return None

    def get_num_timesteps(self) -> int:
        """
        Get number of unique timesteps from arrival time data.

        Returns:
            Number of unique arrival time values (excluding nodata)
        """
        arrival, _ = self.load_arrival_time()
        unique_times = np.unique(arrival[~np.isnan(arrival)])
        return len(unique_times)

    def get_metadata(self) -> Dict:
        """
        Get comprehensive metadata for this scenario.

        Returns:
            Dictionary with scenario metadata including:
            - patch_id, patch_name
            - scenario_name, scenario_id
            - wind_speed, wind_direction, moisture_condition
            - spatial_extent, crs, resolution
            - num_timesteps

        Examples:
            >>> parser = FARSITEParser("path/to/scenario")
            >>> meta = parser.get_metadata()
            >>> print(f"Wind: {meta['wind_speed']} km/h at {meta['wind_direction']}°")
        """
        # Load one raster to get spatial metadata
        try:
            _, raster_meta = self.load_arrival_time()
        except FileNotFoundError:
            # Try another file
            try:
                _, raster_meta = self.load_flame_length()
            except FileNotFoundError:
                logger.warning("Could not load any raster files for metadata")
                raster_meta = {
                    'crs': None,
                    'bounds': None,
                    'resolution': None,
                    'width': None,
                    'height': None
                }

        metadata = {
            # Patch information
            'patch_id': self.patch_id,
            'patch_name': self.patch_name,

            # Scenario information
            'scenario_name': self.scenario_name,
            'scenario_id': f"{self.patch_id}_{self.scenario_name}",

            # Weather parameters
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'moisture_condition': self.moisture_condition,

            # Spatial information
            'crs': str(raster_meta.get('crs', 'Unknown')),
            'bounds': raster_meta.get('bounds'),
            'resolution': raster_meta.get('resolution'),
            'width': raster_meta.get('width'),
            'height': raster_meta.get('height'),

            # Temporal information
            'num_timesteps': self.get_num_timesteps(),

            # Paths
            'scenario_dir': str(self.scenario_dir)
        }

        return metadata

    def get_all_data(self) -> Dict:
        """
        Load all available data layers in a single call.

        Returns:
            Dictionary with keys:
            - 'arrival_time': (array, metadata) tuple
            - 'flame_length': (array, metadata) tuple
            - 'rate_of_spread': (array, metadata) tuple
            - 'spread_direction': (array, metadata) tuple
            - 'perimeters': GeoDataFrame or None
            - 'metadata': scenario metadata dict

        Examples:
            >>> parser = FARSITEParser("path/to/scenario")
            >>> data = parser.get_all_data()
            >>> flame_length, fl_meta = data['flame_length']
            >>> print(f"Max flame length: {np.nanmax(flame_length):.2f} m")
        """
        data = {
            'metadata': self.get_metadata()
        }

        # Load raster data
        try:
            data['arrival_time'] = self.load_arrival_time()
        except FileNotFoundError as e:
            logger.warning(f"Could not load arrival time: {e}")
            data['arrival_time'] = None

        try:
            data['flame_length'] = self.load_flame_length()
        except FileNotFoundError as e:
            logger.warning(f"Could not load flame length: {e}")
            data['flame_length'] = None

        try:
            data['rate_of_spread'] = self.load_rate_of_spread()
        except FileNotFoundError as e:
            logger.warning(f"Could not load rate of spread: {e}")
            data['rate_of_spread'] = None

        try:
            data['spread_direction'] = self.load_spread_direction()
        except FileNotFoundError as e:
            logger.warning(f"Could not load spread direction: {e}")
            data['spread_direction'] = None

        # Load vector data
        data['perimeters'] = self.load_perimeters()

        return data

    def clear_cache(self):
        """Clear cached data to free memory."""
        self._cache.clear()
        logger.debug("Cleared data cache")

    def __repr__(self) -> str:
        """String representation showing scenario information."""
        return (
            f"FARSITEParser({self.patch_id}_{self.patch_name}, "
            f"{self.scenario_name}: "
            f"WS={self.wind_speed}km/h, WD={self.wind_direction}°, "
            f"{self.moisture_condition})"
        )


def load_all_scenarios(
    base_dir: str,
    patches: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None
) -> Dict[str, FARSITEParser]:
    """
    Load parsers for multiple scenarios.

    Args:
        base_dir: Base directory containing patch folders
            (e.g., "C:/Users/horhe/OneDrive/Documents/DSS/malaga_files")
        patches: Optional list of patch IDs to load (e.g., ["P1", "P2"])
            If None, loads all patches found
        scenarios: Optional list of scenario names to load
            (e.g., ["ws12_wd90_dry", "ws25_wd270_extreme"])
            If None, loads all scenarios found

    Returns:
        Dictionary mapping scenario_id -> FARSITEParser
        where scenario_id is "{patch_id}_{scenario_name}"

    Examples:
        >>> parsers = load_all_scenarios(
        ...     "C:/Users/horhe/OneDrive/Documents/DSS/malaga_files",
        ...     patches=["P1", "P2"],
        ...     scenarios=["ws12_wd90_dry"]
        ... )
        >>> print(f"Loaded {len(parsers)} scenarios")
        >>> for scenario_id, parser in parsers.items():
        ...     print(f"  {scenario_id}: {parser}")
    """
    base_path = Path(base_dir)
    parsers = {}

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Find all patch directories
    patch_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    # Filter by patch ID if specified
    if patches:
        patch_dirs = [
            d for d in patch_dirs
            if any(d.name.startswith(p) for p in patches)
        ]

    logger.info(f"Found {len(patch_dirs)} patch directories")

    for patch_dir in patch_dirs:
        outputs_dir = patch_dir / "Outputs"

        if not outputs_dir.exists():
            logger.warning(f"No Outputs folder in {patch_dir.name}, skipping")
            continue

        # Find all scenario directories
        scenario_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]

        # Filter by scenario name if specified
        if scenarios:
            scenario_dirs = [d for d in scenario_dirs if d.name in scenarios]

        for scenario_dir in scenario_dirs:
            try:
                parser = FARSITEParser(str(scenario_dir))
                scenario_id = parser.get_metadata()['scenario_id']
                parsers[scenario_id] = parser
                logger.info(f"Loaded parser for {scenario_id}")

            except Exception as e:
                logger.error(
                    f"Error loading scenario {scenario_dir.name} "
                    f"from {patch_dir.name}: {e}"
                )

    logger.info(f"Successfully loaded {len(parsers)} scenario parsers")

    return parsers


# Legacy compatibility - alias for old class name
FarsiteParser = FARSITEParser
