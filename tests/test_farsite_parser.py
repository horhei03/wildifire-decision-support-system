"""
Unit tests for FARSITE parser
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.farsite_parser import FarsiteParser


class TestFarsiteParser:
    """Test cases for FarsiteParser."""

    @pytest.fixture
    def parser(self, tmp_path):
        """Create a FarsiteParser instance."""
        return FarsiteParser(str(tmp_path))

    def test_initialization(self, parser, tmp_path):
        """Test parser initialization."""
        assert parser.data_root == tmp_path

    def test_get_available_simulations(self, parser, tmp_path):
        """Test getting available simulations."""
        # Create dummy directories
        (tmp_path / "P1_ws12_wd90_dry").mkdir()
        (tmp_path / "P2_ws25_wd270_extreme").mkdir()

        simulations = parser.get_available_simulations()

        assert len(simulations) == 2
        assert ("P1", "ws12_wd90_dry") in simulations
        assert ("P2", "ws25_wd270_extreme") in simulations

    def test_parse_raster_not_found(self, parser, tmp_path):
        """Test parsing non-existent raster raises error."""
        fake_path = tmp_path / "nonexistent.tif"

        with pytest.raises(Exception):
            parser.parse_raster(fake_path)
