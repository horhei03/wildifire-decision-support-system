"""
Data preprocessing module for FARSITE outputs
"""

from .farsite_parser import FARSITEParser, parse_scenario_name, load_all_scenarios
from .label_generator import LabelGenerator
from .tensor_builder import TensorBuilder
from .dataset_pipeline import DatasetPipeline
from .audit_simulations import (
    audit_all_simulations,
    generate_summary,
    scan_scenario,
    find_file_with_pattern
)

# Legacy alias
FarsiteParser = FARSITEParser

__all__ = [
    "FARSITEParser",
    "FarsiteParser",  # Legacy
    "parse_scenario_name",
    "load_all_scenarios",
    "LabelGenerator",
    "TensorBuilder",
    "DatasetPipeline",
    "audit_all_simulations",
    "generate_summary",
    "scan_scenario",
    "find_file_with_pattern",
]
