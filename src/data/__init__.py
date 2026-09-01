"""Data handling package for the AM01 project.

Modules
-------
- preprocessing : load, split (temporal), StandardScale, save to data/processed/
- dataset      : KukaDataset — sliding-window torch Dataset + DataLoader factory
"""
from src.data.preprocessing import (
    load_kuka_data,
    normalize_data,
    run_preprocessing,
    save_processed_data,
    split_temporal_data,
)
from src.data.dataset import KukaDataset, build_datasets

__all__ = [
    "load_kuka_data",
    "normalize_data",
    "run_preprocessing",
    "save_processed_data",
    "split_temporal_data",
    "KukaDataset",
    "build_datasets",
]
