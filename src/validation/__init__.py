"""Validation package for hyperparameter search.

Exports experiment runner functions for both vanilla AE and adversarial AAE,
plus analysis and search space utilities.
"""
from src.validation.run_experiment_ae import train_and_evaluate_ae
from src.validation.run_experiment_aae import train_and_evaluate_aae
from src.validation.search_space import (
    get_search_space_ae,
    get_search_space_aae,
    merge_search_sample,
)
from src.validation.analyze_results import run_analysis

__all__ = [
    "train_and_evaluate_ae",
    "train_and_evaluate_aae",
    "get_search_space_ae",
    "get_search_space_aae",
    "merge_search_sample",
    "run_analysis",
]
