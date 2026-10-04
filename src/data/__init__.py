"""Data handling package for the AM01 project.

Modules
-------
- preprocessing : load, split (temporal), StandardScale, save to data/processed/
- dataset       : KukaDataset — sliding-window torch Dataset + DataLoader factory

Note: ``preprocessing`` submodule is not imported eagerly here to avoid a
``RuntimeWarning`` from ``runpy`` when executing ``python -m src.data.preprocessing``
(which would find ``preprocessing`` already in ``sys.modules``).  Import it
explicitly when needed: ``from src.data.preprocessing import run_preprocessing``.
"""
from src.data.dataset import KukaDataset, build_datasets

__all__ = [
    "KukaDataset",
    "build_datasets",
]
