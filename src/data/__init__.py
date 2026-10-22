"""Data handling package for the AM01 project.

Modules
-------
- preprocessing : load, plan partitions, MinMaxScale, save to data/processed/
- dataset       : KukaDataset — window-index-based torch Dataset + DataLoader factory
- partitioning  : interleaved chunk-based partitioning with guard bands

Note: ``preprocessing`` submodule is not imported eagerly here to avoid a
``RuntimeWarning`` from ``runpy`` when executing ``python -m src.data.preprocessing``
(which would find ``preprocessing`` already in ``sys.modules``).  Import it
explicitly when needed: ``from src.data.preprocessing import run_preprocessing``.
"""
from src.data.dataset import KukaDataset, build_dataloaders, build_loader

__all__ = [
    "KukaDataset",
    "build_dataloaders",
    "build_loader",
]