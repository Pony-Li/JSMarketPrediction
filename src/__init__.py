"""
Jane Street Forecasting - Core Source Code Package
==================================================
This package provides the core functionality for:
- Data preparation and synthetic data generation.
- GRU-based time-series modeling (offline training and online inference).
- Dataset loaders and utilities for the Kaggle Jane Street competition.

Modules:
--------
- dataset.py: Defines TimeSeriesDataset and TimeSeriesDataModule.
- model_gru.py: Defines GRUNetworkWithConv and Hyperparameters.
- train.py: Script for offline training.
- online_predictor.py: Online inference (JsGruOnlinePredictor).
- inference.py: Entry point for running inference pipeline.
- synthetic_data.py: Generates synthetic test and lags data.
"""

__version__ = "0.1.0"
__author__ = "Junyi Li"

# 自动导入核心组件
from .dataset import TimeSeriesDataset, TimeSeriesDataModule
from .model_gru import GRUNetworkWithConv, Hyperparameters
from .online_predictor import JsGruOnlinePredictor


__all__ = [
    "TimeSeriesDataset",
    "TimeSeriesDataModule",
    "GRUNetworkWithConv",
    "Hyperparameters",
    "JsGruOnlinePredictor"
]


def get_version():
    """Return the current version of the src package."""
    return __version__


def about():
    """Return a detailed description of the package."""
    return (
        f"Jane Street Forecasting Package v{__version__}\n"
        f"Author: {__author__}\n"
        "Modules: dataset, model_gru, online_predictor, train, inference, synthetic_data\n"
        "This package provides GRU-based training and inference for real-time market forecasting."
    )
