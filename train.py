"""Backward-compatible exports for training and evaluation helpers."""

from hano.trainer import (
    test_data,
    test_model,
    test_ns,
    train_NS_model,
    train_data,
    train_model,
    train_ns,
)

__all__ = [
    "train_data",
    "test_data",
    "train_ns",
    "test_ns",
    "train_model",
    "train_NS_model",
    "test_model",
]
