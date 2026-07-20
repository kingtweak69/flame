"""
Lightweight ModelConverter protocol and registry for flame.

Replaces torchtitan.protocols.model_converter so that quantization and other
model-conversion plugins have no torchtitan dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import torch.nn as nn

__all__ = [
    "ModelConverter",
    "register_model_converter",
    "build_model_converters",
    "ModelConverterGroup",
]

_CONVERTER_REGISTRY: Dict[str, Type[ModelConverter]] = {}


class ModelConverter(ABC):
    """Base class for model converters (e.g. quantization, float8)."""

    @abstractmethod
    def convert(self, model: nn.Module) -> None:
        """Apply the conversion to *model* in-place (called before training)."""

    def post_optimizer_hook(self, model: nn.Module) -> None:
        """Optional hook called after every optimizer step."""


class ModelConverterGroup:
    """Holds zero or more ModelConverter instances and dispatches to all of them."""

    def __init__(self, converters: List[ModelConverter]) -> None:
        self._converters = converters

    def convert(self, model: nn.Module) -> None:
        for c in self._converters:
            c.convert(model)

    def post_optimizer_hook(self, model: nn.Module) -> None:
        for c in self._converters:
            c.post_optimizer_hook(model)

    def __bool__(self) -> bool:  # truthy if any converters present
        return bool(self._converters)


def register_model_converter(cls: Type[ModelConverter], name: str) -> None:
    """Register a ModelConverter subclass under *name*."""
    _CONVERTER_REGISTRY[name] = cls


def build_model_converters(job_config) -> ModelConverterGroup:
    """Instantiate and return all converters listed in ``job_config.model.converters``."""
    names: List[str] = getattr(getattr(job_config, "model", None), "converters", []) or []
    converters: List[ModelConverter] = []
    for name in names:
        if name not in _CONVERTER_REGISTRY:
            raise ValueError(
                f"Unknown model converter '{name}'. "
                f"Available: {sorted(_CONVERTER_REGISTRY)}"
            )
        converters.append(_CONVERTER_REGISTRY[name](job_config))
    return ModelConverterGroup(converters)
