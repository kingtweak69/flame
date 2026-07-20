# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 quantized training converter.

Provides an NvFp4Converter that replaces nn.Linear layers with NVFP4Linear
for quantized training using NVIDIA's FP4 format via torchao.

Usage:
    # In your training config:
    # [model]
    # converters = ["nvfp4"]
    #
    # [nvfp4]
    # filter_fqns = []  # Optional: FQN substrings to skip
"""

from functools import partial
from typing import List, Union

import torch.nn as nn

from flame.logging import logger
from flame.models.converter import ModelConverter, register_model_converter


def _module_filter_fn(mod: nn.Module, fqn: str, filter_fqns: List[str]) -> bool:
    """Filter function for NVFP4 quantization.

    Returns True if the module should be converted (i.e., is an nn.Linear
    and not in the skip list).
    """
    if not isinstance(mod, nn.Linear):
        return False
    return not any(skip_fqn in fqn for skip_fqn in filter_fqns)


class NvFp4Converter(ModelConverter):
    """Converts nn.Linear layers to NVFP4Linear for quantized training.

    Uses NVIDIA's FP4 quantization format via torchao's prototype implementation.
    All three training GEMMs (forward and both backward passes) are quantized
    to FP4 using a Randomized Hadamard Transform (RHT) for improved accuracy.

    Requires torchao with NVFP4 support (nightly build recommended).
    See: https://github.com/pytorch/ao

    Config section: [nvfp4]
        filter_fqns (list[str]): FQN substrings of modules to skip. Default: [].
    """

    def __init__(self, job_config):
        self.enabled = False

        try:
            from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
                NVFP4TrainingConfig,
            )
        except ImportError as e:
            raise ImportError(
                "torchao is not installed or does not have NVFP4 training support. "
                "Please install torchao nightly build: https://github.com/pytorch/ao"
            ) from e

        nvfp4_config = getattr(job_config, "nvfp4", None)
        self.filter_fqns: List[str] = (
            getattr(nvfp4_config, "filter_fqns", []) if nvfp4_config is not None else []
        )

        self.config = NVFP4TrainingConfig()
        self.enabled = True
        logger.info("NVFP4 quantized training converter initialized")

    def convert(self, model: nn.Module):
        """Replaces nn.Linear layers with NVFP4Linear for quantized training."""
        if not self.enabled:
            return

        from torchao.quantization import quantize_

        quantize_(
            model,
            config=self.config,
            filter_fn=partial(_module_filter_fn, filter_fqns=self.filter_fqns),
        )
        logger.info("Swapped to NVFP4Linear layers for quantized training")

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """NVFP4 does not require any post-optimizer hooks."""
        return


register_model_converter(NvFp4Converter, "nvfp4")
