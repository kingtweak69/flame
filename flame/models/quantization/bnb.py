# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
BitsAndBytes quantization converter for training.

Provides a BitsAndBytesConverter that replaces nn.Linear layers with
bitsandbytes quantized linear layers to reduce memory during training.

Supported quantization types:
    - "int8": LLM.int8() 8-bit quantization (Linear8bitLt)
    - "fp4":  FP4 4-bit quantization (LinearFP4)
    - "nf4":  NF4 4-bit quantization from QLoRA (LinearNF4)

Usage:
    # In your training config:
    # [model]
    # converters = ["bnb"]
    #
    # [bnb]
    # quant_type = "int8"      # or "fp4" / "nf4"
    # filter_fqns = []         # Optional: FQN substrings to skip
"""

from typing import List, Type, Union

import torch.nn as nn

from flame.logging import logger
from flame.models.converter import ModelConverter, register_model_converter

VALID_QUANT_TYPES = ("int8", "fp4", "nf4")


class BitsAndBytesConverter(ModelConverter):
    """Converts nn.Linear layers to bitsandbytes quantized linear layers.

    Supports INT8, FP4, and NF4 quantization using the bitsandbytes library.
    Quantized layers reduce GPU memory usage during training by storing
    weights in a compressed format.

    Requires bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes

    Config section: [bnb]
        quant_type (str): Quantization type — "int8", "fp4", or "nf4". Default: "int8".
        filter_fqns (list[str]): FQN substrings of modules to skip. Default: [].
    """

    def __init__(self, job_config):
        self.enabled = False

        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is not installed. "
                "Please install it: pip install bitsandbytes"
            ) from e

        bnb_config = getattr(job_config, "bnb", None)
        quant_type: str = (
            getattr(bnb_config, "quant_type", "int8") if bnb_config is not None else "int8"
        )
        if quant_type not in VALID_QUANT_TYPES:
            raise ValueError(
                f"Invalid bnb.quant_type '{quant_type}'. "
                f"Valid options: {VALID_QUANT_TYPES}"
            )
        self.quant_type = quant_type
        self.filter_fqns: List[str] = (
            getattr(bnb_config, "filter_fqns", []) if bnb_config is not None else []
        )

        self.enabled = True
        logger.info(f"BitsAndBytes {quant_type} quantization converter initialized")

    def _get_linear_cls(self) -> Type[nn.Linear]:
        """Return the bitsandbytes linear class for the configured quant_type."""
        import bitsandbytes as bnb

        if self.quant_type == "int8":
            return bnb.nn.Linear8bitLt
        elif self.quant_type == "fp4":
            return bnb.nn.LinearFP4
        else:  # nf4
            return bnb.nn.LinearNF4

    def _replace_linear_layers(
        self, model: nn.Module, linear_cls: Type[nn.Linear], prefix: str = ""
    ) -> None:
        """Recursively replace nn.Linear modules with quantized linear layers."""
        for name, module in list(model.named_children()):
            fqn = f"{prefix}.{name}" if prefix else name
            if isinstance(module, nn.Linear) and not any(
                skip_fqn in fqn for skip_fqn in self.filter_fqns
            ):
                new_module = linear_cls(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                )
                setattr(model, name, new_module)
            else:
                self._replace_linear_layers(module, linear_cls, prefix=fqn)

    def convert(self, model: nn.Module) -> None:
        """Replaces nn.Linear layers with bitsandbytes quantized linear layers."""
        if not self.enabled:
            return

        linear_cls = self._get_linear_cls()
        self._replace_linear_layers(model, linear_cls)
        logger.info(f"Swapped to BitsAndBytes {self.quant_type} layers")

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        """BitsAndBytes does not require any post-optimizer hooks."""
        return


register_model_converter(BitsAndBytesConverter, "bnb")
