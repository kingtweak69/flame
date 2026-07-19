# -*- coding: utf-8 -*-
"""
Tests for the NVFP4 and BitsAndBytes quantization converters.

These tests focus on converter initialization, configuration validation,
and module replacement logic without requiring a real GPU or the full
training stack.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stubs: torchtitan is not available in the lightweight test environment
# ---------------------------------------------------------------------------

def _install_torchtitan_stubs() -> None:
    """Install minimal torchtitan stubs needed by the quantization converters."""
    try:
        import torchtitan.protocols.model_converter  # noqa: F401
        return
    except Exception:
        pass

    # torchtitan top-level
    tt = sys.modules.setdefault("torchtitan", types.ModuleType("torchtitan"))

    # torchtitan.tools.logging
    tt_tools = sys.modules.setdefault("torchtitan.tools", types.ModuleType("torchtitan.tools"))
    tt_tools_logging = sys.modules.setdefault(
        "torchtitan.tools.logging", types.ModuleType("torchtitan.tools.logging")
    )
    import logging
    tt_tools_logging.logger = logging.getLogger("flame.test")
    tt_tools.logging = tt_tools_logging
    tt.tools = tt_tools

    # torchtitan.config_manager
    tt_cfg = sys.modules.setdefault(
        "torchtitan.config_manager", types.ModuleType("torchtitan.config_manager")
    )
    tt_cfg.JobConfig = object
    tt.config_manager = tt_cfg

    # torchtitan.distributed
    tt_dist = sys.modules.setdefault(
        "torchtitan.distributed", types.ModuleType("torchtitan.distributed")
    )
    tt_dist.ParallelDims = object
    tt.distributed = tt_dist

    # torchtitan.protocols
    tt_proto = sys.modules.setdefault(
        "torchtitan.protocols", types.ModuleType("torchtitan.protocols")
    )
    tt.protocols = tt_proto

    # torchtitan.protocols.model_converter
    _registry: dict = {}

    class _ModelConverter:
        pass

    def _register_model_converter(cls, name):
        _registry[name] = cls

    tt_mc = sys.modules.setdefault(
        "torchtitan.protocols.model_converter",
        types.ModuleType("torchtitan.protocols.model_converter"),
    )
    tt_mc.ModelConverter = _ModelConverter
    tt_mc.register_model_converter = _register_model_converter
    tt_mc._registry = _registry
    tt_proto.model_converter = tt_mc


_install_torchtitan_stubs()

# Now import the converters under test (after stubs are in place)
from flame.models.quantization.bnb import BitsAndBytesConverter, VALID_QUANT_TYPES  # noqa: E402
from flame.models.quantization.nvfp4 import NvFp4Converter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job_config(**sections):
    """Return a minimal fake JobConfig with the given section attributes."""
    cfg = MagicMock()
    for section_name, attrs in sections.items():
        section = MagicMock()
        for k, v in attrs.items():
            setattr(section, k, v)
        setattr(cfg, section_name, section)
    return cfg


def _make_tiny_model() -> nn.Module:
    """Return a small MLP with Linear layers for testing replacement."""

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32, bias=True)
            self.fc2 = nn.Linear(32, 16, bias=False)
            self.lm_head = nn.Linear(16, 8, bias=False)

        def forward(self, x):
            return self.lm_head(self.fc2(self.fc1(x)))

    return TinyMLP()


# ---------------------------------------------------------------------------
# NVFP4 converter tests
# ---------------------------------------------------------------------------

class TestNvFp4ConverterInit:

    def test_raises_when_torchao_missing(self):
        """NvFp4Converter should raise ImportError when torchao is absent."""
        cfg = _make_job_config()
        with patch.dict(sys.modules, {"torchao": None,
                                       "torchao.prototype": None,
                                       "torchao.prototype.moe_training": None,
                                       "torchao.prototype.moe_training.nvfp4_training": None,
                                       "torchao.prototype.moe_training.nvfp4_training.nvfp4_training": None}):
            with pytest.raises(ImportError, match="torchao"):
                NvFp4Converter(cfg, parallel_dims=None)

    def test_reads_filter_fqns_from_config(self):
        """filter_fqns are read from job_config.nvfp4.filter_fqns."""
        cfg = _make_job_config(nvfp4={"filter_fqns": ["lm_head", "embed"]})

        mock_config_cls = MagicMock()
        mock_config_cls.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "torchao": MagicMock(),
                "torchao.prototype": MagicMock(),
                "torchao.prototype.moe_training": MagicMock(),
                "torchao.prototype.moe_training.nvfp4_training": MagicMock(),
                "torchao.prototype.moe_training.nvfp4_training.nvfp4_training": MagicMock(
                    NVFP4TrainingConfig=mock_config_cls
                ),
            },
        ):
            conv = NvFp4Converter(cfg, parallel_dims=None)

        assert conv.filter_fqns == ["lm_head", "embed"]
        assert conv.enabled is True

    def test_defaults_to_empty_filter_fqns_when_no_nvfp4_section(self):
        """filter_fqns default to [] when job_config has no nvfp4 section."""
        cfg = MagicMock(spec=[])  # no attributes

        mock_config_cls = MagicMock()
        mock_config_cls.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "torchao": MagicMock(),
                "torchao.prototype": MagicMock(),
                "torchao.prototype.moe_training": MagicMock(),
                "torchao.prototype.moe_training.nvfp4_training": MagicMock(),
                "torchao.prototype.moe_training.nvfp4_training.nvfp4_training": MagicMock(
                    NVFP4TrainingConfig=mock_config_cls
                ),
            },
        ):
            conv = NvFp4Converter(cfg, parallel_dims=None)

        assert conv.filter_fqns == []


class TestNvFp4ConverterConvert:

    def _make_converter(self, filter_fqns=None):
        cfg = _make_job_config(nvfp4={"filter_fqns": filter_fqns or []})
        mock_config_cls = MagicMock()
        mock_config_cls.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "torchao": MagicMock(),
                "torchao.prototype": MagicMock(),
                "torchao.prototype.moe_training": MagicMock(),
                "torchao.prototype.moe_training.nvfp4_training": MagicMock(),
                "torchao.prototype.moe_training.nvfp4_training.nvfp4_training": MagicMock(
                    NVFP4TrainingConfig=mock_config_cls
                ),
            },
        ):
            return NvFp4Converter(cfg, parallel_dims=None)

    def test_convert_calls_quantize(self):
        """convert() should call torchao.quantization.quantize_ on the model."""
        conv = self._make_converter()
        model = _make_tiny_model()
        mock_quantize = MagicMock()
        mock_torchao_quant = MagicMock(quantize_=mock_quantize)

        with patch.dict(sys.modules, {"torchao.quantization": mock_torchao_quant}):
            conv.convert(model)

        mock_quantize.assert_called_once()
        call_kwargs = mock_quantize.call_args
        assert call_kwargs[0][0] is model  # first positional arg is the model

    def test_convert_is_noop_when_disabled(self):
        """convert() should be a no-op when enabled=False."""
        conv = self._make_converter()
        conv.enabled = False
        model = _make_tiny_model()
        mock_quantize = MagicMock()

        with patch.dict(sys.modules, {"torchao.quantization": MagicMock(quantize_=mock_quantize)}):
            conv.convert(model)

        mock_quantize.assert_not_called()

    def test_post_optimizer_hook_is_noop(self):
        """post_optimizer_hook should be a no-op (returns None)."""
        conv = self._make_converter()
        assert conv.post_optimizer_hook(_make_tiny_model()) is None


# ---------------------------------------------------------------------------
# BitsAndBytes converter tests
# ---------------------------------------------------------------------------

class TestBitsAndBytesConverterInit:

    def test_raises_when_bitsandbytes_missing(self):
        """BitsAndBytesConverter should raise ImportError when bnb is absent."""
        cfg = _make_job_config()
        with patch.dict(sys.modules, {"bitsandbytes": None}):
            with pytest.raises(ImportError, match="bitsandbytes"):
                BitsAndBytesConverter(cfg, parallel_dims=None)

    @pytest.mark.parametrize("quant_type", VALID_QUANT_TYPES)
    def test_accepts_valid_quant_types(self, quant_type):
        """All valid quant_type values should be accepted without error."""
        cfg = _make_job_config(bnb={"quant_type": quant_type, "filter_fqns": []})
        mock_bnb = MagicMock()
        with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
            conv = BitsAndBytesConverter(cfg, parallel_dims=None)
        assert conv.quant_type == quant_type
        assert conv.enabled is True

    def test_raises_on_invalid_quant_type(self):
        """An unrecognised quant_type should raise ValueError."""
        cfg = _make_job_config(bnb={"quant_type": "bogus", "filter_fqns": []})
        mock_bnb = MagicMock()
        with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
            with pytest.raises(ValueError, match="bogus"):
                BitsAndBytesConverter(cfg, parallel_dims=None)

    def test_default_quant_type_is_int8(self):
        """quant_type defaults to 'int8' when no bnb config section is present."""
        cfg = MagicMock(spec=[])  # no attributes
        mock_bnb = MagicMock()
        with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
            conv = BitsAndBytesConverter(cfg, parallel_dims=None)
        assert conv.quant_type == "int8"

    def test_reads_filter_fqns_from_config(self):
        """filter_fqns are read from job_config.bnb.filter_fqns."""
        cfg = _make_job_config(bnb={"quant_type": "int8", "filter_fqns": ["lm_head"]})
        mock_bnb = MagicMock()
        with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
            conv = BitsAndBytesConverter(cfg, parallel_dims=None)
        assert conv.filter_fqns == ["lm_head"]


class TestBitsAndBytesConverterConvert:

    def _make_mock_bnb(self):
        """Return a mock bitsandbytes module with fake linear classes."""
        mock_bnb = MagicMock()

        class FakeLinear8bitLt(nn.Linear):
            def __init__(self, in_f, out_f, bias=True, device=None, **kwargs):
                super().__init__(in_f, out_f, bias=bias, device=device)

        class FakeLinearFP4(nn.Linear):
            def __init__(self, in_f, out_f, bias=True, device=None, **kwargs):
                super().__init__(in_f, out_f, bias=bias, device=device)

        class FakeLinearNF4(nn.Linear):
            def __init__(self, in_f, out_f, bias=True, device=None, **kwargs):
                super().__init__(in_f, out_f, bias=bias, device=device)

        mock_bnb.nn.Linear8bitLt = FakeLinear8bitLt
        mock_bnb.nn.LinearFP4 = FakeLinearFP4
        mock_bnb.nn.LinearNF4 = FakeLinearNF4
        return mock_bnb

    def _make_converter(self, quant_type="int8", filter_fqns=None):
        cfg = _make_job_config(bnb={"quant_type": quant_type, "filter_fqns": filter_fqns or []})
        mock_bnb = self._make_mock_bnb()
        with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
            conv = BitsAndBytesConverter(cfg, parallel_dims=None)
        # Attach the mock so convert() can use the same classes
        conv._mock_bnb = mock_bnb
        return conv

    @pytest.mark.parametrize("quant_type,attr", [
        ("int8", "Linear8bitLt"),
        ("fp4", "LinearFP4"),
        ("nf4", "LinearNF4"),
    ])
    def test_all_linears_replaced(self, quant_type, attr):
        """All nn.Linear layers (except filtered) should be replaced."""
        conv = self._make_converter(quant_type=quant_type)
        model = _make_tiny_model()
        expected_cls = getattr(conv._mock_bnb.nn, attr)

        with patch.dict(sys.modules, {"bitsandbytes": conv._mock_bnb}):
            conv.convert(model)

        assert isinstance(model.fc1, expected_cls)
        assert isinstance(model.fc2, expected_cls)
        assert isinstance(model.lm_head, expected_cls)

    def test_filter_fqns_skip_matching_modules(self):
        """Modules whose FQN contains a filter string should not be replaced."""
        conv = self._make_converter(quant_type="int8", filter_fqns=["lm_head"])
        model = _make_tiny_model()

        with patch.dict(sys.modules, {"bitsandbytes": conv._mock_bnb}):
            conv.convert(model)

        # fc1 and fc2 should be replaced
        assert isinstance(model.fc1, conv._mock_bnb.nn.Linear8bitLt)
        assert isinstance(model.fc2, conv._mock_bnb.nn.Linear8bitLt)
        # lm_head should remain a plain nn.Linear
        assert type(model.lm_head) is nn.Linear

    def test_bias_preserved(self):
        """Bias presence should be preserved in the replacement layer."""
        conv = self._make_converter(quant_type="int8")
        model = _make_tiny_model()

        with patch.dict(sys.modules, {"bitsandbytes": conv._mock_bnb}):
            conv.convert(model)

        assert model.fc1.bias is not None   # original had bias=True
        assert model.fc2.bias is None       # original had bias=False

    def test_convert_is_noop_when_disabled(self):
        """convert() should be a no-op when enabled=False."""
        conv = self._make_converter()
        conv.enabled = False
        model = _make_tiny_model()
        original_fc1 = model.fc1

        with patch.dict(sys.modules, {"bitsandbytes": conv._mock_bnb}):
            conv.convert(model)

        assert model.fc1 is original_fc1  # unchanged

    def test_post_optimizer_hook_is_noop(self):
        """post_optimizer_hook should be a no-op (returns None)."""
        conv = self._make_converter()
        assert conv.post_optimizer_hook(_make_tiny_model()) is None
