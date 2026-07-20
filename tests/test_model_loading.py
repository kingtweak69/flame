# -*- coding: utf-8 -*-
"""
Tests verifying that model loading paths in flame work correctly with the
transformers <5.0 constraint.

These tests mock out heavy dependencies (torch, transformers, fla) so they
run in a lightweight CI environment without a GPU.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Stubs: install minimal fakes so flame modules can be imported
# ---------------------------------------------------------------------------

def _install_stubs() -> None:
    """Install lightweight stubs for torch, transformers, and fla."""

    # --- torch ---
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.inference_mode = lambda f=None: (f if f else lambda fn: fn)

        dcp = types.ModuleType("torch.distributed.checkpoint")
        dcp.filesystem = MagicMock()
        dcp.save = MagicMock()
        dcp.load = MagicMock()

        fs_mod = types.ModuleType("torch.distributed.checkpoint.filesystem")
        fs_mod.FileSystemWriter = MagicMock()
        dcp.filesystem = fs_mod

        torch_mod.distributed = types.ModuleType("torch.distributed")
        torch_mod.distributed.checkpoint = dcp
        sys.modules.setdefault("torch", torch_mod)
        sys.modules.setdefault("torch.distributed", torch_mod.distributed)
        sys.modules.setdefault("torch.distributed.checkpoint", dcp)
        sys.modules.setdefault("torch.distributed.checkpoint.filesystem", fs_mod)

    # --- transformers ---
    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        tf.AutoConfig = MagicMock()
        tf.AutoModelForCausalLM = MagicMock()
        tf.AutoTokenizer = MagicMock()
        sys.modules.setdefault("transformers", tf)

    # --- fla ---
    sys.modules.setdefault("fla", types.ModuleType("fla"))


_install_stubs()


# ---------------------------------------------------------------------------
# Tests: convert_hf_weights  (HF → DCP)
# ---------------------------------------------------------------------------

class TestConvertHfWeights:
    """Verifies that convert_hf_weights loads and serialises the model correctly."""

    def _reload_mod(self):
        import importlib
        import flame.utils.convert_hf_to_dcp as mod
        importlib.reload(mod)
        return mod

    def test_loads_model_from_pretrained(self, tmp_path):
        """AutoModelForCausalLM.from_pretrained is called with the supplied model path."""
        mod = self._reload_mod()

        fake_model = MagicMock()
        fake_model.state_dict.return_value = {"weight": MagicMock()}
        tf = sys.modules["transformers"]
        tf.AutoModelForCausalLM.from_pretrained.return_value = fake_model

        sys.modules["torch"].distributed.checkpoint.save = MagicMock()

        mod.convert_hf_weights("org/my-model", tmp_path)

        tf.AutoModelForCausalLM.from_pretrained.assert_called_once_with("org/my-model")

    def test_state_dict_written_to_checkpoint(self, tmp_path):
        """The model's state_dict is passed to DCP.save under the 'model' key."""
        mod = self._reload_mod()

        fake_state = {"transformer.h.0.weight": MagicMock()}
        fake_model = MagicMock()
        fake_model.state_dict.return_value = fake_state
        tf = sys.modules["transformers"]
        tf.AutoModelForCausalLM.from_pretrained.return_value = fake_model

        mock_save = MagicMock()
        sys.modules["torch"].distributed.checkpoint.save = mock_save

        mod.convert_hf_weights("org/my-model", tmp_path)

        mock_save.assert_called_once()
        saved_payload = mock_save.call_args[0][0]
        assert saved_payload == {"model": fake_state}

    def test_checkpoint_directory_created(self, tmp_path):
        """convert_hf_weights creates the checkpoint directory if it does not exist."""
        mod = self._reload_mod()

        checkpoint_dir = tmp_path / "new_checkpoint"
        assert not checkpoint_dir.exists()

        fake_model = MagicMock()
        fake_model.state_dict.return_value = {}
        tf = sys.modules["transformers"]
        tf.AutoModelForCausalLM.from_pretrained.return_value = fake_model
        sys.modules["torch"].distributed.checkpoint.save = MagicMock()

        mod.convert_hf_weights("org/my-model", checkpoint_dir)

        assert checkpoint_dir.exists()

    def test_state_dict_called_on_model(self, tmp_path):
        """state_dict() is called exactly once to serialise the model weights."""
        mod = self._reload_mod()

        fake_model = MagicMock()
        fake_model.state_dict.return_value = {}
        tf = sys.modules["transformers"]
        tf.AutoModelForCausalLM.from_pretrained.return_value = fake_model
        sys.modules["torch"].distributed.checkpoint.save = MagicMock()

        mod.convert_hf_weights("org/my-model", tmp_path)

        fake_model.state_dict.assert_called_once()
