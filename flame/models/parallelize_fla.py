# This file applies optional training techniques (activation checkpointing and
# torch.compile) to FLA models.  Distributed data parallelism is handled by
# Accelerate / DeepSpeed — no FSDP2 / TP / DDP wrappers here.

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper

from flame.logging import logger


# ---------------------------------------------------------------------------
# Activation checkpointing
# ---------------------------------------------------------------------------

# ops whose outputs are always saved (never recomputed) in selective AC
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
}


def _apply_ac_to_block(module: nn.Module, ac_config):
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.mode == "selective"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )

    if use_op_sac:
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                to_save = func in _save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )
            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )

    # use_layer_sac
    ac_freq = int(ac_config.selective_ac_option)
    ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
    ptd_checkpoint_wrapper._count += 1
    if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
    return module


def apply_ac(model: nn.Module, ac_config) -> None:
    """Apply activation checkpointing to each transformer block."""
    blocks = get_blocks(model)
    if blocks is None:
        logger.warning("No block found for activation checkpointing")
        return
    for layer_id, block in blocks.named_children():
        block = _apply_ac_to_block(block, ac_config)
        blocks.register_module(layer_id, block)
    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

def apply_compile(model: nn.Module) -> None:
    """Compile each transformer block individually (efficient due to repeated structure)."""
    blocks = get_blocks(model)
    if blocks is None:
        logger.warning("No block found for torch.compile")
    else:
        for layer_id, block in blocks.named_children():
            block = torch.compile(block)
            blocks.register_module(layer_id, block)
        logger.info("Compiled each block with torch.compile")

    real_model = get_model(model)

    logger.info("Compiling the embedding, norm, and lm_head layers with torch.compile")
    embeddings_key = get_components_name(real_model, "tok_embeddings")
    if embeddings_key is not None:
        emb = torch.compile(getattr(real_model, embeddings_key), fullgraph=True)
        real_model.register_module(embeddings_key, emb)

    norm_key = get_components_name(real_model, "norm")
    if norm_key is not None:
        norm = torch.compile(getattr(real_model, norm_key), fullgraph=True)
        real_model.register_module(norm_key, norm)

    lm_head_key = get_components_name(model, "lm_head")
    if lm_head_key is not None:
        lm_head = torch.compile(getattr(model, lm_head_key), fullgraph=True)
        model.register_module(lm_head_key, lm_head)

    logger.info("Compiling the entire model with torch.compile")
    model = torch.compile(model)


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------

def get_model(model: nn.Module):
    base_model_prefix = getattr(model, "base_model_prefix", "model")
    if not hasattr(model, base_model_prefix):
        return None
    return getattr(model, base_model_prefix)


def get_blocks(model: nn.Module):
    real_model = get_model(model)
    if real_model is None or not hasattr(real_model, "layers"):
        logger.warning('No "layers" attribute found in model')
        return None
    return real_model.layers


def get_components_name(model: nn.Module, component_name: str):
    """Return the attribute name for a well-known component, or None."""
    if component_name == "tok_embeddings":
        for name in ("tok_embeddings", "embed_tokens", "embeddings"):
            if hasattr(model, name):
                return name
        logger.warning("No tok_embeddings found in model")
        return None
    elif component_name == "norm":
        for name in ("norm", "norms", "layernorm"):
            if hasattr(model, name):
                return name
        logger.warning("No norm found in model")
        return None
    elif component_name == "lm_head":
        if hasattr(model, "lm_head"):
            return "lm_head"
        logger.warning("No lm_head found in model")
        return None
    raise ValueError(f"Unknown component_name: {component_name}")
