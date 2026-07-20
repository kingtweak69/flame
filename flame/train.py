# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this source tree.

import gc
import importlib
import importlib.util
import json
import os
import sys
import time
from datetime import timedelta

import fla  # noqa
import torch
from accelerate import Accelerator
from accelerate.utils import init_empty_weights, set_seed
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models  # noqa — registers custom FLA model types
from flame.components.checkpoint import TrainState
from flame.components.lr_scheduler import build_lr_scheduler
from flame.config_manager import JobConfig, TORCH_DTYPE_MAP
from flame.data import build_dataloader, build_dataset
from flame.logging import Color, NoColor, init_logger, logger
from flame.models.converter import build_model_converters
from flame.models.parallelize_fla import apply_ac, apply_compile
from flame.tools.utils import get_nparams_and_flops
import flame.models.quantization  # noqa: F401 — registers bnb and nvfp4 converters


# ---------------------------------------------------------------------------
# Mixed-precision mapping: flame dtype string → accelerate mixed_precision str
# ---------------------------------------------------------------------------

_DTYPE_TO_MIXED_PRECISION = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "no",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class GarbageCollection:
    """Periodically runs gc.collect() to avoid GC-induced stragglers."""

    def __init__(self, gc_freq: int = 50) -> None:
        self.gc_freq = gc_freq

    def run(self, step: int) -> None:
        if self.gc_freq > 0 and step % self.gc_freq == 0:
            gc.collect()


def _import_module_from_path(path: str) -> None:
    """Dynamically import *path* (filesystem path or dotted module name)."""
    if not path:
        return
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("_custom_module", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        importlib.import_module(path)


def _build_optimizer(model: torch.nn.Module, job_config: JobConfig) -> torch.optim.Optimizer:
    opt_cfg = job_config.optimizer
    name = opt_cfg.name
    impl = opt_cfg.implementation  # "fused" | "foreach" | "for-loop"
    kwargs = dict(
        lr=opt_cfg.lr,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
        weight_decay=opt_cfg.weight_decay,
        eps=opt_cfg.eps,
    )
    if impl == "fused":
        kwargs["fused"] = True
    elif impl == "foreach":
        kwargs["foreach"] = True

    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    elif name == "Adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def _checkpoint_dir(job_config: JobConfig, step: int) -> str:
    return os.path.join(
        job_config.job.dump_folder,
        job_config.checkpoint.folder,
        f"step_{step}",
    )


def _find_latest_checkpoint_step(job_config: JobConfig) -> int:
    """Return the highest step number among existing checkpoint dirs, or -1."""
    ckpt_root = os.path.join(job_config.job.dump_folder, job_config.checkpoint.folder)
    if not os.path.isdir(ckpt_root):
        return -1
    steps = []
    for name in os.listdir(ckpt_root):
        if name.startswith("step_"):
            try:
                steps.append(int(name[len("step_"):]))
            except ValueError:
                pass
    return max(steps) if steps else -1


def _purge_old_checkpoints(job_config: JobConfig, current_step: int) -> None:
    """Delete old checkpoints beyond keep_latest_k."""
    k = job_config.checkpoint.keep_latest_k
    if k <= 0:
        return
    ckpt_root = os.path.join(job_config.job.dump_folder, job_config.checkpoint.folder)
    steps = sorted(
        int(n[len("step_"):])
        for n in os.listdir(ckpt_root)
        if n.startswith("step_") and n[len("step_"):].isdigit()
    )
    to_delete = steps[: max(0, len(steps) - k)]
    for s in to_delete:
        import shutil
        path = _checkpoint_dir(job_config, s)
        shutil.rmtree(path, ignore_errors=True)
        logger.info(f"Removed old checkpoint: {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(job_config: JobConfig) -> None:
    init_logger()

    if job_config.experimental.custom_model_path:
        _import_module_from_path(job_config.experimental.custom_model_path)

    color = NoColor if job_config.metrics.disable_color_printing else Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    gc_handler = GarbageCollection(gc_freq=job_config.training.gc_freq)

    # ------------------------------------------------------------------
    # Set up Accelerate
    # ------------------------------------------------------------------
    mixed_precision = _DTYPE_TO_MIXED_PRECISION.get(
        job_config.training.mixed_precision_param, "bf16"
    )

    log_with = []
    if job_config.metrics.enable_wandb:
        log_with.append("wandb")
    if job_config.metrics.enable_tensorboard:
        log_with.append("tensorboard")

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=job_config.training.gradient_accumulation_steps,
        project_dir=job_config.job.dump_folder,
        log_with=log_with or None,
    )

    # Random seed (before model/data init for reproducibility)
    set_seed(job_config.training.seed)

    dp_rank = accelerator.process_index
    dp_degree = accelerator.num_processes

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        job_config.model.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    if accelerator.is_main_process:
        logger.info(f"{tokenizer}")

    # ------------------------------------------------------------------
    # Dataset & dataloader
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        logger.info(
            f"Loading dataset {job_config.training.dataset}"
            + (f":{job_config.training.dataset_name}" if job_config.training.dataset_name else "")
        )
    dataset = build_dataset(
        dataset=job_config.training.dataset,
        dataset_name=job_config.training.dataset_name,
        dataset_split=job_config.training.dataset_split,
        data_dir=job_config.training.data_dir,
        data_files=job_config.training.data_files,
        data_probs=job_config.training.data_probs,
        streaming=job_config.training.streaming,
        dp_degree=dp_degree,
        num_workers=job_config.training.num_workers,
        seed=job_config.training.seed,
    )

    if accelerator.is_main_process:
        logger.info("Building dataloader...")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        context_len=job_config.training.context_len,
        varlen=job_config.training.varlen,
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    if accelerator.is_main_process:
        logger.info(
            f"Building model\n{color.green}{model_config}{color.reset}"
        )

    # Build model structure on the meta device (no memory allocated yet), then
    # materialize and initialize weights on CPU.  This mirrors the upstream
    # torchtitan approach (`with torch.device("meta"):`) but uses Accelerate's
    # `init_empty_weights()` so the pattern is consistent with the rest of the
    # Accelerate-based training loop.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(model_config)
        # Defer weight initialization: post_init() was already called inside
        # __init__ on meta tensors (a no-op), so we reset the flag here so
        # that the real initialization runs after we materialize the weights.
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))

    # Materialize parameters into CPU RAM, then initialize weights.
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.post_init()

    if (
        getattr(model_config, "fuse_linear_cross_entropy", False)
        and FusedLinearCrossEntropyLoss is not None
    ):
        model.criterion = FusedLinearCrossEntropyLoss(num_chunks=8)

    # Apply model converters (quantization, float8, …)
    model_converters = build_model_converters(job_config)
    model_converters.convert(model)
    if job_config.model.print_after_conversion and accelerator.is_main_process:
        logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # Activation checkpointing
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # torch.compile (applied before accelerator.prepare)
    if job_config.training.compile:
        apply_compile(model)

    # ------------------------------------------------------------------
    # Optimizer & LR scheduler
    # ------------------------------------------------------------------
    optimizer = _build_optimizer(model, job_config)
    lr_scheduler = build_lr_scheduler(optimizer, job_config)

    # ------------------------------------------------------------------
    # Compute parameter count / FLOPs
    # ------------------------------------------------------------------
    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    # ------------------------------------------------------------------
    # Prepare with Accelerate (wraps model in DDP/DeepSpeed/FSDP as
    # configured in the accelerate config / DeepSpeed config)
    # ------------------------------------------------------------------
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    train_state = TrainState()

    if accelerator.is_main_process:
        os.makedirs(
            os.path.join(job_config.job.dump_folder, job_config.checkpoint.folder),
            exist_ok=True,
        )

    resume_step = job_config.checkpoint.load_step
    if resume_step == -1 and job_config.checkpoint.enable_checkpoint:
        resume_step = _find_latest_checkpoint_step(job_config)

    if resume_step > 0:
        ckpt_dir = _checkpoint_dir(job_config, resume_step)
        if os.path.isdir(ckpt_dir):
            accelerator.load_state(ckpt_dir)
            # load train_state separately
            ts_path = os.path.join(ckpt_dir, "train_state.json")
            if os.path.isfile(ts_path):
                with open(ts_path) as f:
                    train_state.load_state_dict(json.load(f))
            logger.info(f"Resumed from checkpoint: {ckpt_dir} (step {train_state.step})")
        else:
            logger.warning(f"Checkpoint dir not found: {ckpt_dir}. Starting from scratch.")
    elif job_config.checkpoint.initial_load_path:
        # Warm-start: load model weights only
        load_path = job_config.checkpoint.initial_load_path
        if os.path.isdir(load_path):
            accelerator.load_state(load_path)
            logger.info(f"Loaded initial checkpoint from {load_path}")
        else:
            logger.warning(f"initial_load_path not found: {load_path}")

    # ------------------------------------------------------------------
    # Trackers (wandb / tensorboard)
    # ------------------------------------------------------------------
    if log_with and accelerator.is_main_process:
        run_name = os.environ.get("WANDB_NAME", job_config.job.description)
        accelerator.init_trackers(
            project_name=os.environ.get("WANDB_PROJECT", "flame"),
            config=job_config.to_dict(),
            init_kwargs={"wandb": {"name": run_name, "id": os.environ.get("WANDB_RUN_ID")}},
        )

    # ------------------------------------------------------------------
    # Tensorboard writer (non-accelerate)
    # ------------------------------------------------------------------
    tb_writer = None
    if job_config.metrics.enable_tensorboard and accelerator.is_main_process:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(job_config.job.dump_folder, job_config.metrics.save_tb_folder)
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)
        except ImportError:
            logger.warning("TensorBoard not installed; skipping tb logging.")

    # ------------------------------------------------------------------
    # Training-loop housekeeping
    # ------------------------------------------------------------------
    grad_accum_steps = job_config.training.gradient_accumulation_steps
    global_batch_size = job_config.training.batch_size * dp_degree * grad_accum_steps
    num_tokens_per_step = global_batch_size * job_config.training.seq_len

    if accelerator.is_main_process:
        logger.info(f"{color.red}***** Running training *****{color.reset}")
        logger.info(f"{color.green}  Start step            = {train_state.step + 1}")
        logger.info(f"{color.green}  Tokens / sequence     = {job_config.training.seq_len:,}")
        logger.info(f"{color.green}  Grad accum steps      = {grad_accum_steps}")
        logger.info(f"{color.green}  Batch size (per rank) = {job_config.training.batch_size:,}")
        logger.info(
            f"{color.green}  Global batch size     = {global_batch_size:,}"
            f" ({num_tokens_per_step:,} tokens)"
        )
        logger.info(
            f"{color.green}  Total steps           = {job_config.training.steps:,}"
            f" ({job_config.training.steps * num_tokens_per_step:,} tokens)"
        )
        logger.info(
            f"{color.green}  Warmup steps          = {job_config.lr_scheduler.warmup_steps:,}"
        )
        logger.info(f"{color.green}  Parameters            = {model_param_count:,}{color.reset}")

    data_iterator = iter(dataloader)
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    while train_state.step < job_config.training.steps:
        train_state.step += 1
        gc_handler.run(train_state.step)

        optimizer.zero_grad()

        losses: list[torch.Tensor] = []
        aux_losses: list[torch.Tensor] = []

        for micro_step in range(grad_accum_steps):
            # Fetch batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            data_loading_times.append(time.perf_counter() - data_load_start)

            input_ids = batch["input_ids"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            ntokens_since_last_log += labels.numel()

            cu_seqlens = (
                batch["cu_seqlens"].to(accelerator.device)
                if "cu_seqlens" in batch
                else None
            )
            if cu_seqlens is not None:
                position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
            else:
                position_ids = (
                    torch.arange(0, input_ids.shape[1], device=accelerator.device)
                    .unsqueeze(0)
                    .expand(input_ids.shape[0], -1)
                    .to(torch.int32)
                )

            # Use accelerator.accumulate to handle no_sync on all but the
            # last micro-step, ensuring gradient all-reduces happen only once
            # per optimizer step regardless of distributed strategy.
            with accelerator.accumulate(model):
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                )
                loss = output.loss / grad_accum_steps
                accelerator.backward(loss)

            losses.append(loss.detach())
            aux_loss = getattr(output, "aux_loss", None)
            if aux_loss is not None:
                aux_losses.append(aux_loss.detach() / grad_accum_steps)

        step_loss = torch.stack(losses).sum()
        step_aux_loss = torch.stack(aux_losses).sum() if aux_losses else None

        # Gradient clipping
        if job_config.training.max_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), job_config.training.max_norm
            )
        else:
            grad_norm = torch.tensor(0.0)

        # Skip NaN/Inf updates
        if job_config.training.skip_nan_inf and (
            grad_norm.isnan() or grad_norm.isinf()
        ):
            logger.warning(
                f"Skipping step {train_state.step} — invalid grad norm: {grad_norm:.4f}"
            )
            optimizer.zero_grad()
            train_state.skipped_step += 1
        else:
            optimizer.step()

        lr_scheduler.step()

        # Post-optimizer hooks (e.g. float8 scale update)
        model_converters.post_optimizer_hook(accelerator.unwrap_model(model))

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if train_state.step % job_config.metrics.log_freq == 0:
            # Gather loss across all ranks
            gathered = accelerator.gather(step_loss.unsqueeze(0))
            global_avg_loss = gathered.mean().item()
            global_max_loss = gathered.max().item()

            time_now = time.perf_counter()
            time_delta = time_now - time_last_log

            train_state.token += (
                ntokens_since_last_log * dp_degree
            )
            train_state.elapsed += timedelta(seconds=time_delta)
            train_state.log_steps.append(train_state.step)
            train_state.global_avg_losses.append(global_avg_loss)
            train_state.global_max_losses.append(global_max_loss)

            last_lr = lr_scheduler.get_last_lr()[0]
            wps = ntokens_since_last_log * dp_degree / max(time_delta, 1e-6)
            eta = (
                train_state.elapsed
                * (job_config.training.steps - train_state.step)
                / train_state.step
            )
            avg_data_time = (
                sum(data_loading_times) / len(data_loading_times)
                if data_loading_times
                else 0.0
            )

            log_dict = {
                "loss/train": global_avg_loss,
                "loss/max": global_max_loss,
                "optimizer/lr": last_lr,
                "optimizer/grad_norm": grad_norm.item(),
                "optimizer/skipped_step": train_state.skipped_step,
                "throughput/tokens_per_sec": wps,
                "throughput/data_loading_ms": avg_data_time * 1000,
            }
            if step_aux_loss is not None:
                gathered_aux = accelerator.gather(step_aux_loss.unsqueeze(0))
                log_dict["loss/aux"] = gathered_aux.mean().item()

            if log_with:
                accelerator.log(log_dict, step=train_state.step)
            if tb_writer is not None:
                for k, v in log_dict.items():
                    tb_writer.add_scalar(k, v, train_state.step)

            if accelerator.is_main_process:
                logger.info(
                    f"step={train_state.step:6d} "
                    f"{color.blue}loss={global_avg_loss:.4f} "
                    f"lr={last_lr:.4e} gnorm={grad_norm:.3f} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}"
                    f"<{str(eta).split('.')[0]:>8}] "
                    f"{color.reset}wps={wps:,.0f}"
                )

            # reset per-log counters
            ntokens_since_last_log = 0
            data_loading_times.clear()
            time_last_log = time_now

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        is_final_step = train_state.step == job_config.training.steps
        should_checkpoint = (
            job_config.checkpoint.enable_checkpoint
            and (
                train_state.step % job_config.checkpoint.interval == 0
                or is_final_step
            )
        )
        if should_checkpoint:
            ckpt_dir = _checkpoint_dir(job_config, train_state.step)
            if accelerator.is_main_process:
                logger.info(f"Saving checkpoint to {ckpt_dir}")
            accelerator.save_state(ckpt_dir)
            if accelerator.is_main_process:
                ts_path = os.path.join(ckpt_dir, "train_state.json")
                with open(ts_path, "w") as f:
                    json.dump(train_state.state_dict(), f)
                _purge_old_checkpoints(job_config, train_state.step)

            # Final checkpoint: optionally save HF model weights
            if is_final_step:
                if job_config.checkpoint.last_save_model_weights_only:
                    hf_dir = os.path.join(ckpt_dir, "hf_model")
                    if accelerator.is_main_process:
                        logger.info(f"Saving HF model weights to {hf_dir}")
                    unwrapped = accelerator.unwrap_model(model)
                    export_dtype = TORCH_DTYPE_MAP.get(
                        job_config.checkpoint.export_dtype, torch.float32
                    )
                    unwrapped.to(export_dtype)
                    if accelerator.is_main_process:
                        unwrapped.save_pretrained(hf_dir)
                        tokenizer.save_pretrained(hf_dir)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if tb_writer is not None:
        tb_writer.close()
    if log_with:
        accelerator.end_training()
    if accelerator.is_main_process:
        logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.parse_args()
    main(config)
