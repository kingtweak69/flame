"""
Cosine-with-warmup LR scheduler for flame.

Replaces torchtitan.components.lr_scheduler.build_lr_schedulers.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = ["build_lr_scheduler"]


def build_lr_scheduler(optimizer: Optimizer, job_config) -> LambdaLR:
    """
    Build a one-cycle cosine LR schedule with optional warmup.

    Schedule phases (all measured in optimizer steps):
    1. Linear warmup:  0 → ``warmup_steps``
    2. Cosine decay:   ``warmup_steps`` → ``decay_end`` (= ``decay_ratio * steps``)
    3. Constant hold:  ``decay_end`` → ``steps``  at ``lr_min`` * base_lr

    Args:
        optimizer:  The optimizer whose param-group lr values will be scaled.
        job_config: A config object with ``training.steps``,
                    ``lr_scheduler.warmup_steps``, ``lr_scheduler.lr_min``,
                    ``lr_scheduler.decay_ratio``, and
                    ``lr_scheduler.decay_type``.

    Returns:
        A :class:`~torch.optim.lr_scheduler.LambdaLR` instance.
    """
    sched = job_config.lr_scheduler
    train = job_config.training

    warmup_steps: int = sched.warmup_steps
    lr_min_ratio: float = sched.lr_min       # fraction of peak lr
    decay_ratio = sched.decay_ratio          # fraction of total steps at which decay ends
    decay_type: str = sched.decay_type       # "cosine" | "linear"
    total_steps: int = train.steps

    # If decay_ratio is None (WSD schedule default), decay starts right after warmup
    if decay_ratio is None:
        decay_ratio = 1.0
    decay_end = int(decay_ratio * total_steps)

    def lr_lambda(current_step: int) -> float:
        # 1. Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)

        # 2. After decay end — hold at lr_min
        if current_step >= decay_end:
            return lr_min_ratio

        # 3. Decay phase
        progress = float(current_step - warmup_steps) / max(
            1, decay_end - warmup_steps
        )
        if decay_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif decay_type == "linear":
            decay = 1.0 - progress
        else:
            raise ValueError(f"Unknown decay_type: '{decay_type}'")

        return lr_min_ratio + (1.0 - lr_min_ratio) * decay

    return LambdaLR(optimizer, lr_lambda)
