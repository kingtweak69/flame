# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List


@dataclass
class TrainState:
    """Lightweight training-state container saved alongside model checkpoints."""

    step: int = 0
    skipped_step: int = 0
    token: int = 0
    elapsed: timedelta = timedelta(0)
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "skipped_step": self.skipped_step,
            "token": self.token,
            "elapsed_seconds": self.elapsed.total_seconds(),
            "global_avg_losses": self.global_avg_losses,
            "global_max_losses": self.global_max_losses,
            "log_steps": self.log_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.step = int(state_dict["step"])
        self.skipped_step = int(state_dict.get("skipped_step", 0))
        self.token = int(state_dict["token"])
        self.elapsed = timedelta(seconds=float(state_dict.get("elapsed_seconds", 0)))
        self.global_avg_losses = list(state_dict.get("global_avg_losses", []))
        self.global_max_losses = list(state_dict.get("global_max_losses", []))
        self.log_steps = list(state_dict.get("log_steps", []))
