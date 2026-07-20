"""
Minimal logging utilities for flame — replaces torchtitan.tools.logging.
"""

import logging
import os
import sys

__all__ = ["logger", "init_logger", "Color", "NoColor"]


class _ColorCodes:
    """ANSI color codes for terminal output."""

    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[0m"


class _NoColor:
    """Null color object — all attributes return an empty string."""

    def __getattr__(self, name: str) -> str:
        return ""


Color = _ColorCodes()
NoColor = _NoColor()


def init_logger(log_level: str = "INFO") -> None:
    """Configure the root logger for flame training."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = (
        f"[rank{rank}:{local_rank}] %(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    else:
        # Replace existing handlers so format is consistent.
        root.handlers.clear()
        root.addHandler(handler)


logger: logging.Logger = logging.getLogger("flame")
