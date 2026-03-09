"""Loguru-based logging setup for crowd counting experiments."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_dir: str | Path, log_file: str = "train.log", level: str = "INFO"
) -> None:
    """Configure loguru to write to stdout and a rotating file.

    Args:
        log_dir: directory where the log file will be created.
        log_file: filename for the log output.
        level: minimum log level (e.g. "DEBUG", "INFO", "WARNING").
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Stdout handler with colour
    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # File handler (auto-rotating at 50 MB)
    logger.add(
        log_dir / log_file,
        level=level,
        rotation="50 MB",
        retention=5,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    )


__all__ = ["logger", "setup_logger"]
