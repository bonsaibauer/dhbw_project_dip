"""Simple console logger with percentage progress output."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

Level = Literal["INFO", "WARN", "ERROR", "DEBUG"]
BAR_WIDTH = 30


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _format_prefix(level: Level, progress: float | None) -> str:
    prefix = f"[{_timestamp()}] [{level}]"
    if progress is not None:
        prefix += f" [{progress:6.2f}%]"
    return prefix


def log(level: Level, message: str, progress: float | None = None) -> None:
    """Print a single log line."""

    print(f"{_format_prefix(level, progress)} {message}")


def log_info(message: str, progress: float | None = None) -> None:
    log("INFO", message, progress)


def log_warning(message: str, progress: float | None = None) -> None:
    log("WARN", message, progress)


def log_error(message: str, progress: float | None = None) -> None:
    log("ERROR", message, progress)


def log_debug(message: str, progress: float | None = None) -> None:
    log("DEBUG", message, progress)


def log_progress(stage: str, current: int, total: int) -> None:
    """Render a textual progress bar between 0 and 100%."""

    if total <= 0:
        percent = 100.0
    else:
        percent = min(100.0, max(0.0, (current / total) * 100.0))
    filled = int(BAR_WIDTH * percent / 100)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    log_info(f"{stage}: [{bar}] {current}/{total}", progress=percent)
