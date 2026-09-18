"""Shared terminal progress-bar settings."""

from __future__ import annotations

import shutil


MAX_PROGRESS_COLUMNS = 120


def progress_ncols(max_columns=MAX_PROGRESS_COLUMNS):
    """Use the terminal width without ever exceeding ``max_columns``."""
    terminal_columns = shutil.get_terminal_size(fallback=(max_columns, 24)).columns
    return min(terminal_columns, max_columns)
