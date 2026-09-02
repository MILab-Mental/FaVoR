#!/usr/bin/env python3
"""Recursively remove Python and Jupyter cache directories from this project."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CACHE_DIRECTORIES = {"__pycache__", ".ipynb_checkpoints"}


def clean(root: Path) -> int:
    """Delete cache directories below *root* and return their count."""
    removed = 0

    # ``rglob`` does not follow directory symlinks. Sorting deepest-first keeps
    # traversal robust should one cache directory appear inside another.
    targets = sorted(
        (
            path
            for path in root.rglob("*")
            if path.name in CACHE_DIRECTORIES and path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )

    for path in targets:
        shutil.rmtree(path)
        print(f"Removed: {path.relative_to(root)}")
        removed += 1

    print(f"Done. Removed {removed} cache director{'y' if removed == 1 else 'ies'}.")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively remove __pycache__ and .ipynb_checkpoints directories."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to clean (default: directory containing this script).",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    clean(root)


if __name__ == "__main__":
    main()
