#!/usr/bin/env python3
"""Plot V-JEPA training loss from all rank CSV logs in a directory."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Keep the script directory out of import resolution so a future local module
# cannot shadow a standard-library dependency imported by matplotlib.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != Path(_SCRIPT_DIR)]
PLOT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PLOT_ROOT / "output" / "train"


def read_losses(logdir: Path) -> dict[tuple[int, int], list[float]]:
    losses: dict[tuple[int, int], list[float]] = defaultdict(list)

    for csv_path in sorted(logdir.rglob("*.csv")):
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                lines = handle.readlines()
                header_indices = [
                    index
                    for index, line in enumerate(lines)
                    if line.startswith("epoch,itr,loss,")
                ]
                if not header_indices:
                    continue

                # CSVLogger appends a new header when a training run is
                # restarted. Keep only the most recent run in each rank log.
                reader = csv.DictReader(lines[header_indices[-1] :])
                fieldnames = set(reader.fieldnames or ())
                if not {"epoch", "itr", "loss"}.issubset(fieldnames):
                    continue

                for row in reader:
                    try:
                        epoch = int(row["epoch"])
                        iteration = int(row["itr"])
                        loss = float(row["loss"])
                    except (KeyError, TypeError, ValueError):
                        # Ignore repeated headers or incomplete rows.
                        continue
                    losses[(epoch, iteration)].append(loss)
        except (OSError, UnicodeDecodeError) as error:
            print(f"warning: skip {csv_path}: {error}")

    return losses


def moving_average(values: list[float], window: int) -> list[float]:
    result = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG path (default: PLOT/output/train/<logdir>/train_loss.png)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="moving-average window in logged iterations (default: 20)",
    )
    args = parser.parse_args()

    if not args.logdir.is_dir():
        parser.error(f"log directory does not exist: {args.logdir}")
    if args.window < 1:
        parser.error("--window must be at least 1")

    losses = read_losses(args.logdir)
    if not losses:
        parser.error(f"no training rows found under {args.logdir}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required; install it with `pip install matplotlib`"
        ) from error

    points = sorted(
        (epoch, iteration, sum(values) / len(values))
        for (epoch, iteration), values in losses.items()
    )
    y_values = [point[2] for point in points]
    x_values = list(range(len(points)))
    smoothed = moving_average(y_values, args.window)

    output = args.out or DEFAULT_OUTPUT_ROOT / args.logdir.resolve().name / "train_loss.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6), dpi=150)
    axis.plot(x_values, y_values, color="#9aa5b1", linewidth=0.8, alpha=0.45, label="loss")
    axis.plot(
        x_values,
        smoothed,
        color="#d1495b",
        linewidth=2.0,
        label=f"moving average ({args.window})",
    )
    axis.set_title("V-JEPA Training Loss")
    axis.set_xlabel("Logged iteration")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)

    rank_count = max(len(values) for values in losses.values())
    print(f"read {len(points)} averaged points from {args.logdir}")
    print(f"maximum rank samples per point: {rank_count}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
