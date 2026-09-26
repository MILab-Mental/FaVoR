#!/usr/bin/env python3
"""Plot training curves from V-JEPA rank logs or a LeJEPA history.csv."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# Keep the script directory out of import resolution so a future local module
# cannot shadow a standard-library dependency imported by matplotlib.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != Path(_SCRIPT_DIR)]
PLOT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PLOT_ROOT / "output" / "train"

LEJEPA_FIELDS = (
    "epoch",
    "iteration",
    "global_step",
    "loss",
    "invariance_loss",
    "sigreg_loss",
    "embedding_std",
    "rankme",
    "lr",
    "grad_norm",
)
AUDIO_LEJEPA_FIELDS = (
    "step",
    "loss",
    "invariance_loss",
    "sigreg_loss",
    "global_embedding_std",
    "local_embedding_std",
    "rankme",
    "lr_pretrained",
    "lr_new",
    "grad_norm",
)


def read_losses(logdir: Path) -> dict[tuple[int, int], list[float]]:
    """Read and group the legacy per-rank V-JEPA CSV logs."""
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
                    if math.isfinite(loss):
                        losses[(epoch, iteration)].append(loss)
        except (OSError, UnicodeDecodeError) as error:
            print(f"warning: skip {csv_path}: {error}", file=sys.stderr)

    return losses


def find_lejepa_history(logdir: Path) -> Path | None:
    """Return history.csv when it has the VIDEO-LeJEPA schema."""
    history_path = logdir / "history.csv"
    if not history_path.is_file():
        return None
    try:
        with history_path.open(newline="", encoding="utf-8-sig") as handle:
            fieldnames = set(csv.DictReader(handle).fieldnames or ())
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return history_path if set(LEJEPA_FIELDS).issubset(fieldnames) else None


def find_audio_lejepa_history(logdir: Path) -> Path | None:
    history_path = logdir / "logs" / "history.csv"
    if not history_path.is_file():
        return None
    try:
        with history_path.open(newline="", encoding="utf-8-sig") as handle:
            fieldnames = set(csv.DictReader(handle).fieldnames or ())
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return history_path if set(AUDIO_LEJEPA_FIELDS).issubset(fieldnames) else None


def read_audio_lejepa_history(history_path: Path) -> list[dict[str, float]]:
    """Read a running AUDIO-LeJEPA CSV, using the latest row for each step."""
    rows_by_step: dict[int, dict[str, float]] = {}
    with history_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            try:
                row = {field: float(raw_row[field]) for field in AUDIO_LEJEPA_FIELDS}
                step = int(row["step"])
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in row.values()):
                rows_by_step[step] = row
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def read_lejepa_history(history_path: Path) -> list[dict[str, float]]:
    """Read complete LeJEPA rows, keeping the last duplicate iteration.

    A training process may be appending to the CSV while this function reads
    it, so malformed/incomplete rows are skipped. Restarts can append an
    already-seen (epoch, iteration); in that case the newest row is used.
    """
    rows_by_iteration: dict[tuple[int, int], dict[str, float]] = {}
    with history_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(LEJEPA_FIELDS) - fieldnames
        if missing:
            raise ValueError(
                f"{history_path} is missing LeJEPA columns: {', '.join(sorted(missing))}"
            )
        for raw_row in reader:
            try:
                row = {field: float(raw_row[field]) for field in LEJEPA_FIELDS}
                epoch = int(row["epoch"])
                iteration = int(row["iteration"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in row.values()):
                continue
            rows_by_iteration[(epoch, iteration)] = row

    return sorted(
        rows_by_iteration.values(),
        key=lambda row: (row["global_step"], row["epoch"], row["iteration"]),
    )


def moving_average(values: list[float], window: int) -> list[float]:
    result = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def infer_sigreg_weight(logdir: Path, fallback: float = 0.02) -> float:
    """Read loss.sigreg.weight from the experiment parameter snapshot."""
    candidates = [
        logdir / "params-pretrain_video_lejepa.yaml",
        logdir / "params-pretrain_v_lejepa.yaml",  # legacy runs
    ]
    candidates.extend(sorted(logdir.glob("params*.yaml")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            import yaml

            with path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            weight = float(config["loss"]["sigreg"]["weight"])
            if math.isfinite(weight) and weight >= 0:
                return weight
        except (ImportError, OSError, TypeError, ValueError, KeyError):
            continue
    return fallback


def plot_vjepa(losses, output: Path, window: int, plt) -> int:
    """Plot the legacy V-JEPA loss figure and return its point count."""
    points = sorted(
        (epoch, iteration, sum(values) / len(values))
        for (epoch, iteration), values in losses.items()
    )
    y_values = [point[2] for point in points]
    x_values = list(range(len(points)))
    smoothed = moving_average(y_values, window)

    figure, axis = plt.subplots(figsize=(11, 6), dpi=150)
    axis.plot(x_values, y_values, color="#9aa5b1", linewidth=0.8, alpha=0.45, label="loss")
    axis.plot(
        x_values,
        smoothed,
        color="#d1495b",
        linewidth=2.0,
        label=f"moving average ({window})",
    )
    axis.set_title("V-JEPA Training Loss")
    axis.set_xlabel("Logged iteration")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return len(points)


def plot_lejepa(
    rows: list[dict[str, float]],
    output: Path,
    window: int,
    sigreg_weight: float,
    title: str,
    plt,
) -> None:
    """Plot the detailed four-panel VIDEO-LeJEPA training report."""
    steps = [row["global_step"] for row in rows]

    def values(field: str) -> list[float]:
        return [row[field] for row in rows]

    def smooth(field: str) -> list[float]:
        return moving_average(values(field), window)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160, constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(steps, values("loss"), color="tab:blue", alpha=0.12, linewidth=0.6)
    axis.plot(steps, smooth("loss"), color="tab:blue", linewidth=2, label=f"total loss (MA={window})")
    axis.plot(
        steps,
        smooth("invariance_loss"),
        color="tab:orange",
        linewidth=2,
        label=f"invariance (MA={window})",
    )
    axis.set(title="Training Loss", xlabel="Optimizer step", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[0, 1]
    raw_sigreg = values("sigreg_loss")
    smooth_sigreg = moving_average(raw_sigreg, window)
    axis.plot(steps, raw_sigreg, color="tab:green", alpha=0.10, linewidth=0.6)
    axis.plot(
        steps,
        smooth_sigreg,
        color="tab:green",
        linewidth=2,
        label=f"raw SIGReg (MA={window})",
    )
    axis.set(title="SIGReg Loss", xlabel="Optimizer step", ylabel="Raw SIGReg")
    axis.grid(alpha=0.25)
    weighted_axis = axis.twinx()
    weighted_axis.plot(
        steps,
        [sigreg_weight * value for value in smooth_sigreg],
        color="tab:red",
        linewidth=1.7,
        linestyle="--",
        label=f"weighted SIGReg (×{sigreg_weight:g})",
    )
    weighted_axis.set_ylabel("Weighted contribution", color="tab:red")
    weighted_axis.tick_params(axis="y", labelcolor="tab:red")
    lines = axis.get_lines()[1:] + weighted_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], loc="best")

    axis = axes[1, 0]
    axis.plot(
        steps,
        smooth("embedding_std"),
        color="tab:purple",
        linewidth=2,
        label="embedding std",
    )
    axis.set(
        title="Representation Statistics",
        xlabel="Optimizer step",
        ylabel="Embedding std",
    )
    axis.tick_params(axis="y", labelcolor="tab:purple")
    axis.grid(alpha=0.25)
    rank_axis = axis.twinx()
    rank_axis.plot(steps, smooth("rankme"), color="tab:brown", linewidth=2, label="RankMe")
    rank_axis.set_ylabel("RankMe", color="tab:brown")
    rank_axis.tick_params(axis="y", labelcolor="tab:brown")
    lines = axis.get_lines() + rank_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], loc="best")

    axis = axes[1, 1]
    axis.plot(steps, values("lr"), color="tab:cyan", linewidth=2, label="learning rate")
    axis.set(title="Optimization", xlabel="Optimizer step", ylabel="Learning rate")
    axis.tick_params(axis="y", labelcolor="tab:cyan")
    axis.grid(alpha=0.25)
    grad_axis = axis.twinx()
    grad_axis.plot(
        steps,
        smooth("grad_norm"),
        color="tab:red",
        linewidth=1.5,
        alpha=0.85,
        label=f"grad norm (MA={window})",
    )
    grad_axis.set_ylabel("Gradient norm", color="tab:red")
    grad_axis.tick_params(axis="y", labelcolor="tab:red")
    lines = axis.get_lines() + grad_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], loc="best")

    figure.suptitle(title, fontsize=15)
    figure.savefig(output)
    plt.close(figure)


def plot_audio_lejepa(
    rows: list[dict[str, float]],
    output: Path,
    window: int,
    sigreg_weight: float,
    title: str,
    plt,
) -> None:
    """Plot AUDIO-LeJEPA metrics with loss terms on readable scales."""
    steps = [row["step"] for row in rows]

    def values(field: str) -> list[float]:
        return [row[field] for row in rows]

    def smooth(field: str) -> list[float]:
        return moving_average(values(field), window)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160, constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(steps, values("loss"), color="#ED66A4", alpha=0.18, linewidth=0.6)
    axis.plot(steps, smooth("loss"), color="#C94F8B", linewidth=2, label="total loss")
    axis.plot(steps, smooth("invariance_loss"), color="#F59E42", linewidth=2, label="invariance")
    axis.set(title="Training Loss", xlabel="Optimizer step", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[0, 1]
    axis.plot(steps, values("sigreg_loss"), color="#8CC7A1", alpha=0.18, linewidth=0.6)
    axis.plot(steps, smooth("sigreg_loss"), color="#6FA987", linewidth=2, label="raw SIGReg")
    axis.set(title="SIGReg Loss", xlabel="Optimizer step", ylabel="Raw SIGReg")
    axis.grid(alpha=0.25)
    weighted_axis = axis.twinx()
    weighted_axis.plot(
        steps, [sigreg_weight * value for value in smooth("sigreg_loss")],
        color="#D17E32", linewidth=1.7, linestyle="--",
        label=f"weighted SIGReg (×{sigreg_weight:g})",
    )
    weighted_axis.set_ylabel("Weighted contribution")
    lines = axis.get_lines()[1:] + weighted_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines])

    axis = axes[1, 0]
    axis.plot(steps, smooth("global_embedding_std"), color="#9E7BBC", linewidth=2, label="global std")
    axis.plot(steps, smooth("local_embedding_std"), color="#805F9F", linewidth=2, label="local std")
    axis.set(title="Representation Statistics", xlabel="Optimizer step", ylabel="Embedding std")
    axis.grid(alpha=0.25)
    rank_axis = axis.twinx()
    rank_axis.plot(steps, smooth("rankme"), color="#C4B080", linewidth=1.7, label="RankMe")
    rank_axis.set_ylabel("RankMe")
    lines = axis.get_lines() + rank_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines])

    axis = axes[1, 1]
    axis.plot(steps, values("lr_pretrained"), color="#8690C2", linewidth=2, label="pretrained LR")
    axis.plot(steps, values("lr_new"), color="#6673A8", linewidth=2, label="new LR")
    axis.set(title="Optimization", xlabel="Optimizer step", ylabel="Learning rate")
    axis.grid(alpha=0.25)
    grad_axis = axis.twinx()
    grad_axis.plot(steps, smooth("grad_norm"), color="#F59E42", linewidth=1.5, label="grad norm")
    grad_axis.set_ylabel("Gradient norm")
    lines = axis.get_lines() + grad_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines])

    figure.suptitle(title, fontsize=15)
    figure.savefig(output)
    plt.close(figure)


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
        default=None,
        help="moving-average window (default: 100 for video LeJEPA, 20 otherwise)",
    )
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        default=None,
        help="LeJEPA SIGReg weight (default: infer from params YAML, otherwise 0.02)",
    )
    args = parser.parse_args()

    if not args.logdir.is_dir():
        parser.error(f"log directory does not exist: {args.logdir}")
    if args.window is not None and args.window < 1:
        parser.error("--window must be at least 1")
    if args.sigreg_weight is not None and (
        not math.isfinite(args.sigreg_weight) or args.sigreg_weight < 0
    ):
        parser.error("--sigreg-weight must be a finite, non-negative number")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required; install it with `pip install matplotlib`"
        ) from error

    output = args.out or DEFAULT_OUTPUT_ROOT / args.logdir.resolve().name / "train_loss.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    audio_history_path = find_audio_lejepa_history(args.logdir)
    history_path = find_lejepa_history(args.logdir)
    if audio_history_path is not None:
        window = args.window if args.window is not None else 20
        rows = read_audio_lejepa_history(audio_history_path)
        if not rows:
            parser.error(f"no complete AUDIO-LeJEPA training rows found in {audio_history_path}")
        sigreg_weight = (
            args.sigreg_weight
            if args.sigreg_weight is not None
            else infer_sigreg_weight(args.logdir)
        )
        plot_audio_lejepa(
            rows, output, window, sigreg_weight,
            f"AUDIO-LeJEPA Training — {args.logdir.resolve().name}", plt,
        )
        print(f"detected AUDIO-LeJEPA history: {audio_history_path}")
        print(
            f"read {len(rows)} points; moving-average window: {window}; "
            f"SIGReg weight: {sigreg_weight:g}"
        )
    elif history_path is not None:
        window = args.window if args.window is not None else 100
        rows = read_lejepa_history(history_path)
        if not rows:
            parser.error(f"no complete LeJEPA training rows found in {history_path}")
        sigreg_weight = (
            args.sigreg_weight
            if args.sigreg_weight is not None
            else infer_sigreg_weight(args.logdir)
        )
        plot_lejepa(
            rows,
            output,
            window,
            sigreg_weight,
            f"VIDEO-LeJEPA Training — {args.logdir.resolve().name}",
            plt,
        )
        print(f"detected LeJEPA history: {history_path}")
        print(
            f"read {len(rows)} points; moving-average window: {window}; "
            f"SIGReg weight: {sigreg_weight:g}"
        )
    else:
        window = args.window if args.window is not None else 20
        losses = read_losses(args.logdir)
        if not losses:
            parser.error(f"no supported training rows found under {args.logdir}")
        point_count = plot_vjepa(losses, output, window, plt)
        rank_count = max(len(values) for values in losses.values())
        print(f"detected legacy V-JEPA rank logs under: {args.logdir}")
        print(
            f"read {point_count} averaged points; moving-average window: {window}; "
            f"maximum rank samples per point: {rank_count}"
        )

    print(f"saved: {output}")


if __name__ == "__main__":
    main()
