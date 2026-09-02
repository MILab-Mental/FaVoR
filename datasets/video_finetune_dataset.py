"""CSV-supervised video datasets built on the pre-training video sampler."""

import csv
from pathlib import Path

import numpy as np

from datasets.video_pretrain_dataset import VideoDataset


class VideoCSVDataset(VideoDataset):
    """A :class:`VideoDataset` filtered by a CSV ``split`` and label column.

    Video decoding, short-video handling, multi-clip sampling, and output
    layout are inherited unchanged from ``VideoDataset``. The split is read
    from ``{label_column}_split``. Classification labels are converted from
    CSV's one-based ``1..num_class`` convention to zero-based targets; 
    regression labels remain floating point values.
    """

    def __init__(self, csv_paths, split, *, label_column, task="classification", root_paths=None, **kwargs):
        if not isinstance(csv_paths, list):
            raise ValueError("csv_paths must be a list with one manifest per root_paths entry")
        paths = [Path(path).expanduser().resolve() for path in csv_paths]
        if not paths:
            raise ValueError("At least one CSV manifest is required")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"CSV file does not exist: {path}")

        if not isinstance(root_paths, list) or len(root_paths) != len(paths):
            raise ValueError("root_paths must be a list with exactly one path per CSV manifest")
        roots = [Path(root).expanduser().resolve() for root in root_paths]

        self.split = int(split)
        self.label_column = label_column
        self.split_column = f"{label_column}_split"
        self.task = task.lower()
        self._root_by_csv = dict(zip(paths, roots))
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        super().__init__(data_paths=[str(path) for path in paths], **kwargs)
        if not self.samples:
            raise ValueError(f"No samples with split={self.split} found in {paths}")

    def _load_data_path(self, data_path):
        csv_path = Path(data_path)
        samples, labels = [], []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"video_path", self.split_column, self.label_column}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Expected CSV columns {sorted(required)}; got {reader.fieldnames} in {csv_path}"
                )
            for line_no, row in enumerate(reader, start=2):
                # 先按 split 过滤：split 为空或 -1 表示该样本未分配到此任务的标签，跳过。
                split_raw = str(row.get(self.split_column, "") or "").strip()
                if split_raw in ("", "-1"):
                    continue
                try:
                    row_split = int(split_raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{self.split_column!r} must be an integer at {csv_path}:{line_no}") from error
                if row_split != self.split:
                    continue
                if not str(row.get("video_path", "")).strip():
                    raise ValueError(f"Empty video_path at {csv_path}:{line_no}")
                if not str(row.get(self.label_column, "")).strip():
                    raise ValueError(f"Empty label in {self.label_column!r} at {csv_path}:{line_no}")
                try:
                    if self.task == "classification":
                        label = int(str(row[self.label_column]).strip()) - 1
                    else:
                        label = float(str(row[self.label_column]).strip())
                except (TypeError, ValueError) as error:
                    kind = "an integer" if self.task == "classification" else "a floating-point number"
                    raise ValueError(
                        f"{self.label_column!r} must be {kind} at {csv_path}:{line_no}"
                    ) from error
                video_path = Path(row["video_path"].strip()).expanduser()
                if not video_path.is_absolute():
                    video_path = self._root_by_csv[csv_path] / video_path
                samples.append(str(video_path.resolve()))
                labels.append(label)
        return samples, labels

    def __getitem__(self, index):
        """Return parent-sampled clips paired with the CSV label for ``index``."""
        # Keep the retry behavior aligned with VideoDataset while retaining
        # the final path for per-sample prediction reports.
        sample = self.samples[index]
        while True:
            if sample.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
                item = self.get_item_image(index)
            else:
                item = self.get_item_video(index)
            if item:
                clips, label, clip_indices = item
                label = int(label) if self.task == "classification" else np.float32(label)
                return clips, label, clip_indices, sample
            index = np.random.randint(len(self.samples))
            sample = self.samples[index]


def make_videodataset_finetune_v(
    csv_paths,
    *,
    label_column,
    task="classification",
    root_paths=None,
    frames_per_clip,
    fps=None,
    train_transform=None,
    val_transform=None,
    num_clips=1,
):
    """Return CSV split-0 training and split-1 validation ``VideoDataset``s."""
    common_kwargs = dict(
        label_column=label_column,
        task=task,
        root_paths=root_paths,
        frames_per_clip=frames_per_clip,
        fps=fps,
        frame_step=None if fps is not None else 4,
        num_clips=num_clips,
    )
    train_dataset = VideoCSVDataset(
        csv_paths, split=0, transform=train_transform, random_clip_sampling=True, **common_kwargs
    )
    val_dataset = VideoCSVDataset(
        csv_paths, split=1, transform=val_transform, random_clip_sampling=False, **common_kwargs
    )
    return train_dataset, val_dataset
