"""CSV-supervised video datasets built on the pre-training video sampler."""

import csv
from pathlib import Path

import numpy as np

from .pretrain_dataset import VideoDataset


class VideoCSVDataset(VideoDataset):
    """A :class:`VideoDataset` filtered by a CSV ``split`` and label column.

    Video decoding, short-video handling, multi-clip sampling, and output
    layout are inherited unchanged from ``VideoDataset``. The split is read
    from ``{label_column}_split``. Classification labels are converted from
    CSV's one-based ``1..num_class`` convention to zero-based targets;
    regression labels remain floating point values.

    Multi-label classification reads a pipe-separated list of one-based class
    indices (e.g. ``2|11|20``) and returns a zero-based ``[num_class]``
    ``float32`` multi-hot vector.
    """

    def __init__(self, csv_paths, split, *, label_column, task="classification", root_paths=None,
                 num_class=None, **kwargs):
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
        if self.task not in {"classification", "regression", "multi_label_classification"}:
            raise ValueError(
                "task must be 'classification', 'regression', or 'multi_label_classification'"
            )
        self.num_class = num_class
        if self.task == "multi_label_classification" and (not isinstance(num_class, int) or num_class < 2):
            raise ValueError("multi_label_classification requires data.num_class to be an integer of at least 2")
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
                    elif self.task == "multi_label_classification":
                        label = self._parse_multilabel(row[self.label_column], csv_path, line_no)
                    else:
                        label = float(str(row[self.label_column]).strip())
                except (TypeError, ValueError) as error:
                    kind = {"classification": "an integer",
                            "multi_label_classification": f"pipe-separated integers in [1, {self.num_class}]",
                            "regression": "a floating-point number"}[self.task]
                    raise ValueError(
                        f"{self.label_column!r} must be {kind} at {csv_path}:{line_no}"
                    ) from error
                video_path = Path(row["video_path"].strip()).expanduser()
                if not video_path.is_absolute():
                    video_path = self._root_by_csv[csv_path] / video_path
                samples.append(str(video_path.resolve()))
                labels.append(label)
        return samples, labels

    def _parse_multilabel(self, raw, csv_path, line_no):
        """Turn a pipe-separated one-based class list into a zero-based multi-hot vector."""
        multi_hot = np.zeros(self.num_class, dtype=np.float32)
        for token in str(raw).split("|"):
            token = token.strip()
            if not token:
                continue
            index = int(token) - 1
            if index < 0 or index >= self.num_class:
                raise ValueError(
                    f"class index {token!r} out of range [1, {self.num_class}] "
                    f"at {csv_path}:{line_no}"
                )
            multi_hot[index] = 1.0
        if not multi_hot.any():
            raise ValueError(f"no valid class index in {self.label_column!r} at {csv_path}:{line_no}")
        return multi_hot

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
                if self.task == "classification":
                    label = int(label)
                elif self.task == "multi_label_classification":
                    label = np.asarray(label, dtype=np.float32)
                else:
                    label = np.float32(label)
                return clips, label, clip_indices, sample
            index = np.random.randint(len(self.samples))
            sample = self.samples[index]


def make_videodataset_finetune_v(
    csv_paths,
    *,
    label_column,
    task="classification",
    root_paths=None,
    num_class=None,
    frames_per_clip,
    fps=None,
    frame_step=None,
    train_transform=None,
    val_transform=None,
    num_clips=1,
):
    """Return CSV split-0 training and split-1 validation ``VideoDataset``s."""
    common_kwargs = dict(
        label_column=label_column,
        task=task,
        root_paths=root_paths,
        num_class=num_class,
        frames_per_clip=frames_per_clip,
        fps=fps,
        # Preserve the historical fine-tuning sampler when requested.  If an
        # explicit fps is supplied it remains authoritative; otherwise use
        # the caller-provided frame_step (falling back to the old default 4).
        frame_step=None if fps is not None else (4 if frame_step is None else frame_step),
        num_clips=num_clips,
    )
    train_dataset = VideoCSVDataset(
        csv_paths, split=0, transform=train_transform, random_clip_sampling=True, **common_kwargs
    )
    val_dataset = VideoCSVDataset(
        csv_paths, split=1, transform=val_transform, random_clip_sampling=False, **common_kwargs
    )
    return train_dataset, val_dataset
