import numpy as np
import torch
from datasets.audio_jepa.finetune_dataset import AudioCSVDataset
from .dataset import VALeJEPADataset


class AVCSVDataset(VALeJEPADataset):
    def __init__(self, cfg, split, training=False, **kwargs):
        cfg = dict(cfg, local_views=0)
        super().__init__(cfg, training, **kwargs)
        self.task = str(cfg.get('task', 'classification'))
        self.label_column = cfg.get('label_column', 'label')
        self.num_class = int(cfg.get('num_class', 2))
        if self.task not in {'classification', 'regression', 'multi_label_classification'}:
            raise ValueError(f'unsupported task {self.task}')
        split_key = self.label_column + '_split'
        for row in self.rows:
            if split_key not in row or self.label_column not in row:
                raise ValueError(f'VA downstream CSV requires {self.label_column} and {split_key}')
        self.rows = [r for r in self.rows if str(r[split_key]).strip() not in {'', '-1'} and int(r[split_key]) == split]
        if not self.rows:
            raise ValueError(f'no paired samples for split={split}')
        # Reuse FAVOR's label convention (one-based classes, pipe-delimited multilabel).
        self.labels = [AudioCSVDataset._parse_label(self, row[self.label_column], row['video_path'], i) for i, row in enumerate(self.rows)]
        if self.task == 'classification' and any(label < 0 or label >= self.num_class for label in self.labels):
            raise ValueError('classification label outside one-based class range')
        if self.task == 'regression' and not np.isfinite(self.labels).all():
            raise ValueError('nonfinite regression label')

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if 'decode_error' in item:
            return item
        item['label'] = torch.as_tensor(self.labels[index], dtype=torch.long if self.task == 'classification' else torch.float32)
        return item
