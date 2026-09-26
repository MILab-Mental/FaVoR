import torch
from torch.nn.utils.rnn import pad_sequence


def collate_views(views):
    result = {key: torch.stack([view[key] for view in views]) for key in views[0] if key != 'audio'}
    result['audio'] = pad_sequence([view['audio'] for view in views], batch_first=True)
    return result


def collate_va_lejepa(samples):
    batch = {'global': collate_views([s['global'] for s in samples])}
    counts = {len(s['local']) for s in samples}
    if len(counts) != 1:
        raise ValueError('inconsistent paired local view counts')
    count = counts.pop()
    if count:
        # Audio lengths may differ by a sample because crop boundaries are rounded.
        flattened = collate_views([v for s in samples for v in s['local']])
        batch['local'] = {key: value.reshape(len(samples), count, *value.shape[1:]) for key, value in flattened.items()}
    else:
        batch['local'] = {'video': torch.empty(len(samples), 0)}
    for key in ('pair_id', 'source_id', 'video_path', 'audio_path', 'sync_diagnostics'):
        batch[key] = [s[key] for s in samples]
    if 'label' in samples[0]:
        batch['label'] = torch.stack([s['label'] for s in samples])
    return batch


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(child, device) for key, child in value.items()}
    return value
