import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence


def collate_views(views):
    result = {key: torch.stack([view[key] for view in views]) for key in views[0] if key != 'audio'}
    result['audio'] = pad_sequence([view['audio'] for view in views], batch_first=True)
    return result


def collate_va_lejepa(samples):
    skipped = [sample['decode_error'] for sample in samples if 'decode_error' in sample]
    samples = [sample for sample in samples if 'decode_error' not in sample]
    if not samples:
        return {'skipped_samples': skipped}
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
    batch['skipped_samples'] = skipped
    return batch


def synchronize_training_batch(batch, device):
    """Keep equal valid counts for DDP's equally weighted gradients and SIGReg.

    All ranks must call this for every loader batch, including empty batches.
    Return the usable batch and number of valid local samples discarded to
    match the other ranks. Empty on any rank means all ranks skip this batch.
    """
    local_count = batch['global']['video'].shape[0] if 'global' in batch else 0
    count = torch.tensor(local_count, device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MIN)
    common_count = int(count.item())
    if common_count == 0:
        return None, local_count
    if common_count == local_count:
        return batch, 0

    def trim(value):
        if torch.is_tensor(value):
            return value[:common_count]
        if isinstance(value, dict):
            return {key: trim(child) for key, child in value.items()}
        if isinstance(value, list):
            return value[:common_count]
        return value

    return {key: value if key == 'skipped_samples' else trim(value) for key, value in batch.items()}, local_count - common_count


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(child, device) for key, child in value.items()}
    return value
