"""Average all ordered sentence-window predictions once per instance."""
import torch
from torch.utils.data import default_collate
from datasets.common.instances import instance_window_batches


def average_window_outputs(outputs, task):
    if isinstance(outputs[0], dict):
        return {key: average_window_outputs([o[key] for o in outputs], task) for key in outputs[0]}
    values = torch.stack(outputs).float()
    # Return equivalent logits so existing losses, metrics, and CSV writers
    # consume the mean probability, rather than applying softmax twice.
    if task == 'classification':
        return values.log_softmax(-1).logsumexp(0) - values.new_tensor(len(outputs)).log()
    if task == 'multi_label_classification':
        probabilities = values.sigmoid().mean(0).clamp(1e-7, 1 - 1e-7)
        return torch.logit(probabilities)
    return values.mean(0)


def predict_instance_batch(batch, predict_window, task):
    outputs = [average_window_outputs([predict_window(window) for window in instance_window_batches(sample)], task)
               for sample in batch['instance_samples']]
    if not outputs:
        raise ValueError('empty instance evaluation batch')
    def concatenate(values):
        if isinstance(values[0], dict):
            return {key: concatenate([value[key] for value in values]) for key in values[0]}
        return torch.cat(values, dim=0)
    return concatenate(outputs), default_collate([s['label'] for s in batch['instance_samples']]), [
        s['instance_id'] for s in batch['instance_samples']]
