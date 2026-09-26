import torch
from torch import nn


class VAFinetuner(nn.Module):
    def __init__(self, va, num_outputs, cfg):
        super().__init__()
        self.va = va
        self.strategy = cfg.get('branch_aggregation', 'separate_logits')
        if self.strategy not in {'separate_logits', 'transformer'}:
            raise ValueError('branch_aggregation must be separate_logits or transformer')
        self.heads = nn.ModuleDict({mode: nn.Linear(va.common_dim, num_outputs) for mode in va.enabled_modes})
        weights = cfg.get('branch_weights', {mode: 1 for mode in va.enabled_modes})
        weight_values = torch.tensor([float(weights.get(mode, 1)) for mode in va.enabled_modes])
        if (weight_values < 0).any() or weight_values.sum() <= 0:
            raise ValueError('aggregation weights must be nonnegative with positive sum')
        self.register_buffer('branch_weights', weight_values / weight_values.sum())
        if self.strategy == 'transformer' and len(va.enabled_modes) > 1:
            self.branch_embedding = nn.Parameter(torch.randn(len(va.enabled_modes), va.common_dim) * .02)
            self.task_token = nn.Parameter(torch.randn(1, 1, va.common_dim) * .02)
            self.aggregator = nn.TransformerEncoder(nn.TransformerEncoderLayer(va.common_dim,
                int(cfg.get('aggregator_nhead', 12)), dropout=0, batch_first=True, norm_first=True),
                int(cfg.get('aggregator_depth', 2)), enable_nested_tensor=False)
            self.head = nn.Linear(va.common_dim, num_outputs)
        if cfg.get('freeze_video_encoder', False):
            va.video_encoder.requires_grad_(False)
        if cfg.get('freeze_audio_encoder', False):
            va.audio_encoder.requires_grad_(False)
        self.freeze_fusion = bool(cfg.get('freeze_fusion_adapter', False))
        for module in (va.input_adapters, va.fusion_adapters):
            module.requires_grad_(not self.freeze_fusion)
        va.loss_projectors.requires_grad_(False)
        # With no auxiliary branch supervision, branch heads are reporting-only.
        if self.strategy == 'transformer' and len(va.enabled_modes) > 1 and float(cfg.get('branch_loss_weight', .1)) == 0:
            self.heads.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.va.train(mode)
        if self.freeze_fusion:
            self.va.input_adapters.eval()
            self.va.fusion_adapters.eval()
        return self

    def forward(self, batch):
        representations = self.va.encode_view(batch['global'])
        branches = {mode: self.heads[mode](representations[mode]) for mode in self.va.enabled_modes}
        if len(branches) == 1:
            logits = next(iter(branches.values()))
        elif self.strategy == 'separate_logits':
            logits = sum(self.branch_weights[i] * branches[mode] for i, mode in enumerate(self.va.enabled_modes))
        else:
            tokens = torch.stack([representations[mode] for mode in self.va.enabled_modes], 1) + self.branch_embedding
            seq = torch.cat((self.task_token.expand(tokens.shape[0], -1, -1), tokens), 1)
            logits = self.head(self.aggregator(seq)[:, 0])
        return {'logits': logits, 'branches': branches}
