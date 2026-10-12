"""Feature aggregation over sentences, before downstream task prediction."""
import torch
from torch import nn


class InstanceAggregator(nn.Module):
    def __init__(self, feature_dim, cfg):
        super().__init__()
        self.pooling = cfg.get('pooling', 'attention')
        if self.pooling not in {'mean', 'attention'}:
            raise ValueError('model.instance_aggregator.pooling must be mean or attention')
        if self.pooling == 'attention':
            hidden = int(cfg.get('hidden_dim', 256))
            self.attention = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(),
                nn.Dropout(float(cfg.get('dropout', 0))), nn.Linear(hidden, 1))

    def forward(self, features):
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError('instance features must be [B,N,feature] with N >= 1')
        if self.pooling == 'mean':
            return features.mean(1)
        weights = self.attention(features).float().softmax(dim=1).to(features.dtype)
        return (weights * features).sum(1)


class InstancePredictionHead(nn.Module):
    def __init__(self, feature_dim, outputs, cfg):
        super().__init__()
        self.instance_aggregator = InstanceAggregator(feature_dim, cfg)
        self.task_head = nn.Sequential(nn.LayerNorm(feature_dim),
                                      nn.Dropout(float(cfg.get('head_dropout', 0))),
                                      nn.Linear(feature_dim, outputs))

    def forward(self, sentence_features):
        return self.task_head(self.instance_aggregator(sentence_features))
