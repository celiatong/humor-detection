"""PyTorch fusion head for binary PQA humor classification."""

import torch
from torch import nn
from transformers import AutoModel


# The feature order matches the exported auxiliary tensors.
FEATURES = {"inc": slice(0, 1), "sub": slice(1, 65)}
FEATURE_WIDTH = 65
ABLATIONS = {
    "control": (),
    "inc": ("inc",),
    "sub": ("sub",),
    "inc-sub": ("inc", "sub"),
}


class RobertaHumorFusion(nn.Module):
    def __init__(self, backbone, ablation):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.dropout = nn.Dropout(backbone.config.hidden_dropout_prob)
        self.dense = nn.Linear(hidden + FEATURE_WIDTH, hidden)
        self.output = nn.Linear(hidden, 2)
        active = torch.zeros(FEATURE_WIDTH, dtype=torch.bool)
        for name in ABLATIONS[ablation]:
            active[FEATURES[name]] = True
        self.register_buffer("active_features", active, persistent=False)

    @classmethod
    def from_pretrained(cls, model_id, revision, ablation):
        backbone = AutoModel.from_pretrained(
            model_id, revision=revision, add_pooling_layer=False,
        )
        return cls(backbone, ablation)

    def forward(self, features, **inputs):
        context = self.backbone(**inputs).last_hidden_state[:, 0, :]
        selected = features.masked_fill(~self.active_features, 0).to(context.dtype)
        joined = torch.cat((context, selected), dim=1)
        hidden = torch.tanh(self.dense(self.dropout(joined)))
        return self.output(self.dropout(hidden))
