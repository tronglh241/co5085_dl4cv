'''ArcFace-style embedding networks on torchvision backbones.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

BACKBONES = {
    'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048, 'fc'),
    'mobilenetv3': (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        960,
        'classifier',
    ),
    'mobilenetv3_small': (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        576,
        'classifier',
    ),
}
BACKBONES['mobilenetv3_large'] = BACKBONES['mobilenetv3']


def build_backbone(name, pretrained=True):
    if name not in BACKBONES:
        raise ValueError(f'unknown backbone: {name!r} (expected one of {sorted(BACKBONES)})')

    factory, weights, features, head = BACKBONES[name]
    net = factory(weights=weights if pretrained else None)
    setattr(net, head, nn.Identity())
    return net, features


class FaceEmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone='resnet50',
        num_classes=8192,
        embedding_dim=512,
        pretrained=True,
        head='arcface',
        dropout=0.4,
        scale=64.0,
        margin=0.5,
    ):
        super().__init__()
        if head not in ('softmax', 'arcface'):
            raise ValueError(f'head must be "softmax" or "arcface", got {head!r}')
        if embedding_dim <= 0:
            raise ValueError(f'embedding_dim must be positive, got {embedding_dim}')

        self.backbone_name = backbone
        self.head = head
        self.scale = float(scale)
        self.margin = float(margin)
        self.net, features = build_backbone(backbone, pretrained)
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(features),
            nn.Dropout(dropout),
            nn.Linear(features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        if head == 'softmax':
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
            nn.init.xavier_uniform_(self.weight)

    def embedding_for_eval(self, x):
        return self.embedding(self.net(x))

    def forward(self, x, return_embedding=False):
        embedding = self.embedding(self.net(x))
        if return_embedding:
            return embedding

        if self.head == 'softmax':
            return {'embedding': embedding, 'logits': self.classifier(embedding)}

        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        return {'embedding': embedding, 'cosine': cosine}

    def num_params(self):
        return sum(param.numel() for param in self.parameters())
