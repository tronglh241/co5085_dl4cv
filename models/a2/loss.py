import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.5, label_smoothing=0.0):
        super().__init__()
        self.scale = float(scale)
        self.margin = float(margin)
        self.label_smoothing = label_smoothing

    def forward(self, cosine, target):
        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        target_theta = theta.gather(1, target.unsqueeze(1)).squeeze(1)
        target_cosine = torch.cos(target_theta + self.margin)
        logits = cosine.clone()
        logits.scatter_(1, target.unsqueeze(1), target_cosine.unsqueeze(1))
        return F.cross_entropy(logits * self.scale, target, label_smoothing=self.label_smoothing)
