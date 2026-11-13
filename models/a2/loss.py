import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.5, label_smoothing=0.0):
        super().__init__()
        self.scale = float(scale)
        self.margin = float(margin)
        self.label_smoothing = label_smoothing
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        # cos(theta + m) is no longer monotonic once theta + m > pi, so below this cosine the
        # target logit falls back to the linear term cos(theta) - m * sin(m). This keeps the logit
        # non-increasing in theta (it jumps down by about m * sin(m) - (1 - cos(m)) at the threshold),
        # which is the same approximation the insightface reference uses.
        self.threshold = math.cos(math.pi - self.margin)
        self.fallback = math.sin(math.pi - self.margin) * self.margin

    def forward(self, cosine, target):
        index = target.unsqueeze(1)
        target_cosine = cosine.gather(1, index).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        target_sine = torch.sqrt(1.0 - target_cosine * target_cosine)
        margin_cosine = target_cosine * self.cos_m - target_sine * self.sin_m
        margin_cosine = torch.where(target_cosine > self.threshold, margin_cosine, target_cosine - self.fallback)
        logits = cosine.scatter(1, index, margin_cosine)
        return F.cross_entropy(logits * self.scale, target, label_smoothing=self.label_smoothing)
