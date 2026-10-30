'''Hugging Face backbones for A1 with three freezing modes.

freeze='none'     every weight trains: full finetune, or from scratch when pretrained=False
freeze='partial'  the first half of the network is frozen (PARTIAL_FREEZE)
freeze='head'     only the classifier trains (linear probe)

The backbones are microsoft/resnet-50 and facebook/deit-small-patch16-224, loaded
through transformers (the DeiT repo is a ViTForImageClassification: the model is
the same, with the CLS token feeding the classifier).

"First half" is counted in stages (CNN) or blocks (Transformer), not in
parameters. ResNet-50 keeps most of its weights in stages 3-4, so its partial
mode still trains about 95% of the parameters while DeiT-S trains about half.
Report num_trainable() next to every result.
'''

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from transformers import AutoConfig, AutoModelForImageClassification

FREEZE_MODES = ('none', 'partial', 'head')

# Parameter-name prefixes frozen in partial mode, keyed by config.model_type.
PARTIAL_FREEZE = {
    'resnet': ('resnet.embedder.', 'resnet.encoder.stages.0.', 'resnet.encoder.stages.1.'),
    'vit': ('vit.embeddings.',) + tuple(f'vit.encoder.layer.{i}.' for i in range(6)),
}

# Parameters that get no weight decay besides biases and norm scales (ndim <= 1).
NO_DECAY_SUFFIXES = ('cls_token', 'position_embeddings')


def build_network(model_name, num_classes, pretrained):
    '''The Hugging Face classifier with a fresh num_classes-way head.

    pretrained=True keeps every pretrained weight except the 1000-way head;
    pretrained=False builds the same architecture with random weights.
    '''
    if pretrained:
        return AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True,
        )

    return AutoModelForImageClassification.from_config(
        AutoConfig.from_pretrained(model_name, num_labels=num_classes),
    )


def _partial_prefixes(net):
    model_type = net.config.model_type

    if model_type not in PARTIAL_FREEZE:
        raise ValueError(f'no partial-freeze rule for {model_type!r}; pass freeze_prefixes')

    return PARTIAL_FREEZE[model_type]


def freeze_parameters(net, freeze, freeze_prefixes=None):
    '''Set requires_grad=False on the parameters a freezing mode keeps fixed.

    Separate from A1Classifier so benchmark.py and analyze.py can count the
    trainable parameters of a mode without downloading pretrained weights.
    '''
    if freeze not in FREEZE_MODES:
        raise ValueError(f'freeze must be one of {FREEZE_MODES}, got {freeze!r}')

    if freeze == 'none':
        return

    names = [name for name, _ in net.named_parameters()]

    if freeze == 'head':
        frozen = {name for name in names if not name.startswith('classifier.')}
    else:
        prefixes = tuple(freeze_prefixes) if freeze_prefixes is not None else _partial_prefixes(net)
        frozen = {name for name in names if name.startswith(prefixes)}

        # A prefix that matches nothing is a typo, not a smaller freeze.
        unused = [prefix for prefix in prefixes if not any(name.startswith(prefix) for name in frozen)]
        if unused:
            raise ValueError(f'freeze prefixes match no parameter of {net.config.model_type}: {unused}')

    for name, param in net.named_parameters():
        if name in frozen:
            param.requires_grad_(False)


class A1Classifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        pretrained=False,
        freeze='none',
        freeze_prefixes=None,
    ):
        super().__init__()
        if freeze != 'none' and not pretrained:
            raise ValueError(f'freeze={freeze!r} keeps random weights fixed; it needs pretrained=True')

        self.model_name = model_name
        self.freeze = freeze
        self.net = build_network(model_name, num_classes, pretrained)
        freeze_parameters(self.net, freeze, freeze_prefixes)

    def train(self, mode=True):
        super().train(mode)

        # A frozen BatchNorm must not keep updating its running statistics,
        # otherwise the "frozen" backbone still drifts during training.
        for module in self.net.modules():
            if isinstance(module, _BatchNorm) and not any(p.requires_grad for p in module.parameters()):
                module.eval()

        return self

    def param_groups(self, weight_decay):
        '''AdamW groups over trainable weights only.

        Biases, norm scales (ndim <= 1) and the CLS token and positional
        embedding get no weight decay.
        '''
        decay, no_decay = [], []

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith(NO_DECAY_SUFFIXES):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]

    def num_params(self):
        return sum(param.numel() for param in self.parameters())

    def num_trainable(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x):
        return self.net(pixel_values=x).logits
