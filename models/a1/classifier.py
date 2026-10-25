'''timm backbones for A1 with three freezing modes.

freeze='none'     every weight trains: full finetune, or from scratch when pretrained=False
freeze='partial'  the first half of the network is frozen (PARTIAL_FREEZE)
freeze='head'     only the classifier trains (linear probe)

"First half" is counted in stages (CNN) or blocks (Transformer), not in
parameters. ResNet-50 keeps most of its weights in layer3-4, so its partial
mode still trains about 95% of the parameters while DeiT-S trains about half.
Report num_trainable() next to every result.
'''

import timm
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

FREEZE_MODES = ('none', 'partial', 'head')

# Parameter-name prefixes frozen in partial mode, keyed by architecture prefix.
PARTIAL_FREEZE = {
    'resnet': ('conv1.', 'bn1.', 'layer1.', 'layer2.'),
    'deit': ('patch_embed.', 'cls_token', 'pos_embed') + tuple(f'blocks.{i}.' for i in range(6)),
    'vit': ('patch_embed.', 'cls_token', 'pos_embed') + tuple(f'blocks.{i}.' for i in range(6)),
    'convnext': ('stem.', 'stages.0.', 'stages.1.'),
    'swin': ('patch_embed.', 'layers.0.', 'layers.1.'),
}


def _partial_prefixes(model_name):
    for arch, prefixes in PARTIAL_FREEZE.items():
        if model_name.startswith(arch):
            return prefixes

    raise ValueError(f'no partial-freeze rule for {model_name!r}; pass freeze_prefixes')


def freeze_parameters(net, model_name, freeze, freeze_prefixes=None):
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
        classifier = net.pretrained_cfg['classifier']
        frozen = {name for name in names if not name.startswith(f'{classifier}.')}
    else:
        prefixes = tuple(freeze_prefixes) if freeze_prefixes is not None else _partial_prefixes(model_name)
        frozen = {name for name in names if name.startswith(prefixes)}

        # A prefix that matches nothing is a typo, not a smaller freeze.
        unused = [prefix for prefix in prefixes if not any(name.startswith(prefix) for name in frozen)]
        if unused:
            raise ValueError(f'freeze prefixes match no parameter of {model_name}: {unused}')

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
        drop_path_rate=0.0,
    ):
        super().__init__()
        if freeze != 'none' and not pretrained:
            raise ValueError(f'freeze={freeze!r} keeps random weights fixed; it needs pretrained=True')

        self.model_name = model_name
        self.freeze = freeze
        self.net = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
        freeze_parameters(self.net, model_name, freeze, freeze_prefixes)

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

        Biases, norm scales (ndim <= 1) and the tokens timm lists in
        no_weight_decay() (cls_token, pos_embed) get no weight decay.
        '''
        skip = self.net.no_weight_decay() if hasattr(self.net, 'no_weight_decay') else set()
        decay, no_decay = [], []

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name in skip:
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
        return self.net(x)
