'''Save CLS-to-token attention of the last encoder layer.'''

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common.config.config import CfgNode

MEAN = 0.1308
STD = 0.3082


def _grid(weights, n_tokens):
    side = int(round(n_tokens ** 0.5))
    if side * side == n_tokens:
        return weights.reshape(side, side)
    if n_tokens == 28:
        return weights.reshape(28, 1)
    raise ValueError(f'cannot reshape {n_tokens} tokens into a grid')


def main():
    parser = argparse.ArgumentParser(description='Plot CLS attention maps.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-images', type=int, default=4)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cfg = CfgNode.load(args.config)
    if not cfg.loader.path or not Path(cfg.loader.path).is_file():
        raise SystemExit(f'checkpoint not found: {cfg.loader.path}')

    cfg.loader.attach(cfg.evaluator, cfg)
    cfg.loader(cfg.evaluator)
    model = cfg.model
    model.eval()
    device = next(model.parameters()).device
    images, _ = next(iter(cfg.evaluator.data))
    images = images[:args.n_images].to(device)
    with torch.no_grad():
        _, attentions = model(images, return_attn=True)

    last = attentions[-1]
    cls_to_tokens = last[:, :, 0, 1:]
    n_tokens = cls_to_tokens.shape[-1]
    n_heads = cls_to_tokens.shape[1]
    shown = images.shape[0]
    figure, axes = plt.subplots(shown, n_heads, figsize=(2.1 * n_heads, 2.1 * shown), squeeze=False)
    gray = (images * STD + MEAN).clamp(0, 1).cpu()
    maps = cls_to_tokens.cpu()
    for row in range(shown):
        for head in range(n_heads):
            axis = axes[row][head]
            canvas = gray[row, 0]
            heat = _grid(maps[row, head], n_tokens).float()
            heat = F.interpolate(heat[None, None], size=canvas.shape, mode='nearest')[0, 0]
            axis.imshow(canvas, cmap='gray', vmin=0, vmax=1)
            axis.imshow(heat, cmap='inferno', alpha=0.45)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(f'head {head}', fontsize=9)
    figure.tight_layout()
    stem = Path(args.config).stem
    if stem.startswith('test_'):
        stem = stem[len('test_'):]
    out = Path(args.out) if args.out else _repo_root / 'docs' / 'assets' / 'e2' / f'attn_{stem}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
