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

from common.config.config import CfgNode
from models.e2.tokenizers import (CNNStemTokenizer, PatchTokenizer,
                                  PixelTokenizer, RowTokenizer)

MEAN = 0.1308
STD = 0.3082


def _scatter(weights, kernel, stride, image_size):
    '''Paint each token onto its window and average where windows overlap.'''
    count = weights.numel()
    side = int(round(count ** 0.5))
    if side * side != count:
        raise ValueError(f'cannot scatter {count} tokens onto a square grid')
    heat = torch.zeros(image_size, image_size)
    cover = torch.zeros(image_size, image_size)
    values = weights.detach().float().reshape(side, side)
    for row in range(side):
        for col in range(side):
            top = row * stride
            left = col * stride
            if top + kernel > image_size or left + kernel > image_size:
                raise ValueError(f'token window {kernel} at {(top, left)} exceeds image_size {image_size}')
            heat[top:top + kernel, left:left + kernel] += values[row, col]
            cover[top:top + kernel, left:left + kernel] += 1
    return heat / cover.clamp(min=1)


def attention_image(weights, tokenizer, image_size):
    '''Map CLS-to-token weights onto pixels.

    Overlapping patches are added on the real kernel window and divided by how
    many windows cover each pixel. A square nearest upsample would shift them.
    '''
    if isinstance(tokenizer, RowTokenizer):
        if weights.numel() != image_size:
            raise ValueError(f'row tokenizer expects {image_size} tokens, got {weights.numel()}')
        return weights.detach().float().reshape(image_size, 1).expand(image_size, image_size)
    if isinstance(tokenizer, PatchTokenizer):
        return _scatter(weights, tokenizer.proj.kernel_size[0], tokenizer.proj.stride[0], image_size)
    if isinstance(tokenizer, PixelTokenizer):
        return _scatter(weights, 1, 1, image_size)
    if isinstance(tokenizer, CNNStemTokenizer):
        side = int(round(weights.numel() ** 0.5))
        if side * side != weights.numel() or image_size % side:
            raise ValueError(f'cannot align {weights.numel()} stem tokens with image_size {image_size}')
        cell = image_size // side
        return _scatter(weights, cell, cell, image_size)
    raise TypeError(f'unsupported tokenizer {type(tokenizer).__name__}')


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
    n_heads = cls_to_tokens.shape[1]
    shown = images.shape[0]
    figure, axes = plt.subplots(shown, n_heads, figsize=(2.1 * n_heads, 2.1 * shown), squeeze=False)
    gray = (images * STD + MEAN).clamp(0, 1).cpu()
    maps = cls_to_tokens.cpu()
    for row in range(shown):
        for head in range(n_heads):
            axis = axes[row][head]
            canvas = gray[row, 0]
            heat = attention_image(maps[row, head], model.tokenizer, canvas.shape[0])
            axis.imshow(canvas, cmap='gray', vmin=0, vmax=1)
            axis.imshow(heat, cmap='inferno', alpha=0.45)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(f'head {head}', fontsize=9)
    figure.tight_layout()
    stem = Path(args.config).stem
    if stem.startswith('test_'):
        stem = stem[len('test_'):]
    out = Path(args.out) if args.out else _repo_root / 'reports' / 'images' / 'e2' / f'attn_{stem}.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
