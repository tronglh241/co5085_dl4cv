'''Check ScratchMSA against nn.MultiheadAttention, then check tokenizer shapes.'''

import csv
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch.nn as nn

from models.e2.attention import ScratchMSA
from models.e2.tokenizers import (CNNStemTokenizer, PatchTokenizer,
                                  PixelTokenizer, RowTokenizer)
from models.e2.transformer import TransformerClassifier
from models.e2.utils import count_params

CASES = (
    (2, 49, 64, 4),
    (2, 16, 64, 4),
    (2, 28, 64, 4),
    (2, 36, 64, 4),
    (4, 8, 32, 4),
    (1, 7, 64, 4),
)
# Forward values must match tightly. Float32 gradients of the big sequence
# differ by about 1e-5 because PyTorch reduces the backward in another order.
FLOAT32_OUT_LIMIT = 1e-5
FLOAT32_GRAD_LIMIT = 1e-4
FLOAT64_LIMIT = 1e-8


def _diff(left, right):
    return (left - right).abs().max().item()


def _copy_weights(scratch, reference):
    scratch.in_proj.weight.data.copy_(reference.in_proj_weight)
    scratch.in_proj.bias.data.copy_(reference.in_proj_bias)
    scratch.out_proj.weight.data.copy_(reference.out_proj.weight)
    scratch.out_proj.bias.data.copy_(reference.out_proj.bias)


def _check_init():
    scratch = ScratchMSA(64, 4)
    if scratch.in_proj.bias.abs().max().item() != 0.0 or scratch.out_proj.bias.abs().max().item() != 0.0:
        raise AssertionError('ScratchMSA biases must start at 0, matching PyTorch')


def _run_case(batch, length, dim, heads, dtype):
    reference = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True).to(dtype=dtype)
    scratch = ScratchMSA(dim, heads, dropout=0.0).to(dtype=dtype)
    _copy_weights(scratch, reference)
    reference.eval()
    scratch.eval()

    source = torch.randn(batch, length, dim, dtype=dtype)
    left_input = source.detach().clone().requires_grad_(True)
    right_input = source.detach().clone().requires_grad_(True)
    left_out, left_attn = scratch(left_input, return_attn=True)
    right_out, right_attn = reference(
        right_input,
        right_input,
        right_input,
        need_weights=True,
        average_attn_weights=False,
    )
    left_out.sum().backward()
    right_out.sum().backward()

    diffs = {
        'out_diff': _diff(left_out, right_out),
        'attn_diff': _diff(left_attn, right_attn),
        'grad_x_diff': _diff(left_input.grad, right_input.grad),
        'grad_in_proj_diff': max(
            _diff(scratch.in_proj.weight.grad, reference.in_proj_weight.grad),
            _diff(scratch.in_proj.bias.grad, reference.in_proj_bias.grad),
        ),
        'grad_out_proj_diff': max(
            _diff(scratch.out_proj.weight.grad, reference.out_proj.weight.grad),
            _diff(scratch.out_proj.bias.grad, reference.out_proj.bias.grad),
        ),
    }
    if dtype == torch.float32:
        out_limit, grad_limit = FLOAT32_OUT_LIMIT, FLOAT32_GRAD_LIMIT
    else:
        out_limit, grad_limit = FLOAT64_LIMIT, FLOAT64_LIMIT
    diffs['ok'] = (
        diffs['out_diff'] < out_limit
        and diffs['attn_diff'] < out_limit
        and diffs['grad_x_diff'] < grad_limit
        and diffs['grad_in_proj_diff'] < grad_limit
        and diffs['grad_out_proj_diff'] < grad_limit
    )
    return diffs


def _check_shapes():
    image = torch.zeros(2, 1, 28, 28)
    specs = (
        ('patch4', lambda: PatchTokenizer(patch_size=4), 49),
        ('patch7', lambda: PatchTokenizer(patch_size=7), 16),
        ('overlap', lambda: PatchTokenizer(patch_size=8, stride=4), 36),
        ('row', lambda: RowTokenizer(), 28),
        ('cnnstem', lambda: CNNStemTokenizer(), 49),
        ('pixel', lambda: PixelTokenizer(), 784),
    )
    for name, build, n_tokens in specs:
        tokens = build()(image)
        if tokens.shape != (2, n_tokens, 64):
            raise AssertionError(f'{name}: expected {(2, n_tokens, 64)}, got {tuple(tokens.shape)}')
        for attn_impl in ('scratch', 'torch'):
            model = TransformerClassifier(build(), attn_impl=attn_impl)
            logits, attentions = model(image, return_attn=True)
            if logits.shape != (2, 10):
                raise AssertionError(f'{name}/{attn_impl}: logits {tuple(logits.shape)}')
            expected = (2, 4, n_tokens + 1, n_tokens + 1)
            if len(attentions) != 4 or attentions[-1].shape != expected:
                raise AssertionError(f'{name}/{attn_impl}: unexpected attention shapes')
            print(f'shape {name:8} {attn_impl:7} tokens={n_tokens:3} params={count_params(model)}')


def main():
    _check_init()
    _check_shapes()
    out_dir = Path(_repo_root) / 'docs' / 'assets' / 'e2'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'msa_check.csv'
    fields = (
        'dtype', 'batch', 'length', 'dim', 'heads',
        'out_diff', 'attn_diff', 'grad_x_diff', 'grad_in_proj_diff', 'grad_out_proj_diff', 'ok',
    )
    rows = []
    failed = False
    for dtype in (torch.float32, torch.float64):
        for batch, length, dim, heads in CASES:
            diffs = _run_case(batch, length, dim, heads, dtype)
            row = {
                'dtype': 'float32' if dtype == torch.float32 else 'float64',
                'batch': batch,
                'length': length,
                'dim': dim,
                'heads': heads,
                **diffs,
            }
            rows.append(row)
            failed = failed or not diffs['ok']
            status = 'ok' if diffs['ok'] else 'FAIL'
            print(
                f"{row['dtype']:7} B={batch} N={length:3} D={dim:3} h={heads} "
                f"out={diffs['out_diff']:.3e} attn={diffs['attn_diff']:.3e} "
                f"gx={diffs['grad_x_diff']:.3e} {status}"
            )

    with out_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out_path}')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
