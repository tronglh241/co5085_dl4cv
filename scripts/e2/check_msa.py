'''Check ScratchMSA against nn.MultiheadAttention, then check tokenizer shapes.'''

import csv
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch.nn as nn

from models.e2.attention import ScratchMSA, TorchMSA
from models.e2.tokenizers import (CNNStemTokenizer, PatchTokenizer,
                                  PixelTokenizer, RowTokenizer)
from models.e2.transformer import TransformerClassifier
from models.e2.utils import count_params
from scripts.e2.attention_maps import attention_image

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
# Training calls TorchMSA with return_attn=False, which dispatches to SDPA.
# SDPA reduces in a different order than the explicit softmax in ScratchMSA.
SDPA_OUT_LIMIT = 1e-4
SDPA_GRAD_LIMIT = 1e-3


def _diff(left, right):
    return (left - right).abs().max().item()


def _copy_weights(scratch, reference):
    scratch.in_proj.weight.data.copy_(reference.in_proj_weight)
    scratch.in_proj.bias.data.copy_(reference.in_proj_bias)
    scratch.out_proj.weight.data.copy_(reference.out_proj.weight)
    scratch.out_proj.bias.data.copy_(reference.out_proj.bias)


def _reference_name(name):
    '''Map a ScratchMSA parameter name onto nn.MultiheadAttention.'''
    name = name.replace('.attn.in_proj.weight', '.attn.attn.in_proj_weight')
    name = name.replace('.attn.in_proj.bias', '.attn.attn.in_proj_bias')
    return name.replace('.attn.out_proj.', '.attn.attn.out_proj.')


def _check_init():
    scratch = ScratchMSA(64, 4)
    if scratch.in_proj.bias.abs().max().item() != 0.0 or scratch.out_proj.bias.abs().max().item() != 0.0:
        raise AssertionError('ScratchMSA biases must start at 0, matching PyTorch')

    def build(attn_impl, seed):
        torch.manual_seed(seed)
        return TransformerClassifier(PatchTokenizer(), attn_impl=attn_impl)

    scratch_model = build('scratch', 0)
    torch_model = build('torch', 0)
    reference = dict(torch_model.named_parameters())
    for name, param in scratch_model.named_parameters():
        other_name = _reference_name(name)
        other = reference.get(other_name)
        if other is None or not torch.equal(param, other):
            raise AssertionError(f'init mismatch at {name} vs {other_name}')

    repeated = build('scratch', 0)
    for name, param in scratch_model.named_parameters():
        if not torch.equal(param, dict(repeated.named_parameters())[name]):
            raise AssertionError(f'seed 0 did not repeat at {name}')

    if torch.equal(scratch_model.head.weight, build('scratch', 1).head.weight):
        raise AssertionError('seed 0 and seed 1 produced the same classifier head')


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


def _copy_torch(scratch, module):
    _copy_weights(scratch, module.attn)


def _check_training_path():
    '''Cover the forward training actually runs.

    Dropout 0 compares ScratchMSA with TorchMSA(return_attn=False), the SDPA
    call. Dropout 0.1 compares ScratchMSA with nn.MultiheadAttention's math
    path: fused SDPA dropout draws a different mask, so it is not the reference.
    '''
    torch.manual_seed(0)
    source = torch.randn(2, 49, 64)
    scratch = ScratchMSA(64, 4, dropout=0.0)
    trained = TorchMSA(64, 4, dropout=0.0)
    _copy_torch(scratch, trained)
    scratch.train()
    trained.train()
    left = source.detach().clone().requires_grad_(True)
    right = source.detach().clone().requires_grad_(True)
    left_out, _ = scratch(left, return_attn=False)
    right_out, _ = trained(right, return_attn=False)
    left_out.sum().backward()
    right_out.sum().backward()
    out_gap = _diff(left_out, right_out)
    grad_gap = max(
        _diff(left.grad, right.grad),
        _diff(scratch.in_proj.weight.grad, trained.attn.in_proj_weight.grad),
        _diff(scratch.out_proj.weight.grad, trained.attn.out_proj.weight.grad),
    )
    print(f'train sdpa dropout=0 out={out_gap:.3e} grad={grad_gap:.3e}')
    if out_gap >= SDPA_OUT_LIMIT or grad_gap >= SDPA_GRAD_LIMIT:
        raise AssertionError(f'SDPA training path exceeded limits out={out_gap} grad={grad_gap}')

    reference = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
    dropped = ScratchMSA(64, 4, dropout=0.1)
    _copy_weights(dropped, reference)
    reference.train()
    dropped.train()
    torch.manual_seed(1)
    scratch_out, _ = dropped(source, return_attn=False)
    torch.manual_seed(1)
    math_out, _ = reference(source, source, source, need_weights=True, average_attn_weights=False)
    drop_gap = _diff(scratch_out, math_out)
    print(f'train math dropout=0.1 out={drop_gap:.3e}')
    if drop_gap >= FLOAT32_OUT_LIMIT:
        raise AssertionError(f'dropout math path exceeded limit out={drop_gap}')

    dropped.eval()
    fast = TorchMSA(64, 4, dropout=0.1)
    _copy_torch(dropped, fast)
    fast.eval()
    eval_gap = _diff(dropped(source, return_attn=False)[0], fast(source, return_attn=False)[0])
    print(f'eval sdpa dropout=0.1 out={eval_gap:.3e}')
    if eval_gap >= SDPA_OUT_LIMIT:
        raise AssertionError(f'SDPA eval path exceeded limit out={eval_gap}')


def _check_token_coverage():
    try:
        PatchTokenizer(image_size=28, patch_size=6, stride=6)
    except ValueError:
        pass
    else:
        raise AssertionError('patch 6 stride 6 on 28px must report the uncovered border')

    overlap = PatchTokenizer(patch_size=8, stride=4)
    weights = torch.zeros(overlap.num_tokens)
    weights[0] = 1
    heat = attention_image(weights, overlap, 28)
    outside = heat[8:].abs().sum().item() + heat[:, 8:].abs().sum().item()
    if not torch.all(heat[:8, :8] > 0) or outside:
        raise AssertionError('overlap token 0 must land on pixels [0:8, 0:8]')

    rows = RowTokenizer()
    row_weights = torch.zeros(rows.num_tokens)
    row_weights[3] = 1
    row_heat = attention_image(row_weights, rows, 28)
    if row_heat[3].min().item() != 1 or row_heat[0].sum().item() or row_heat[4].sum().item():
        raise AssertionError('a row token must paint that image row')


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
    _check_training_path()
    _check_token_coverage()
    _check_shapes()
    out_dir = Path(_repo_root) / 'reports' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'msa_check.csv'
    fields = (
        'dtype', 'batch', 'length', 'dim', 'heads',
        'out_diff', 'attn_diff', 'grad_x_diff', 'grad_in_proj_diff', 'grad_out_proj_diff', 'ok',
    )
    rows = []
    failed = False
    # The init check above builds models and draws from the RNG. Start the
    # numerical cases from a fixed seed so those draws do not change the inputs.
    torch.manual_seed(0)
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
