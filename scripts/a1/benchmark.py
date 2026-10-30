'''Measure speed and memory of every (model, freezing mode), then price the run matrix.

Run once on the target GPU before fixing epoch counts:

    python scripts/a1/benchmark.py --usd-per-hour 1.5
    python scripts/a1/benchmark.py --loader-root data/food101_256 --workers 8

Next to the CSV it writes benchmark.tex (speed and memory) and budget.tex
(hours and cost per run) for reports/assignment_1.tex, all under reports/results/a1/ by
default.

The model part feeds one synthetic batch repeatedly, so the DataLoader does not
cap the numbers. --loader-root times the training DataLoader alone; if it is
slower than the model, the GPU waits on JPEG decoding and more CPU cores help.
'''

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from torch import nn

from models.a1.classifier import A1Classifier, freeze_parameters
from scripts.a1.make_configs import (EVAL_BATCH_SIZE, MODELS,
                                     TRAIN_EVAL_FRACTION, transform)

TRAIN_IMAGES = 68175
VAL_IMAGES = 7575
FREEZES = ('none', 'partial', 'head')


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _timed(step, device, warmup, steps):
    for _ in range(warmup):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)
    return time.perf_counter() - start


def measure(model_key, freeze, batch_size, warmup, steps, device):
    spec = MODELS[model_key]
    model = A1Classifier(spec['hf'], 101)
    freeze_parameters(model.net, freeze)
    model.to(device)
    memory_format = torch.channels_last if spec['channels_last'] else torch.contiguous_format
    model.to(memory_format=memory_format)
    amp = device.type == 'cuda'
    optimizer = torch.optim.AdamW(model.param_groups(0.05), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 224, 224, device=device).contiguous(memory_format=memory_format)
    y = torch.randint(0, 101, (batch_size,), device=device)

    def train_step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    train_s = _timed(train_step, device, warmup, steps)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None

    x_eval = torch.randn(EVAL_BATCH_SIZE, 3, 224, 224, device=device).contiguous(memory_format=memory_format)

    @torch.no_grad()
    def eval_step():
        model.eval()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            model(x_eval)

    eval_s = _timed(eval_step, device, warmup, steps)
    return {
        'model': model_key,
        'freeze': freeze,
        'trainable': model.num_trainable(),
        'params': model.num_params(),
        'train_img_s': steps * batch_size / train_s,
        'eval_img_s': steps * EVAL_BATCH_SIZE / eval_s,
        'peak_mem_gb': peak,
    }


def price(rows, usd_per_hour):
    '''GPU hours of every run in configs/a1/runs.json at the measured speeds.'''
    speed = {(row['model'], row['freeze']): row for row in rows}
    runs = json.loads((_repo_root / 'configs' / 'a1' / 'runs.json').read_text())['runs']
    eval_images = VAL_IMAGES + TRAIN_EVAL_FRACTION * TRAIN_IMAGES
    priced = []

    for run, info in runs.items():
        row = speed[(info['model'], info['freeze'])]
        epoch_s = TRAIN_IMAGES / row['train_img_s'] + eval_images / row['eval_img_s']
        hours = info['epochs'] * epoch_s / 3600
        priced.append({'run': run, 'epochs': info['epochs'], 'epoch_s': epoch_s, 'hours': hours})
        print(f'{run:24} {info["epochs"]:3d} ep  {epoch_s:6.1f} s/ep  {hours:5.2f} h')

    total = sum(item['hours'] for item in priced)
    print(f'total         {total:5.2f} h  ~${total * usd_per_hour:.2f}')
    print('model time only; loader stalls and container startup come on top')
    return priced


def _latex(name):
    return name.replace('_', r'\_')


def _tex(path, header, rows, columns):
    lines = [rf'\begin{{tabular}}{{{columns}}}', r'\toprule', header + r' \\', r'\midrule']
    lines += [row + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', '']
    path.write_text('\n'.join(lines))


def write_tex(rows, priced, usd_per_hour, loader_img_s, device, out_dir):
    '''benchmark.tex and budget.tex for reports/assignment_1.tex.'''
    _tex(
        out_dir / 'benchmark.tex',
        r'Model & Freeze & Trainable & Train (ảnh/s) & Eval (ảnh/s) & Bộ nhớ (GB)',
        [
            f"{_latex(row['model'])} & {row['freeze']} & {row['trainable'] / 1e6:.2f}M & "
            f"{row['train_img_s']:.0f} & {row['eval_img_s']:.0f} & "
            + ('--' if row['peak_mem_gb'] is None else f"{row['peak_mem_gb']:.1f}")
            for row in rows
        ],
        'llrrrr',
    )

    total = sum(item['hours'] for item in priced)
    budget = [
        f"{_latex(item['run'])} & {item['epochs']} & {item['epoch_s']:.0f} & {item['hours']:.2f}"
        for item in priced
    ]
    budget.append(rf'\midrule tổng & & & {total:.2f}')
    budget.append(rf'chi phí (\${usd_per_hour:.2f}/giờ) & & & \${total * usd_per_hour:.2f}')
    if loader_img_s is not None:
        budget.append(rf'DataLoader train & & \multicolumn{{2}}{{r}}{{{loader_img_s:.0f} ảnh/s}}')
    _tex(out_dir / 'budget.tex', r'Run & Epoch & s/epoch & GPU-giờ', budget, 'lrrr')
    print(f'wrote benchmark.tex and budget.tex to {out_dir} (measured on {device})')


def time_loader(root, workers, batch_size, batches):
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from datasets.food101 import Food101

    # transform() returns the config expression; evaluate it the way CfgNode does.
    dataset = Food101(root, split='train', transform=eval(transform('train'), {'transforms': transforms}))
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    iterator = iter(data)
    next(iterator)
    start = time.perf_counter()
    for _ in range(batches):
        next(iterator)
    elapsed = time.perf_counter() - start
    img_s = batches * batch_size / elapsed
    print(f'loader: {img_s:.0f} img/s with {workers} workers')
    return img_s


def main():
    parser = argparse.ArgumentParser(description='Benchmark the A1 models.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--usd-per-hour', type=float, default=1.5, help='GPU + CPU + memory price')
    parser.add_argument('--out', default=str(_repo_root / 'reports' / 'results' / 'a1' / 'benchmark.csv'))
    parser.add_argument('--loader-root', default=None, help='also time the training DataLoader on this folder')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for model_key in MODELS:
        for freeze in FREEZES:
            row = measure(model_key, freeze, args.batch_size, args.warmup, args.steps, device)
            rows.append(row)
            peak = '' if row['peak_mem_gb'] is None else f'{row["peak_mem_gb"]:.1f} GB'
            print(f'{model_key:9} {freeze:8} train {row["train_img_s"]:7.0f} img/s  '
                  f'eval {row["eval_img_s"]:7.0f} img/s  {peak}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out}')

    priced = price(rows, args.usd_per_hour)
    loader_img_s = None
    if args.loader_root:
        loader_img_s = time_loader(args.loader_root, args.workers, args.batch_size, batches=50)

    write_tex(rows, priced, args.usd_per_hour, loader_img_s, device, out.parent)


if __name__ == '__main__':
    main()
