'''How do the A1 runs hold up when the test images are degraded?

Every run's best checkpoint (the one analyze.py scores) classifies a fixed,
class-balanced subset of the test split (--per-class images of each class, seed
0) under these conditions, each at three severities except grayscale:

    gaussian_noise  pixel noise with std 0.04, 0.08, 0.16 (pixels in [0, 1])
    gaussian_blur   Gaussian blur with sigma 1, 2, 3 px
    jpeg            JPEG re-encoding at quality 20, 10, 5
    grayscale       color removed
    patch_shuffle   the image cut into a 2x2, 4x4 or 7x7 grid of tiles and
                    the tiles shuffled. Local texture survives, the global shape
                    of the dish does not. The tiles (112, 56, 32 px) are
                    multiples of DeiT-S's 16 px patch, so the shuffle never cuts
                    through one of its tokens.

The degradation is applied to the 224 px crop before normalization, and it is
deterministic: image i under a condition is degraded the same way for every
run, so all runs see the same pixels.

Writes robustness_a1.csv (accuracy per run, condition, severity),
robustness_a1.tex (mean over severities per condition, and the mean corruption
accuracy relative to the clean accuracy) and robustness_a1.png.

    python scripts/a1/robustness.py
    python scripts/a1/robustness.py --per-class 250    # the whole test split
'''

import argparse
import csv
import io
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets.food101 import Food101, collate_batch
from scripts.a1.explain import MEAN, MODELS, MODES, STD, best_checkpoints, load_model

CONDITIONS = {
    'gaussian_noise': (0.04, 0.08, 0.16),
    'gaussian_blur': (1, 2, 3),
    'jpeg': (20, 10, 5),
    'grayscale': (1,),
    'patch_shuffle': (2, 4, 7),
}


def corrupt(image, condition, severity, rng):
    '''The PIL image under one condition. rng is seeded per image, so the result is fixed.'''
    if condition == 'clean':
        return image
    if condition == 'gaussian_noise':
        pixels = np.asarray(image, dtype=np.float32) / 255
        pixels = np.clip(pixels + rng.normal(0, severity, pixels.shape), 0, 1)
        return Image.fromarray((pixels * 255).round().astype(np.uint8))
    if condition == 'gaussian_blur':
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    if condition == 'jpeg':
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=severity)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert('RGB')
    if condition == 'grayscale':
        return image.convert('L').convert('RGB')
    if condition == 'patch_shuffle':
        pixels = np.asarray(image)
        side = pixels.shape[0] // severity
        tiles = [pixels[r * side:(r + 1) * side, c * side:(c + 1) * side]
                 for r in range(severity) for c in range(severity)]
        order = rng.permutation(len(tiles))
        rows = [np.concatenate([tiles[order[r * severity + c]] for c in range(severity)], axis=1)
                for r in range(severity)]
        return Image.fromarray(np.ascontiguousarray(np.concatenate(rows, axis=0)))
    raise ValueError(f'unknown condition {condition!r}')


class Corrupted(Dataset):
    '''A subset of the test split under one condition and severity.'''

    def __init__(self, dataset, indices, condition, severity, seed=0):
        self.dataset = dataset
        self.indices = indices
        self.condition = condition
        self.severity = severity
        self.seed = seed
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        image = self.dataset.load_image(index).convert('RGB')
        if image.size != (224, 224):
            raise ValueError(f'expected the 224px crops of prepare_data.py, got {image.size}')
        rng = np.random.default_rng([self.seed, index])
        return self.to_tensor(corrupt(image, self.condition, self.severity, rng)), self.dataset.targets[index]


def balanced_subset(targets, per_class, seed=0):
    '''Sorted indices with `per_class` images of every class (all of them if a class has fewer).'''
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    picked = []
    for label in np.unique(targets):
        members = np.flatnonzero(targets == label)
        picked.append(rng.choice(members, size=min(per_class, len(members)), replace=False))
    return np.sort(np.concatenate(picked)).tolist()


def evaluate_all(models, loader, device):
    '''{run: accuracy} of every model on the same batches.'''
    hits, total = dict.fromkeys(models, 0), 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            for run, model in models.items():
                hits[run] += int((model(images).argmax(1) == labels).sum())
            total += len(labels)
    return {run: count / total for run, count in hits.items()}


def write_tables(rows, runs, results_dir):
    with (results_dir / 'robustness_a1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('run', 'model', 'mode', 'condition', 'severity', 'accuracy'))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'accuracy': f"{row['accuracy']:.4f}"})

    summary = {}
    for run, info in runs.items():
        clean = next(row['accuracy'] for row in rows if row['run'] == run and row['condition'] == 'clean')
        means = {condition: float(np.mean([row['accuracy'] for row in rows
                                           if row['run'] == run and row['condition'] == condition]))
                 for condition in CONDITIONS}
        corruption = float(np.mean(list(means.values())))
        summary[run] = {'info': info, 'clean': clean, 'means': means, 'corruption': corruption,
                        'relative': corruption / clean if clean else 0.0}

    header = ' & '.join(('Model', 'Mode', 'Sạch', 'Nhiễu', 'Mờ', 'JPEG', 'Xám', 'Xáo ô', 'TB nhiễu', 'Tương đối'))
    lines = [r'\begin{tabular}{llrrrrrrrr}', r'\toprule', header + r' \\', r'\midrule']
    for model in MODELS:
        for mode in MODES:
            run = f'{model}_{mode}'
            if run not in summary:
                continue
            item = summary[run]
            lines.append(' & '.join((
                model.replace('_', r'\_'),
                mode,
                f"{100 * item['clean']:.1f}",
                *(f"{100 * item['means'][condition]:.1f}" for condition in CONDITIONS),
                f"{100 * item['corruption']:.1f}",
                f"{100 * item['relative']:.0f}\\%",
            )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    (results_dir / 'robustness_a1.tex').write_text('\n'.join(lines))
    return summary


def plot(rows, figures_dir):
    conditions = list(CONDITIONS)
    figure, axes = plt.subplots(1, len(conditions), figsize=(3.2 * len(conditions), 3.4), sharey=True)
    colors = dict(zip(MODES, ('tab:gray', 'tab:green', 'tab:blue', 'tab:red')))
    styles = {'resnet50': ('-', 'o'), 'deit_s': ('--', 's')}
    for axis, condition in zip(axes, conditions):
        severities = (0, *CONDITIONS[condition])
        for model in MODELS:
            for mode in MODES:
                run = f'{model}_{mode}'
                clean = [row['accuracy'] for row in rows if row['run'] == run and row['condition'] == 'clean']
                if not clean:
                    continue
                values = clean + [next(row['accuracy'] for row in rows if row['run'] == run
                                       and row['condition'] == condition and row['severity'] == severity)
                                  for severity in CONDITIONS[condition]]
                line, marker = styles[model]
                axis.plot(range(len(severities)), [100 * value for value in values], linestyle=line,
                          marker=marker, markersize=3, color=colors[mode], label=f'{model}/{mode}')
        axis.set_xticks(range(len(severities)), ['clean', *map(str, CONDITIONS[condition])], fontsize=7)
        axis.set_title(condition.replace('_', ' '), fontsize=9)
        axis.set_xlabel('Severity', fontsize=8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[-1].legend(fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(Path(figures_dir) / 'robustness_a1.png', dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Accuracy of the A1 runs on degraded test images.')
    parser.add_argument('--runs', default=None, help='a set from runs.json or comma-separated run names')
    parser.add_argument('--root', default=str(_repo_root / 'data' / 'food101_256'))
    parser.add_argument('--results-dir', default=str(_repo_root / 'reports' / 'results' / 'a1'))
    parser.add_argument('--figures-dir', default=str(_repo_root / 'docs' / 'assets' / 'a1'))
    parser.add_argument('--per-class', type=int, default=50, help='test images per class (250: all of them)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    found = best_checkpoints(args.runs)
    if not found:
        raise SystemExit('no finished runs')
    models, runs = {}, {}
    for run, (info, checkpoint) in found.items():
        print(f'load {run} <- {checkpoint}', flush=True)
        models[run] = load_model(info, checkpoint, device)
        runs[run] = info

    dataset = Food101(args.root, split='test')
    indices = balanced_subset(dataset.targets, args.per_class)
    print(f'{len(indices)} test images, {args.per_class} per class')

    rows = []
    settings = [('clean', 0)] + [(condition, severity) for condition, levels in CONDITIONS.items()
                                 for severity in levels]
    for condition, severity in settings:
        loader = DataLoader(Corrupted(dataset, indices, condition, severity), batch_size=args.batch_size,
                            num_workers=args.workers, collate_fn=collate_batch, pin_memory=device.type == 'cuda')
        accuracy = evaluate_all(models, loader, device)
        for run, value in accuracy.items():
            rows.append({'run': run, 'model': runs[run]['model'], 'mode': runs[run]['mode'],
                         'condition': condition, 'severity': severity, 'accuracy': value})
        print(f'{condition:15} {severity:>5}  ' + '  '.join(f'{run} {100 * value:5.1f}'
                                                         for run, value in accuracy.items()), flush=True)

    summary = write_tables(rows, runs, results_dir)
    plot(rows, figures_dir)
    for run, item in summary.items():
        print(f"{run:18} clean {100 * item['clean']:5.1f}  corrupted {100 * item['corruption']:5.1f}  "
              f"relative {100 * item['relative']:4.0f}%")
    print(f'wrote {results_dir / "robustness_a1.csv"} and {figures_dir / "robustness_a1.png"}')


if __name__ == '__main__':
    main()
