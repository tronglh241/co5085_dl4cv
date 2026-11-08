'''Where do the A1 models look? Grad-CAM for ResNet-50, attention rollout for DeiT-S.

For every run in configs/a1/runs.json, the best checkpoint that analyze.py scores
(chosen on val) is loaded, and two things are computed on test images:

1. Heat maps on a fixed set of images, the same for all 8 runs. The images are
   drawn (seed 0) from four groups defined by the best ResNet-50 and the best
   DeiT-S run (best on val): both right, only ResNet-50 right, only DeiT-S right,
   both wrong. One figure per model, a row per image, a column per mode:
   explain_resnet50.png and explain_deit_s.png.

2. A deletion test of how faithful each map is, on a fixed random subset of
   test images. The image is cut in a 14x14 grid of 16px patches; the patches
   the map ranks highest are set to the dataset mean (0 after normalization),
   and the accuracy is measured again. The same is done with patches in random
   order. If the map points at what the model uses, deleting its top patches
   must hurt more than deleting random ones. explain_deletion.csv/.tex and
   deletion_a1.png.

The two methods differ, so the deletion test compares modes within a model more
fairly than it compares the two models:

- Grad-CAM (Selvaraju et al., 2017) weights the 7x7 feature maps of the last
  ResNet stage by the mean gradient of the predicted class score; it is class
  specific.
- Attention rollout (Abnar and Zuidema, 2020) multiplies the head-averaged
  attention matrices of all 12 blocks, each mixed half and half with the
  identity for the residual connection, and keeps the CLS token's row over the
  196 patches; it does not depend on the class.

    python scripts/a1/explain.py                      # every run with a checkpoint
    python scripts/a1/explain.py --deletion-images 500
'''

import argparse
import csv
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from datasets.food101 import Food101, collate_batch
from models.a1.classifier import A1Classifier
from scripts.a1.analyze import latest_best, load_runs

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 101
IMAGE_SIZE = 224
PATCH = 16
GRID = IMAGE_SIZE // PATCH
MODELS = ('resnet50', 'deit_s')
MODES = ('scratch', 'head', 'partial', 'full')
GROUPS = ('both right', 'only ResNet-50 right', 'only DeiT-S right', 'both wrong')
DELETE_FRACTIONS = (0.1, 0.25, 0.5)


def test_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_model(info, checkpoint, device):
    '''The A1Classifier of a run with the weights of its best checkpoint, in eval mode.'''
    model = A1Classifier(info['hf'], NUM_CLASSES, pretrained=False, dropout=info.get('dropout', 0.0))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(state)
    return model.eval().to(device)


def best_checkpoints(selection):
    '''{run: (info, checkpoint path)} for every run with a finished session.'''
    found = {}
    for run, info in load_runs(selection).items():
        checkpoint = latest_best(run, info['epochs'], info.get('patience'))
        if checkpoint is None:
            print(f'skip {run}: no finished session with a best_*.pt')
            continue
        found[run] = (info, checkpoint)
    return found


def grad_cam(model, images):
    '''(maps B x 7 x 7, logits) of Grad-CAM on the last ResNet stage for the predicted class.'''
    stage = model.net.resnet.encoder.stages[-1]
    captured = {}
    handle = stage.register_forward_hook(lambda module, args, output: captured.__setitem__('features', output))
    try:
        with torch.enable_grad():
            logits = model(images)
            score = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
            gradient, = torch.autograd.grad(score, captured['features'])
    finally:
        handle.remove()
    features = captured['features'].detach()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    return F.relu((weights * features).sum(dim=1)), logits.detach()


def attention_rollout(model, images):
    '''(maps B x 14 x 14, logits) of attention rollout from the CLS token.

    The attention probabilities are recomputed from each block's query and key
    projections, so this works whatever attention kernel transformers picks.
    '''
    attentions = []

    def keep(module, args, kwargs):
        hidden = args[0] if args else kwargs['hidden_states']
        batch, tokens, _ = hidden.shape
        shape = (batch, tokens, module.num_attention_heads, module.attention_head_size)
        query = module.query(hidden).view(shape).transpose(1, 2)
        key = module.key(hidden).view(shape).transpose(1, 2)
        probs = torch.softmax(query @ key.transpose(-1, -2) * module.attention_head_size ** -0.5, dim=-1)
        attentions.append(probs.mean(dim=1).float())

    blocks = model.net.vit.encoder.layer
    handles = [block.attention.attention.register_forward_pre_hook(keep, with_kwargs=True) for block in blocks]
    try:
        with torch.no_grad():
            logits = model(images)
    finally:
        for handle in handles:
            handle.remove()

    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=images.device)
    rollout = identity.expand(images.shape[0], tokens, tokens)
    for attention in attentions:
        mixed = 0.5 * attention + 0.5 * identity
        mixed = mixed / mixed.sum(dim=-1, keepdim=True)
        rollout = mixed @ rollout
    cls = rollout[:, 0, 1:]
    side = int(round(cls.shape[-1] ** 0.5))
    return cls.reshape(-1, side, side), logits


def saliency(model, info, images):
    if info['model'] == 'resnet50':
        return grad_cam(model, images)
    return attention_rollout(model, images)


def normalize_maps(maps):
    flat = maps.flatten(1)
    low, high = flat.min(dim=1, keepdim=True)[0], flat.max(dim=1, keepdim=True)[0]
    return ((flat - low) / (high - low).clamp_min(1e-8)).view_as(maps)


def to_grid(maps, size):
    return F.interpolate(maps[:, None], size=(size, size), mode='bilinear', align_corners=False)[:, 0]


def read_predictions(path):
    ids, labels, preds = [], [], []
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            ids.append(row['id'])
            labels.append(int(row['label']))
            preds.append(int(row['pred']))
    return ids, np.array(labels), np.array(preds)


def best_runs_on_val(results_dir):
    '''{model: run} with the highest val accuracy in results_a1.csv.'''
    best = {}
    with (results_dir / 'results_a1.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['val_acc'] and (row['model'] not in best or float(row['val_acc']) > best[row['model']][1]):
                best[row['model']] = (row['run'], float(row['val_acc']))
    return {model: run for model, (run, _) in best.items()}


def pick_images(results_dir, dataset_ids, per_group, seed=0):
    '''[(dataset index, group)] drawn from the four agreement groups of the two val-best runs.'''
    best = best_runs_on_val(results_dir)
    ids, labels, cnn = read_predictions(results_dir / 'predictions' / f"{best['resnet50']}.csv")
    _, _, vit = read_predictions(results_dir / 'predictions' / f"{best['deit_s']}.csv")
    if ids != dataset_ids:
        raise SystemExit('the predictions csv and the test dataset list the images in a different order')
    right_cnn, right_vit = cnn == labels, vit == labels
    masks = (right_cnn & right_vit, right_cnn & ~right_vit, ~right_cnn & right_vit, ~right_cnn & ~right_vit)
    rng = np.random.default_rng(seed)
    picked = []
    for group, mask in zip(GROUPS, masks):
        members = np.flatnonzero(mask)
        for index in np.sort(rng.choice(members, size=min(per_group, len(members)), replace=False)):
            picked.append((int(index), group))
    print(f"images picked from {best['resnet50']} and {best['deit_s']} (best on val)")
    return picked


def denormalize(image):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def short(name, width=18):
    name = name.replace('_', ' ')
    return name if len(name) <= width else name[:width - 1] + '.'


def plot_maps(model_name, runs, picked, images, labels, maps, preds, classes, figures_dir):
    '''One row per picked image: the image, then the map of each mode of `model_name`.'''
    modes = [mode for mode in MODES if f'{model_name}_{mode}' in runs]
    rows, cols = len(picked), 1 + len(modes)
    figure, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.05 * rows), squeeze=False)
    for row, (_, group) in enumerate(picked):
        picture = denormalize(images[row])
        axes[row, 0].imshow(picture)
        axes[row, 0].set_title(f'{short(classes[labels[row]])}\n[{group}]', fontsize=6.5)
        for col, mode in enumerate(modes, start=1):
            run = f'{model_name}_{mode}'
            heat = to_grid(maps[run][row:row + 1], IMAGE_SIZE)[0].cpu().numpy()
            axis = axes[row, col]
            axis.imshow(picture)
            axis.imshow(heat, cmap='jet', alpha=0.45, vmin=0, vmax=1)
            pred = int(preds[run][row])
            mark = 'right' if pred == labels[row] else 'wrong'
            axis.set_title(f'{mode}: {short(classes[pred], 14)}\n({mark})', fontsize=6.5,
                           color='black' if mark == 'right' else 'tab:red')
    for axis in axes.flat:
        axis.set_axis_off()
    method = 'Grad-CAM' if model_name == 'resnet50' else 'attention rollout'
    figure.suptitle(f'{model_name}: {method} (red title: wrong prediction)', fontsize=9)
    # Leave room above the first row of titles for the figure title.
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(Path(figures_dir) / f'explain_{model_name}.png', dpi=120)
    plt.close(figure)


def delete_patches(images, order, count):
    '''Set the first `count` patches of `order` (B x 196) to 0, the dataset mean after normalization.'''
    if count == 0:
        return images
    keep = torch.ones(order.shape, device=images.device)
    keep.scatter_(1, order[:, :count], 0.0)
    keep = F.interpolate(keep.view(-1, 1, GRID, GRID), scale_factor=PATCH, mode='nearest')
    return images * keep


def deletion_test(model, info, loader, device, seed=0):
    '''{fraction: (accuracy deleting salient patches, accuracy deleting random patches)}.'''
    generator = torch.Generator(device='cpu').manual_seed(seed)
    hits = {fraction: [0, 0] for fraction in (0.0, *DELETE_FRACTIONS)}
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        maps, logits = saliency(model, info, images)
        scores = to_grid(maps.float(), GRID).flatten(1)
        salient = scores.argsort(dim=1, descending=True)
        random = torch.stack([torch.randperm(GRID * GRID, generator=generator) for _ in range(len(images))]).to(device)
        hits[0.0][0] += int((logits.argmax(1) == labels).sum())
        hits[0.0][1] += int((logits.argmax(1) == labels).sum())
        with torch.no_grad():
            for fraction in DELETE_FRACTIONS:
                count = int(round(fraction * GRID * GRID))
                for slot, order in enumerate((salient, random)):
                    pred = model(delete_patches(images, order, count)).argmax(1)
                    hits[fraction][slot] += int((pred == labels).sum())
        total += len(images)
    return {fraction: (salient_hits / total, random_hits / total)
            for fraction, (salient_hits, random_hits) in hits.items()}


def write_deletion(rows, results_dir):
    fields = ('run', 'model', 'mode', 'fraction', 'acc_salient', 'acc_random')
    with (results_dir / 'explain_deletion.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f'{row[key]:.4f}' if isinstance(row[key], float) else row[key] for key in fields})

    by_run = {}
    for row in rows:
        by_run.setdefault((row['model'], row['mode']), {})[row['fraction']] = row
    lines = [
        r'\begin{tabular}{llrrrrr}',
        r'\toprule',
        r'Model & Mode & Gốc (\%) & Xóa 25\% nổi bật & Xóa 25\% ngẫu nhiên & Xóa 50\% nổi bật & Xóa 50\% ngẫu nhiên \\',
        r'\midrule',
    ]
    for (model, mode), fractions in by_run.items():
        lines.append(' & '.join((
            model.replace('_', r'\_'),
            mode,
            f"{100 * fractions[0.0]['acc_salient']:.1f}",
            f"{100 * fractions[0.25]['acc_salient']:.1f}",
            f"{100 * fractions[0.25]['acc_random']:.1f}",
            f"{100 * fractions[0.5]['acc_salient']:.1f}",
            f"{100 * fractions[0.5]['acc_random']:.1f}",
        )) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    (results_dir / 'explain_deletion.tex').write_text('\n'.join(lines))


def plot_deletion(rows, figures_dir):
    figure, axes = plt.subplots(1, len(MODELS), figsize=(10, 3.8), squeeze=False, sharey=True)
    colors = dict(zip(MODES, ('tab:gray', 'tab:green', 'tab:blue', 'tab:red')))
    for axis, model in zip(axes[0], MODELS):
        for mode in MODES:
            points = sorted((row['fraction'], row['acc_salient'], row['acc_random']) for row in rows
                            if row['model'] == model and row['mode'] == mode)
            if not points:
                continue
            fractions = [100 * point[0] for point in points]
            axis.plot(fractions, [100 * point[1] for point in points], color=colors[mode], marker='o',
                      markersize=3, label=f'{mode}: top patches')
            axis.plot(fractions, [100 * point[2] for point in points], color=colors[mode], linestyle='--',
                      linewidth=1, label=f'{mode}: random')
        method = 'Grad-CAM' if model == 'resnet50' else 'attention rollout'
        axis.set_title(f'{model} ({method})', fontsize=9)
        axis.set_xlabel('Patches set to the mean (%)')
        axis.set_ylabel('Accuracy (%)')
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(Path(figures_dir) / 'deletion_a1.png', dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM / attention rollout and a deletion test for the A1 runs.')
    parser.add_argument('--runs', default=None, help='a set from runs.json or comma-separated run names')
    parser.add_argument('--root', default=str(_repo_root / 'data' / 'food101_256'))
    parser.add_argument('--results-dir', default=str(_repo_root / 'reports' / 'results' / 'a1'))
    parser.add_argument('--figures-dir', default=str(_repo_root / 'docs' / 'assets' / 'a1'))
    parser.add_argument('--per-group', type=int, default=2, help='images shown per agreement group')
    parser.add_argument('--deletion-images', type=int, default=2000, help='test images in the deletion test (0: skip)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    runs = best_checkpoints(args.runs)
    if not runs:
        raise SystemExit('no finished runs')

    dataset = Food101(args.root, split='test', transform=test_transform())
    picked = pick_images(results_dir, dataset.ids, args.per_group)
    images = torch.stack([dataset[index][0] for index, _ in picked])
    labels = [dataset.targets[index] for index, _ in picked]

    rng = np.random.default_rng(0)
    count = min(args.deletion_images, len(dataset))
    subset = Subset(dataset, np.sort(rng.choice(len(dataset), size=count, replace=False)).tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_batch)

    maps, preds, rows = {}, {}, []
    for run, (info, checkpoint) in runs.items():
        print(f'explain {run} <- {checkpoint}', flush=True)
        model = load_model(info, checkpoint, device)
        run_maps, logits = saliency(model, info, images.to(device))
        maps[run] = normalize_maps(run_maps.float()).cpu()
        preds[run] = logits.argmax(1).cpu()
        if count:
            for fraction, (salient, random) in deletion_test(model, info, loader, device).items():
                rows.append({'run': run, 'model': info['model'], 'mode': info['mode'], 'fraction': fraction,
                             'acc_salient': salient, 'acc_random': random})
                print(f'  delete {fraction:4.2f}: top patches {100 * salient:5.1f}%  random {100 * random:5.1f}%')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    for model_name in MODELS:
        if any(run.startswith(f'{model_name}_') for run in runs):
            plot_maps(model_name, runs, picked, images, labels, maps, preds, dataset.classes, figures_dir)
    if rows:
        write_deletion(rows, results_dir)
        plot_deletion(rows, figures_dir)
    print(f'wrote figures to {figures_dir} and tables to {results_dir}')


if __name__ == '__main__':
    main()
