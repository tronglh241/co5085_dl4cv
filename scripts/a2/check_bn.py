'''Check whether the dropout in the embedding head shifts the statistics of the last BatchNorm.

The embedding head is BN - Dropout - Linear - BN. While training, the input of the last BN is the Linear output
with dropout noise on its input, and the BN stores the variance of that noisy signal in running_var. At
evaluation dropout is off, so the same signal has a smaller variance than the stored one (the "variance shift"
of dropout followed by BN). The embedding is then divided by a standard deviation that is too large. A common
factor is removed by the L2 normalisation that follows, but a factor that differs between dimensions is not.

For one trained run this script measures:
  1. the variance of the last BN input with dropout on and off, against the running_var stored in training;
  2. how much the normalised embedding changes if the last BN uses the statistics of the dropout-off signal;
  3. verification and identification metrics of the model as trained and with those statistics.

Dropout on is simulated by switching only the dropout module to train mode. The backbone and the first BN stay
in eval mode, so this is an approximation of the training distribution.
'''

import argparse
import copy
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from datasets.celeba import CelebA, VerificationPairs  # noqa: E402
from models.a2 import evaluator  # noqa: E402
from scripts.a2.demo import TRANSFORM, load_model, newest_best  # noqa: E402
from scripts.a2.result_io import write_result  # noqa: E402


@torch.no_grad()
def input_moments(model, loader, device, dropout_on):
    '''Per-dimension mean and variance of the input of the last BatchNorm.'''
    model.eval()
    dropout = model.embedding[1]
    dropout.train(dropout_on)
    captured = []
    handle = model.embedding[3].register_forward_hook(lambda module, args, output: captured.append(args[0]))
    count, total, total_sq = 0, 0.0, 0.0

    try:
        for images, _ in loader:
            captured.clear()
            model(images.to(device), return_embedding=True)
            x = captured[0].double()
            total = total + x.sum(0)
            total_sq = total_sq + (x * x).sum(0)
            count += x.shape[0]
    finally:
        handle.remove()
        dropout.eval()

    mean = total / count
    return mean.cpu(), (total_sq / count - mean * mean).cpu()


def spread(values):
    values = values.float()
    low, median, high = torch.quantile(values, torch.tensor([0.0, 0.5, 1.0])).tolist()
    return {'min': low, 'median': median, 'max': high}


def metrics_for(model, root, device, workers, batch_size):
    '''Verification (validation threshold applied to test) and identification metrics of a model.'''
    model.eval()
    embeddings = {}
    for split in ('val', 'test'):
        dataset = CelebA(root, split=split, transform=TRANSFORM, download=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        embeddings[split] = evaluator.compute_embeddings(model, loader, device)

    scored = {}
    for split in ('val', 'test'):
        pairs = VerificationPairs(root, split=split, download=False)
        vectors = embeddings[split][0]
        scores = (vectors[torch.tensor(pairs.index_a)] * vectors[torch.tensor(pairs.index_b)]).sum(1).numpy()
        scored[split] = (pairs.labels, scores)

    val = evaluator.verification_metrics(*scored['val'])
    test = evaluator.verification_metrics(*scored['test'], threshold=val['threshold'])
    folder = Path(root) / 'identification'
    ident = evaluator.identification_metrics(
        *embeddings['test'], folder / 'test_gallery.csv', folder / 'test_probe.csv')
    return {
        'val_roc_auc': val['roc_auc'],
        'test_roc_auc': test['roc_auc'],
        'test_eer': test['eer'],
        'test_accuracy': test['accuracy'],
        'rank1': ident['rank1'],
        'recall5': ident['recall5'],
        'map': ident['map'],
    }


def main():
    parser = argparse.ArgumentParser(description='Check the dropout and BatchNorm interaction of an A2 run.')
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--root', default='data/celeba_112')
    parser.add_argument('--out-dir', default='reports/results/a2')
    parser.add_argument('--images', type=int, default=20000, help='train images used to measure the moments')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else newest_best(args.run)
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit(f'no checkpoint for {args.run}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.run, checkpoint).to(device)
    last_bn = model.embedding[3]
    running_mean, running_var = last_bn.running_mean.cpu().double(), last_bn.running_var.cpu().double()

    train = CelebA(args.root, split='train', transform=TRANSFORM, download=False)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train), generator=generator)[:args.images].tolist()
    loader = DataLoader(Subset(train, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    torch.manual_seed(args.seed)
    _, var_on = input_moments(model, loader, device, dropout_on=True)
    mean_off, var_off = input_moments(model, loader, device, dropout_on=False)

    ratio_on = var_on / running_var
    ratio_off = var_off / running_var
    shrink = (ratio_off.sqrt())
    mean_shift = (mean_off - running_mean).abs() / running_var.sqrt()
    stats = {
        'var_dropout_on_over_running_var': spread(ratio_on),
        'var_dropout_off_over_running_var': spread(ratio_off),
        'mean_shift_in_sd': spread(mean_shift),
        'shrink_factor_cv': float(shrink.std() / shrink.mean()),
    }

    # The same last BN with the statistics of the dropout-off signal.
    recalibrated = copy.deepcopy(model)
    recalibrated.embedding[3].running_mean.copy_(mean_off.float())
    recalibrated.embedding[3].running_var.copy_(var_off.float())

    with torch.no_grad():
        model.eval()
        recalibrated.eval()
        batches = [images for images, _ in DataLoader(Subset(train, indices[:2048]), batch_size=args.batch_size,
                                                      shuffle=False, num_workers=args.workers)]
        cosines = torch.cat([
            (F.normalize(model(images.to(device), return_embedding=True).float(), dim=1)
             * F.normalize(recalibrated(images.to(device), return_embedding=True).float(), dim=1)).sum(1).cpu()
            for images in batches
        ])
    stats['cosine_standard_vs_recalibrated'] = {
        'mean': float(cosines.mean()),
        'p01': float(torch.quantile(cosines, 0.01)),
        'min': float(cosines.min()),
    }

    standard = metrics_for(model, args.root, device, args.workers, args.batch_size)
    recal = metrics_for(recalibrated, args.root, device, args.workers, args.batch_size)

    print(f'run: {args.run}  checkpoint: {checkpoint.name}  images for the moments: {len(indices)}')
    for name, values in stats.items():
        if isinstance(values, dict):
            print(f'{name}: ' + '  '.join(f'{key} {value:.4f}' for key, value in values.items()))
        else:
            print(f'{name}: {values:.4f}')
    print(f"{'metric':16s} {'as trained':>11s} {'recalibrated':>13s} {'difference':>11s}")
    for key in standard:
        print(f'{key:16s} {standard[key]:11.4f} {recal[key]:13.4f} {recal[key] - standard[key]:+11.4f}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_result(out_dir / f'bn_check_{args.run}.txt', {
        'run': args.run, 'checkpoint': str(checkpoint), 'images': len(indices),
        'stats': stats, 'as_trained': standard, 'recalibrated': recal,
    })


if __name__ == '__main__':
    main()
