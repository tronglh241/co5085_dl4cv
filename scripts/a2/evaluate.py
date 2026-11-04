'''Evaluate one A2 best checkpoint on validation and test.'''

import argparse
import csv
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch

from common.config.config import CfgNode
from models.a2.evaluator import VerificationEvaluator, identification_metrics


def latest_best(checkpoint_root):
    found = sorted(Path(checkpoint_root).glob('*/best_*.pt'), key=lambda path: path.parent.name)
    return found[-1] if found else None


def main():
    parser = argparse.ArgumentParser(description='Evaluate an A2 checkpoint.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--out-dir', default='reports/results/a2')
    parser.add_argument('--scores-dir', default='reports/results/a2/scores')
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = CfgNode.load(config_path, evaluate=False)
    if 'kwargs' in cfg.model and 'pretrained' in cfg.model.kwargs:
        cfg.model.kwargs.pretrained = False
    cfg = cfg.eval()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_best(cfg.best_checkpoint.root)
    if checkpoint_path is None:
        raise SystemExit(f'no best checkpoint under {cfg.best_checkpoint.root}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg.model.load_state_dict(checkpoint['model'])

    val = VerificationEvaluator(
        cfg.data.valid_pairs,
        cfg.data.valid,
        cfg.model,
        device=cfg.trainer.device,
        channels_last=cfg.trainer.channels_last,
    )
    val_result = val.run()

    test = VerificationEvaluator(
        cfg.data.test_pairs,
        cfg.data.test,
        cfg.model,
        device=cfg.trainer.device,
        channels_last=cfg.trainer.channels_last,
        threshold=val_result['threshold'],
    )
    test_result = test.run()

    identification = None
    ident_dir = Path(cfg.data.test.dataset.root) / 'identification'
    gallery = ident_dir / 'test_gallery.csv'
    probe = ident_dir / 'test_probe.csv'
    if gallery.exists() and probe.exists():
        identification = identification_metrics(test.embeddings, test.ids, gallery, probe)
    else:
        print(f'warn: identification files missing under {ident_dir}')

    run = config_path.stem.removeprefix('train_')
    out_dir = Path(args.out_dir)
    scores_dir = Path(args.scores_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'run': run,
        'config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'params': cfg.model.num_params(),
        'val': val_result,
        'test': test_result,
        'identification': identification,
    }
    (out_dir / f'{run}.json').write_text(json.dumps(result, indent=2) + '\n')

    with (scores_dir / f'{run}_test.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['label', 'score'])
        writer.writerows(zip(test.labels.tolist(), test.scores.tolist()))

    print(f'{run}: val ROC-AUC {val_result["roc_auc"]:.4f}; test ROC-AUC {test_result["roc_auc"]:.4f}')


if __name__ == '__main__':
    main()
