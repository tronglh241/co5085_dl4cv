'''Evaluate one A2 best checkpoint on validation and test.'''

import argparse
import csv
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch

from common.config.config import CfgNode
from common.config.validate import validate_eval_config
from models.a2.evaluator import VerificationEvaluator, identification_metrics
from scripts.a2.result_io import write_result


def latest_best(checkpoint_root):
    found = sorted(Path(checkpoint_root).glob('*/best_*.pt'), key=lambda path: path.parent.name)
    return found[-1] if found else None


def checkpoint_num_classes(path):
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    return int(state['classifier.weight'].shape[0])


def main():
    parser = argparse.ArgumentParser(description='Evaluate an A2 checkpoint.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--name', default=None, help='label of the result files; default is the run of the config')
    parser.add_argument('--out-dir', default='reports/results/a2')
    parser.add_argument('--scores-dir', default='reports/results/a2/scores')
    args = parser.parse_args()

    config_path = Path(args.config)
    run = config_path.stem.removeprefix('test_')
    label = args.name or run
    checkpoint_root = Path('checkpoints/a2') / f'train_{run}'
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_best(checkpoint_root)
    if checkpoint_path is None:
        raise SystemExit(f'no best checkpoint under {checkpoint_root}')

    cfg = CfgNode.load(config_path, evaluate=False)
    if 'kwargs' in cfg.model:
        cfg.model.kwargs.pretrained = False
        # The train config takes the class count from the train split, so follow the checkpoint
        # instead of repeating that number here.
        cfg.model.kwargs.num_classes = checkpoint_num_classes(checkpoint_path)
    cfg = cfg.eval()

    cfg.loader.path = str(checkpoint_path)
    validate_eval_config(cfg)
    cfg.loader.attach(cfg.evaluator, cfg)

    val_result = cfg.evaluator.run()

    test = VerificationEvaluator(
        cfg.data.test_pairs,
        cfg.data.test,
        cfg.model,
        device=cfg.evaluator.device,
        channels_last=cfg.evaluator.channels_last,
        threshold=val_result['threshold'],
        amp=cfg.evaluator.amp,
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

    out_dir = Path(args.out_dir)
    scores_dir = Path(args.scores_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'run': label,
        'config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'params': cfg.model.num_params(),
        'params_inference': cfg.model.num_inference_params(),
        'val': val_result,
        'test': test_result,
        'identification': identification,
    }
    write_result(out_dir / f'{label}.txt', result)

    with (scores_dir / f'{label}_test.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['label', 'score'])
        writer.writerows(zip(test.labels.tolist(), test.scores.tolist()))

    print(f'{label}: val ROC-AUC {val_result["roc_auc"]:.4f}; test ROC-AUC {test_result["roc_auc"]:.4f}')


if __name__ == '__main__':
    main()
