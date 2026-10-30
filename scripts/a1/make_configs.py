'''Write configs/a1/{train,test}_<run>.yml and configs/a1/runs.json.

Every A1 config comes from this file, so two runs differ only where the
experiment matrix says they do. Change MODELS / MODES here and regenerate:

    python scripts/a1/make_configs.py

The matrix is the 8 core runs, 2 models x 4 modes, each trained once with
seed 42 on the full train split. Run names are <model>_<mode>, for example
resnet50_full or deit_s_scratch.
'''

import argparse
import json
from pathlib import Path

import yaml

ROOT = 'data/food101_256'
NUM_CLASSES = 101
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
# The train-split evaluator reads this nested subset of train, not all 68k
# images, to keep the per-epoch evaluation cheap. The two evaluation loaders
# get half the workers: they run only between epochs.
TRAIN_EVAL_FRACTION = 0.1

# Both pretrained on ImageNet-1k only, so pretraining data is not a confound.
MODELS = {
    'resnet50': {'timm': 'resnet50.a1_in1k', 'channels_last': True},
    'deit_s': {'timm': 'deit_small_patch16_224.fb_in1k', 'channels_last': False},
}

# One data pipeline for every model and mode, as with MNIST in E1/E2: the runs
# differ only in the model and how it is trained, never in the data they see.
# prepare_data.py already saved the 224px center crop, so train only augments
# it and eval only normalizes it.
TRAIN_STEPS = [
    'transforms.RandomHorizontalFlip(p=0.5)',
    'transforms.RandomRotation(degrees=15)',
    'transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)',
    'transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))',
]
EVAL_STEPS = []

# Starting points; check the LRs on val before the final runs (README).
MODES = {
    'scratch': {
        'pretrained': False, 'freeze': 'none', 'epochs': 40, 'warmup': 5,
        'lr': 5e-4, 'drop_path': 0.1, 'clip_grad_norm': 1.0,
    },
    'head': {
        'pretrained': True, 'freeze': 'head', 'epochs': 10, 'warmup': 1,
        'lr': 1e-3, 'drop_path': 0.0, 'clip_grad_norm': None,
    },
    'partial': {
        'pretrained': True, 'freeze': 'partial', 'epochs': 15, 'warmup': 1,
        'lr': 2e-4, 'drop_path': 0.0, 'clip_grad_norm': None,
    },
    'full': {
        'pretrained': True, 'freeze': 'none', 'epochs': 15, 'warmup': 1,
        'lr': 1e-4, 'drop_path': 0.1, 'clip_grad_norm': None,
    },
}


def q(text):
    '''A string literal: config strings are evaluated as Python.'''
    return repr(text)


def transform(kind):
    '''Config expression of the 'train' or 'eval' pipeline.'''
    if kind not in ('train', 'eval'):
        raise ValueError(f"kind must be 'train' or 'eval', got {kind!r}")

    steps = TRAIN_STEPS if kind == 'train' else EVAL_STEPS
    steps = steps + ['transforms.ToTensor()', f'transforms.Normalize({MEAN}, {STD})']
    return 'transforms.Compose([' + ', '.join(steps) + '])'


def loader(root, split, kind, batch_size, workers, shuffle=False, drop_last=False, fraction=1.0):
    return {
        'module': 'torch.utils.data',
        'name': 'DataLoader',
        'kwargs': {
            'dataset': {
                'module': 'datasets.food101',
                'name': 'Food101',
                'kwargs': {
                    'root': q(root),
                    'split': q(split),
                    'fraction': fraction,
                    'transform': transform(kind),
                },
            },
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': workers,
            'collate_fn': 'collate_batch',
            'pin_memory': True,
            'persistent_workers': workers > 0,
            'drop_last': drop_last,
        },
    }


def model_block(spec, mode, pretrained=None):
    return {
        'module': 'models.a1.classifier',
        'name': 'A1Classifier',
        'kwargs': {
            'model_name': q(spec['timm']),
            'num_classes': NUM_CLASSES,
            'pretrained': mode['pretrained'] if pretrained is None else pretrained,
            'freeze': q(mode['freeze'] if pretrained is None else 'none'),
            'drop_path_rate': mode['drop_path'] if pretrained is None else 0.0,
        },
    }


def metrics_block():
    def metric(module, name, **kwargs):
        block = {'module': f'common.metrics.{module}', 'name': name}
        if kwargs:
            block['kwargs'] = kwargs
        return block

    return {
        'loss': metric('loss', 'Loss', loss_fn={'module': 'torch.nn', 'name': 'CrossEntropyLoss'}),
        'accuracy': metric('accuracy', 'Accuracy', num_classes=NUM_CLASSES),
        'precision': metric('precision', 'Precision', num_classes=NUM_CLASSES, average=True),
        'recall': metric('recall', 'Recall', num_classes=NUM_CLASSES, average=True),
        'f1_score': metric('f1_score', 'F1Score', num_classes=NUM_CLASSES, average=True),
    }


def evaluator(data, spec):
    return {
        'module': 'common.engine.evaluator',
        'name': 'Evaluator',
        'kwargs': {
            'data': data,
            'model': 'model',
            'device': None,
            'amp': True,
            'channels_last': spec['channels_last'],
        },
    }


def train_config(run, spec, mode, seed, root, workers,
                 batch_size=BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE):
    warmup = f'len(data.train) * min({mode["warmup"]}, trainer.max_epochs - 1)'
    return {
        'data': {
            'train': loader(root, 'train', 'train', batch_size, workers,
                            shuffle=True, drop_last=True),
            'train_eval': loader(root, 'train', 'eval', eval_batch_size, workers // 2, fraction=TRAIN_EVAL_FRACTION),
            'valid': loader(root, 'val', 'eval', eval_batch_size, workers // 2),
        },
        'model': model_block(spec, mode),
        'optimizer': {
            'module': 'torch.optim',
            'name': 'AdamW',
            'kwargs': {'params': f'model.param_groups({WEIGHT_DECAY})', 'lr': mode['lr']},
        },
        'loss_fn': {
            'module': 'torch.nn',
            'name': 'CrossEntropyLoss',
            'kwargs': {'label_smoothing': LABEL_SMOOTHING},
        },
        'trainer': {
            'module': 'common.engine.trainer',
            'name': 'Trainer',
            'kwargs': {
                'data': 'data.train',
                'model': 'model',
                'optimizer': 'optimizer',
                'loss_fn': 'loss_fn',
                'device': None,
                'max_epochs': mode['epochs'],
                'amp': True,
                'channels_last': spec['channels_last'],
                'clip_grad_norm': mode['clip_grad_norm'],
            },
        },
        'evaluators': {
            'train': evaluator('data.train_eval', spec),
            'valid': evaluator('data.valid', spec),
        },
        'metrics': metrics_block(),
        'lr_scheduler': {
            'module': 'common.handlers.scheduler.iteration',
            'name': 'IterationScheduler',
            'kwargs': {
                'scheduler': {
                    'module': 'common.handlers.scheduler.warmup_cosine',
                    'name': 'WarmupCosineLR',
                    'kwargs': {
                        'optimizer': 'optimizer',
                        'warmup_steps': warmup,
                        'total_steps': 'len(data.train) * trainer.max_epochs',
                    },
                },
            },
        },
        'logging': {
            'tensorboard': {
                'module': 'common.handlers.logging.tensorboard',
                'name': 'TensorBoardLogger',
                'kwargs': {'log_dir': q(f'logs/a1/train_{run}')},
            },
            'tqdm': {'module': 'common.handlers.logging.tqdm_logger', 'name': 'TQDMLogger'},
        },
        'timer': {
            'module': 'common.handlers.timer',
            'name': 'EpochTimer',
            'kwargs': {'logger': 'logging.tensorboard'},
        },
        'memory': {
            'module': 'common.handlers.memory',
            'name': 'PeakMemory',
            'kwargs': {'logger': 'logging.tensorboard'},
        },
        'best_checkpoint': {
            'module': 'common.handlers.checkpoint.best',
            'name': 'BestCheckpoint',
            'kwargs': {
                'dirname': q(f'checkpoints/a1/train_{run}'),
                'modules': [q('model')],
                'evaluator': 'evaluators.valid',
                'score_name': q('accuracy'),
                'mode': q('max'),
                'n_saved': 1,
            },
        },
        'backup_checkpoint': {
            'module': 'common.handlers.checkpoint.backup',
            'name': 'BackupCheckpoint',
            'kwargs': {
                'dirname': q(f'checkpoints/a1/train_{run}'),
                'modules': [q(name) for name in ('model', 'optimizer', 'trainer', 'lr_scheduler', 'best_checkpoint')],
                'n_saved': 1,
            },
        },
        'loader': {
            'module': 'common.handlers.checkpoint.loader',
            'name': 'CheckpointLoader',
            'kwargs': {'path': None},
        },
        'seed': seed,
        'extralibs': {
            'transforms': 'torchvision.transforms',
            'collate_batch': {'module': 'datasets.food101', 'name': 'collate_batch'},
        },
    }


def test_config(run, spec, mode, root, workers, eval_batch_size=EVAL_BATCH_SIZE):
    # Weights come from the checkpoint, so build the bare architecture.
    return {
        'data': {'test': loader(root, 'test', 'eval', eval_batch_size, workers)},
        'model': model_block(spec, mode, pretrained=False),
        'evaluator': evaluator('data.test', spec),
        'metrics': metrics_block(),
        'logging': {'tqdm': {'module': 'common.handlers.logging.tqdm_logger', 'name': 'TQDMLogger'}},
        'loader': {
            'module': 'common.handlers.checkpoint.loader',
            'name': 'CheckpointLoader',
            'kwargs': {'path': None},
        },
        'extralibs': {
            'transforms': 'torchvision.transforms',
            'collate_batch': {'module': 'datasets.food101', 'name': 'collate_batch'},
        },
    }


def matrix():
    '''(model, mode) of every run: each model in each mode.'''
    for model in MODELS:
        for mode in MODES:
            yield model, mode


HEADER = '# Generated by scripts/a1/make_configs.py. Edit that file and regenerate.\n'


class _Dumper(yaml.SafeDumper):
    pass


def _represent_str(dumper, text):
    # "'train'" reads better than '''train''', as in the hand-written E1/E2 configs.
    style = '"' if "'" in text else None
    return dumper.represent_scalar('tag:yaml.org,2002:str', text, style=style)


_Dumper.add_representer(str, _represent_str)


def write(path, config):
    path.write_text(HEADER + yaml.dump(config, Dumper=_Dumper, sort_keys=False, width=120))


def main():
    parser = argparse.ArgumentParser(description='Generate the A1 configs.')
    parser.add_argument('--out', default='configs/a1')
    parser.add_argument('--root', default=ROOT, help='prepared Food-101 folder')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers of the train loader')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='train batch; eval uses twice this')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    sets = {}

    for model, mode_name in matrix():
        spec, mode = MODELS[model], MODES[mode_name]
        run = f'{model}_{mode_name}'
        sizes = {'batch_size': args.batch_size, 'eval_batch_size': 2 * args.batch_size}
        write(out / f'train_{run}.yml', train_config(run, spec, mode, SEED, args.root, args.workers, **sizes))
        write(out / f'test_{run}.yml', test_config(run, spec, mode, args.root, args.workers, sizes['eval_batch_size']))
        runs[run] = {
            'set': 'core',
            'model': model,
            'timm': spec['timm'],
            'mode': mode_name,
            'pretrained': mode['pretrained'],
            'freeze': mode['freeze'],
            'seed': SEED,
            'epochs': mode['epochs'],
            'lr': mode['lr'],
        }
        sets.setdefault('core', []).append(run)

    (out / 'runs.json').write_text(json.dumps({'runs': runs, 'sets': sets}, indent=2) + '\n')
    counts = ', '.join(f'{name}={len(names)}' for name, names in sets.items())
    print(f'wrote {2 * len(runs)} configs and runs.json to {out}: {counts}')


if __name__ == '__main__':
    main()
