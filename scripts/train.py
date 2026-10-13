import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from common.config.config import CfgNode
from common.config.validate import validate_train_config
from common.seed import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    # Seed before eval(). Loading evaluates the YAML and builds the model, so a
    # later set_seed never reaches the weight initialization.
    cfg = CfgNode.load(args.config, evaluate=False)
    seed = cfg['seed'] if 'seed' in cfg else 42
    if not isinstance(seed, int):
        raise ValueError(f'seed must be a plain int, got {seed!r}')
    set_seed(seed)
    cfg = cfg.eval()
    validate_train_config(cfg)

    if 'loader' in cfg:
        cfg.loader.attach(cfg.trainer, cfg)

    if 'evaluators' in cfg:
        if 'metrics' in cfg:
            for key in cfg.evaluators:
                for name, metric in cfg.metrics.items():
                    metric.attach(cfg.evaluators[key], name)

        def evaluate(engine):
            for key in cfg.evaluators:
                cfg.evaluators[key].run()

        cfg.trainer.on_epoch_completed(evaluate)

    if 'logging' in cfg:
        if 'tensorboard' in cfg.logging:
            cfg.logging.tensorboard.attach(cfg.trainer, 'training', optimizer=cfg.optimizer)

            if 'evaluators' in cfg:
                for name, evaluator in cfg.evaluators.items():
                    cfg.logging.tensorboard.attach(evaluator, name, epoch_source=cfg.trainer)

        if 'tqdm' in cfg.logging:
            evaluators = cfg.evaluators if 'evaluators' in cfg else None
            cfg.logging.tqdm.attach(cfg.trainer, evaluators)

    if 'best_checkpoint' in cfg:
        cfg.best_checkpoint.attach(cfg.trainer, cfg)

    if 'backup_checkpoint' in cfg:
        cfg.backup_checkpoint.attach(cfg.trainer, cfg)

    if 'lr_scheduler' in cfg:
        cfg.lr_scheduler.attach(cfg.trainer)

    if 'early_stopping' in cfg:
        cfg.early_stopping.attach(cfg.trainer)

    if 'timer' in cfg:
        cfg.timer.attach(cfg.trainer)

    cfg.trainer.run()
