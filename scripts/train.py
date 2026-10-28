import argparse
import shutil
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from common.config.config import CfgNode
from common.config.validate import validate_train_config
from common.seed import set_seed
from common.stamp import is_stamp


def run_config(config_path, on_epoch_completed=None):
    '''Train from a YAML config.

    on_epoch_completed runs after the backup for that epoch is on disk. The Modal
    runner uses it to commit the volume, so a preempted retry can resume.
    '''
    cfg = CfgNode.load(config_path, evaluate=False)
    set_seed(cfg.get('seed', 42))
    cfg = cfg.eval()
    validate_train_config(cfg)

    # Resuming continues the run folder of the checkpoint, so its curves and
    # checkpoints are not split across two stamps.
    if 'loader' in cfg and cfg.loader.path:
        stamp = Path(cfg.loader.path).parent.name
        if is_stamp(stamp):
            for key in ('best_checkpoint', 'backup_checkpoint'):
                if key in cfg:
                    cfg[key].use_stamp(stamp)
            if 'logging' in cfg and 'tensorboard' in cfg.logging:
                cfg.logging.tensorboard.use_stamp(stamp)

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
            cfg.logging.tqdm.attach(cfg.trainer, cfg.get('evaluators'))

    if 'best_checkpoint' in cfg:
        cfg.best_checkpoint.attach(cfg.trainer, cfg)

    if 'lr_scheduler' in cfg:
        cfg.lr_scheduler.attach(cfg.trainer)

    if 'early_stopping' in cfg:
        cfg.early_stopping.attach(cfg.trainer)

    # Attached after every handler that changes state on EPOCH_COMPLETED, so the
    # backup holds the scheduler and early stopping as stepped for this epoch.
    if 'backup_checkpoint' in cfg:
        cfg.backup_checkpoint.attach(cfg.trainer, cfg)
        # Keep the config next to the backups. Checkpoint folders are otherwise
        # created on the first save, so make this one now.
        cfg.backup_checkpoint.dirname.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, cfg.backup_checkpoint.dirname)

    if 'timer' in cfg:
        cfg.timer.attach(cfg.trainer)

    if 'memory' in cfg:
        cfg.memory.attach(cfg.trainer)

    # After the backup handler, so the committed epoch is the one just saved.
    if on_epoch_completed is not None:
        cfg.trainer.on_epoch_completed(on_epoch_completed)

    cfg.trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    run_config(args.config)
