import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from common.config.config import CfgNode
from common.config.validate import validate_eval_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    cfg = CfgNode.load(args.config)
    validate_eval_config(cfg)

    if 'loader' in cfg:
        cfg.loader.attach(cfg.evaluator, cfg)

    if 'metrics' in cfg:
        for name, metric in cfg.metrics.items():
            metric.attach(cfg.evaluator, name)

    if 'logging' in cfg:
        if 'tensorboard' in cfg.logging:
            cfg.logging.tensorboard.attach(cfg.evaluator, prefix='test')

        if 'tqdm' in cfg.logging:
            cfg.logging.tqdm.attach(cfg.evaluator)

    cfg.evaluator.run()
