import argparse
import csv
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch

from common.config.config import CfgNode

MODELS = (
    ('MLP', 'configs/e1/test_mlp.yml'),
    ('CNN', 'configs/e1/test_cnn.yml'),
    ('LSTM', 'configs/e3/test_lstm_row.yml'),
    ('GRU', 'configs/e3/test_gru_row.yml'),
)
ROWS = (7, 14, 21, 28)
MNIST_MEAN = 0.1308
MNIST_STD = 0.3082
BACKGROUND = -MNIST_MEAN / MNIST_STD
COLUMNS = (
    'Model', 'Params',
    'Keep first 7', 'Keep first 14', 'Keep first 21', 'Keep first 28',
    'Hide first 7', 'Hide first 14', 'Hide first 21', 'Hide first 28',
    'Reverse row order',
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else _repo_root / path


def _load_config(config_path):
    cfg = CfgNode.load(_resolve(config_path))
    if 'loader' not in cfg or not cfg.loader.path:
        raise ValueError(f'test config must set loader.path: {config_path}')

    checkpoint = _resolve(cfg.loader.path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint}')

    cfg.loader.path = str(checkpoint)
    return cfg, checkpoint


def _load_model(cfg, checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg.model.load_state_dict(state['model'])
    cfg.model.to(device)
    cfg.model.eval()
    return cfg.model


def _keep_first(rows):
    def apply(images):
        perturbed = images.clone()
        perturbed[:, :, rows:, :] = BACKGROUND
        return perturbed

    return apply


def _hide_first(rows):
    def apply(images):
        perturbed = images.clone()
        perturbed[:, :, :rows, :] = BACKGROUND
        return perturbed

    return apply


def _reverse(images):
    return images.flip(2)


def _conditions():
    for rows in ROWS:
        yield f'Keep first {rows}', _keep_first(rows)
    for rows in ROWS:
        yield f'Hide first {rows}', _hide_first(rows)
    yield 'Reverse row order', _reverse


def _accuracy(model, loader, transform, device):
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = transform(images).to(device)
            labels = labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    return correct / total


def _write_results(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Measure row truncation and row reversal accuracy.')
    parser.add_argument('--output', default=str(_repo_root / 'reports' / 'results' / 'shared' / 'row_perturbation.csv'),
                        help='CSV output path')
    parser.add_argument('--device', default=None, help='torch device, defaults to cuda when available')
    args = parser.parse_args()

    loaded = []
    try:
        for label, config_path in MODELS:
            cfg, checkpoint = _load_config(config_path)
            loaded.append((label, cfg, checkpoint))
    except FileNotFoundError as error:
        raise SystemExit(f'{error}\nrun `bash scripts/artifacts/download.sh` to restore checkpoints') from error

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    loader = loaded[0][1].data.test
    results = []

    for label, cfg, checkpoint in loaded:
        model = _load_model(cfg, checkpoint, device)
        params = sum(parameter.numel() for parameter in model.parameters())
        result = {'Model': label, 'Params': params}
        print(f'{label} on {device}')

        for name, transform in _conditions():
            accuracy = _accuracy(model, loader, transform, device) * 100
            result[name] = f'{accuracy:.2f}'
            print(f'  {name}: {accuracy:.2f}')

        results.append(result)

    output = _resolve(args.output)
    _write_results(results, output)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
