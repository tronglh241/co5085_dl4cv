'''Open-set verification and identification for A2.'''

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from common.engine.engine import Events
from common.engine.evaluator import Evaluator


def compute_embeddings(model, loader, device, channels_last=False):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            embedding = model(images, return_embedding=True)
            embeddings.append(F.normalize(embedding.float(), dim=1).cpu())

    if not embeddings:
        raise ValueError('cannot compute embeddings of an empty loader')

    total = sum(embedding.shape[0] for embedding in embeddings)
    ids = list(loader.dataset.ids)
    if total != len(ids):
        raise ValueError(
            f'image loader dropped images: loaded {total}, split has {len(ids)}; set drop_last=False'
        )

    return torch.cat(embeddings, dim=0), ids


def verification_metrics(labels, scores, threshold=None):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.min() == labels.max():
        raise ValueError('verification pairs need both positive and negative labels')

    roc_auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)

    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    accuracy = (tpr * positive + (1.0 - fpr) * negative) / len(labels)
    finite = np.isfinite(thresholds)

    if threshold is None:
        best = int(np.argmax(accuracy[finite]))
        threshold = float(thresholds[finite][best])
        accuracy_value = float(accuracy[finite][best])
    else:
        threshold = float(threshold)
        accuracy_value = float(((scores >= threshold) == labels).mean())

    result = {
        'roc_auc': roc_auc,
        'eer': eer,
        'accuracy': accuracy_value,
        'threshold': threshold,
    }
    for far, name in ((1e-3, '1e-3'), (1e-2, '1e-2')):
        result[f'tar_at_far_{name}'] = float(np.interp(far, fpr, tpr))

    return result


def _read_rows(path, required):
    path = Path(path)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        return list(reader)


def identification_metrics(embeddings, ids, gallery_csv, probe_csv):
    index = {image_id: position for position, image_id in enumerate(ids)}
    gallery_rows = _read_rows(gallery_csv, ('id', 'identity'))
    probe_rows = _read_rows(probe_csv, ('id', 'identity'))

    gallery_index = torch.tensor([index[row['id']] for row in gallery_rows], dtype=torch.long)
    probe_index = torch.tensor([index[row['id']] for row in probe_rows], dtype=torch.long)
    gallery_identities = [row['identity'] for row in gallery_rows]
    identity_to_gallery = {identity: position for position, identity in enumerate(gallery_identities)}

    try:
        correct = torch.tensor([identity_to_gallery[row['identity']] for row in probe_rows], dtype=torch.long)
    except KeyError as error:
        raise ValueError(f'probe identity {error.args[0]!r} is missing from the gallery') from None

    gallery = embeddings[gallery_index]
    probe = embeddings[probe_index]
    scores = probe @ gallery.T
    ranks = (scores > scores[torch.arange(scores.shape[0]), correct].unsqueeze(1)).sum(dim=1) + 1

    return {
        'rank1': float((ranks <= 1).float().mean()),
        'recall5': float((ranks <= 5).float().mean()),
        'map': float((1.0 / ranks.float()).mean()),
        'gallery': len(gallery_rows),
        'probes': len(probe_rows),
    }


class VerificationEvaluator(Evaluator):
    def __init__(
        self,
        data,
        image_data,
        model,
        device=None,
        channels_last=False,
        threshold=None,
        amp=False,
    ):
        super().__init__(
            data,
            model,
            device=device,
            amp=amp,
            channels_last=channels_last,
        )
        self.image_data = image_data
        self.threshold = threshold
        self.scores = None
        self.labels = None
        self.embeddings = None
        self.ids = None

    @torch.no_grad()
    def run(self):
        self._fire(Events.STARTED)
        self.result = {}
        self.model.eval()
        self._fire(Events.EPOCH_STARTED)

        self.embeddings, self.ids = compute_embeddings(
            self.model,
            self.image_data,
            self.device,
            self.channels_last,
        )

        scores, labels = [], []
        total_n = 0

        for batch in self.data:
            self._fire(Events.ITERATION_STARTED)

            index_a, index_b, label = batch
            score = (self.embeddings[index_a] * self.embeddings[index_b]).sum(dim=1)
            scores.append(score)
            labels.append(label)
            total_n += label.numel()
            self.output = {'n': label.numel()}
            self.state['iteration'] += 1

            self._fire(Events.ITERATION_COMPLETED)

        self.scores = torch.cat(scores).numpy()
        self.labels = torch.cat(labels).numpy()
        self.result = verification_metrics(self.labels, self.scores, self.threshold)
        self.result['n'] = total_n
        self.state['epoch'] += 1

        self._fire(Events.EPOCH_COMPLETED)
        self._fire(Events.COMPLETED)

        return self.result
