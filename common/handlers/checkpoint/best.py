import re

from .base import Checkpoint


class BestCheckpoint(Checkpoint):
    def __init__(
        self,
        dirname,
        modules,
        evaluator,
        score_name='loss',
        mode='min',
        n_saved=None,
    ):
        super().__init__(dirname, modules, n_saved)
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.evaluator = evaluator
        self.score_name = score_name
        self.mode = mode
        self.best_score = None

    def filename_prefix(self):
        return 'best'

    def score_value(self):
        score = self.evaluator.result[self.score_name]

        if score is None:
            return None

        return float(score)

    def save_priority(self, engine):
        score = self.score_value()
        return score if self.mode == 'max' else -score

    def is_better(self, score):
        if self.best_score is None:
            return True

        if self.mode == 'min':
            return score < self.best_score

        return score > self.best_score

    def should_save(self, engine):
        score = self.score_value()
        if score is None:
            return False

        if self.is_better(score):
            self.best_score = score

            return True

        return False

    def state_filename(self, engine):
        return f'{self.filename_prefix()}_epoch_{engine.state["epoch"]}_{self.score_name}={self.score_value():.4f}.pt'

    def priority_from_filename(self, filename):
        pattern = rf'{self.filename_prefix()}_epoch_\d+_{re.escape(self.score_name)}=(.+)\.pt'
        match = re.fullmatch(pattern, filename)
        if match is None:
            return None

        try:
            score = float(match.group(1))
        except ValueError:
            return None

        return score if self.mode == 'max' else -score

    def state_dict(self):
        return {'best_score': self.best_score}

    def load_state_dict(self, state_dict):
        if 'best_score' not in state_dict:
            raise ValueError(f'missing required key "best_score" in {sorted(state_dict)}')

        self.best_score = state_dict['best_score']
