'''One run-folder stamp per process.

TensorBoardLogger and Checkpoint both name their run folder with RUN_STAMP, so
the logs and checkpoints of one training run share the same folder name.

CO5085_RUN_STAMP overrides the clock. The Modal runner sets it so that every
attempt of one session, including a retry after a preemption, writes into the
same folder and can resume from the backup the previous attempt left there.
'''

import os
from datetime import datetime

STAMP_FORMAT = '%y%m%d%H%M%S'
STAMP_ENV = 'CO5085_RUN_STAMP'


def parse_stamp(name):
    try:
        return datetime.strptime(name, STAMP_FORMAT)
    except ValueError:
        return None


def is_stamp(name):
    return parse_stamp(name) is not None


def _run_stamp():
    override = os.environ.get(STAMP_ENV)

    if override:
        if not is_stamp(override):
            raise ValueError(f'{STAMP_ENV}={override!r} is not a yymmddHHMMSS stamp')

        return override

    return datetime.now().strftime(STAMP_FORMAT)


RUN_STAMP = _run_stamp()
