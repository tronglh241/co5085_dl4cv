import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def find_run_dirs(root):
    if any(p.is_file() and p.name.startswith('events.out.tfevents') for p in root.iterdir()):
        return [root]

    runs, seen = [], set()

    for p in sorted(root.rglob('events.out.tfevents*')):
        if p.is_file() and p.parent not in seen:
            seen.add(p.parent)
            runs.append(p.parent)

    return runs


def extract_run(run_dir, filename, tags):
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={'scalars': 0})
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']

    if tags is not None:
        unknown = sorted(tags - set(scalar_tags))

        if unknown:
            raise ValueError(f'run {run_dir}: unknown tags: {unknown}')

        scalar_tags = [t for t in scalar_tags if t in tags]

    latest = {}

    for tag in scalar_tags:
        for ev in ea.Scalars(tag):
            key = (tag, ev.step)

            if key not in latest or ev.wall_time > latest[key][0]:
                latest[key] = (ev.wall_time, ev.value)

    rows = [(tag, step, wall, value) for (tag, step), (wall, value) in latest.items()]
    rows.sort(key=lambda r: (r[0], r[1]))
    out = run_dir / filename

    with out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'step', 'wall_time', 'value'])
        writer.writerows(rows)

    return out


def main():
    parser = argparse.ArgumentParser(description='Extract TensorBoard scalars to per-run CSVs.')
    parser.add_argument('--logdir', default='logs', help='run folder or parent folder to walk')
    parser.add_argument('--tags', default=None, help='comma-separated scalar tags (default: all)')
    parser.add_argument('--filename', default='metrics.csv', help='csv name written into each run folder')
    args = parser.parse_args()

    if Path(args.filename).name != args.filename:
        parser.error('--filename must be a bare file name, not a path')

    root = Path(args.logdir)

    if not root.is_dir():
        parser.error(f'--logdir is not a directory: {root}')

    tags = {tag.strip() for tag in args.tags.split(',') if tag.strip()} if args.tags else None
    runs = find_run_dirs(root)

    if not runs:
        parser.error(f'no TensorBoard event files under: {root}')

    for run_dir in runs:
        print(f'wrote {extract_run(run_dir, args.filename, tags)}')


if __name__ == '__main__':
    main()
