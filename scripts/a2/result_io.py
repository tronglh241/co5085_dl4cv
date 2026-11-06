from pathlib import Path


def _flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            name = f'{prefix}.{key}' if prefix else key
            yield from _flatten(item, name)
    else:
        yield prefix, value


def _format(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _parse(text):
    if text == 'null':
        return None
    if text == 'true':
        return True
    if text == 'false':
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def write_result(path, result):
    lines = [f'{key}: {_format(value)}' for key, value in _flatten(result)]
    Path(path).write_text('\n'.join(lines) + '\n')


def read_result(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split(':', 1)
        parts = key.strip().split('.')
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _parse(value.strip())
    return result
