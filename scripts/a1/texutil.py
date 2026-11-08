'''Small helpers for the LaTeX tables the A1 scripts write.'''


def bold_best(lines, higher=(), lower=()):
    '''Return `lines` with the best value of the given columns in bold.

    `lines` is a tabular as a list of lines. Data rows are the ones that end with
    "\\\\" and come after \\midrule. `higher` and `lower` are column indices
    where a larger or a smaller value is better. Values that tie at the printed
    precision are all bold.
    '''
    start = next(i for i, line in enumerate(lines) if line.startswith(r'\midrule')) + 1
    rows = [i for i in range(start, len(lines)) if lines[i].endswith(r'\\')]
    cells = {i: lines[i][:-2].split(' & ') for i in rows}
    cells = {i: [c.strip() for c in row] for i, row in cells.items()}

    for columns, pick in ((higher, max), (lower, min)):
        for column in columns:
            values = {i: row[column] for i, row in cells.items()}
            numbers = {i: float(v) for i, v in values.items() if v.replace('.', '', 1).isdigit()}
            if not numbers:
                continue
            best = pick(numbers.values())
            for i, number in numbers.items():
                if number == best:
                    cells[i][column] = r'\textbf{' + cells[i][column] + '}'

    out = list(lines)
    for i, row in cells.items():
        out[i] = ' & '.join(row) + r' \\'
    return out
