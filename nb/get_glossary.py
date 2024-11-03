import nbformat as nbf
from glob import glob


def process_cell(cell):
    cell_tags = cell.get('metadata', {}).get('tags', [])

    if cell['cell_type'] == 'markdown':
        source = cell['source']
        if source.startswith('## Glossary'):
            print(source)


def process_notebook(path):
    ntbk = nbf.read(path, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        process_cell(cell)

    nbf.write(ntbk, path)


# Collect a list of the notebooks in the content folder
paths = glob("chap[0-1][0-9].ipynb")

for path in sorted(paths):
    print('\n#', path, '\n')
    process_notebook(path)
