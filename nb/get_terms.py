import nbformat as nbf
from glob import glob
import re


def parse_glossary(file_path):
    glossary_dict = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    term_pattern = re.compile(r'^([a-zA-Z_]+):$')

    current_term = None
    current_definition = []

    for line in lines:
        line = line.strip()

        # Check if the line matches the pattern for a glossary term
        term_match = term_pattern.match(line)
        if term_match:
            # Save the previous term and definition
            if current_term is not None:
                glossary_dict[current_term] = ' '.join(current_definition)

            # Start a new term
            current_term = term_match.group(1)
            current_definition = []

        # Add lines to the current definition only if a term has been encountered
        elif current_term is not None:
            if line == '' and current_definition:
                # Break if encountering two consecutive newlines (end of definition)
                glossary_dict[current_term] = ' '.join(current_definition)
                current_term = None
            else:
                # Add lines to the current definition
                current_definition.append(line)

    # Add the last term and definition
    if current_term is not None:
        glossary_dict[current_term] = ' '.join(current_definition)

    return glossary_dict

# Read all glossary terms and definitions from the glossary.md file
file_path = 'glossary.md'
#glossary = parse_glossary(file_path)
glossary = {}




def print_terms(text):
    # Use regex to search for Markdown phrases surrounded with **
    bold_phrases = re.findall(r'\*\*(.*?)\*\*', text)

    # Print the found bold phrases
    for phrase in bold_phrases:
        print(f"- **{phrase}:**")
        if phrase in glossary:
            print(glossary[phrase], '\n')
        elif phrase[:-1] in glossary:
            print(glossary[phrase[:-1]], '\n')
        else:    
            print()


def process_cell(cell):
    cell_tags = cell.get('metadata', {}).get('tags', [])

    if cell['cell_type'] == 'markdown':
        source = cell['source']
        print_terms(source)


def process_notebook(path):
    ntbk = nbf.read(path, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        process_cell(cell)

    # no need to write the notebook back to the file
    # nbf.write(ntbk, path)


# Collect a list of the notebooks in the content folder
paths = glob("chap[0-1][0-9].ipynb")

for path in sorted(paths):
    print('\n#', path, '\n')
    process_notebook(path)
