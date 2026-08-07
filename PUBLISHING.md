# Publishing ThinkDSP to PyPI

Manual process for releasing `think-dsp` to PyPI. Releases are intentional — there is no tag-triggered auto-publish.

The **PyPI distribution name** is `think-dsp`. The **import name** remains `thinkdsp`
(`import thinkdsp`).

## Prerequisites

1. PyPI account at https://pypi.org
2. API token (or credentials) configured for upload — e.g. `~/.pypirc` with a `__token__` entry, or an interactive token when twine prompts
3. Conda env: `conda activate ThinkDSP` (needs `build` and `twine`)

## Release checklist

### 1. Bump version

Update the version in `pyproject.toml`:

```toml
[tool.poetry]
version = "0.2.0"  # Update this
```

Commit that change on `v2` when you are ready.

### 2. Build and check

```bash
conda activate ThinkDSP
rm -rf dist/ build/
python -m build
python -m twine check dist/*
```

### 3. Upload to PyPI (when you decide)

```bash
python -m twine upload dist/*
```

### 4. Verify

- https://pypi.org/project/think-dsp/
- Install with: `pip install think-dsp`
- Import with: `import thinkdsp`

### 5. Optional: annotate the release in git

Tagging is optional documentation only (it does **not** publish):

```bash
git tag v0.2.0
git push origin v0.2.0
```

## Testing on TestPyPI

```bash
python -m build
python -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ think-dsp
```

## Current Configuration

- **Package Name (PyPI)**: `think-dsp`
- **Import Name**: `thinkdsp`
- **Current Version**: `0.2.0` (check `pyproject.toml`)
- **Python Version**: `>=3.9,<4.0`
- **Build Backend**: `poetry-core`
- **License**: MIT

## References

- [TestPyPI](https://test.pypi.org/)
- [Twine](https://twine.readthedocs.io/)
