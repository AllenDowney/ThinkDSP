# Publishing ThinkDSP to PyPI

Manual releases of `think-dsp` via **GitHub Trusted Publishing**.
You choose when to release by running the workflow in the Actions UI — nothing
publishes on push or tag.

The **PyPI distribution name** is `think-dsp`. The **import name** remains `thinkdsp`
(`import thinkdsp`).

## One-time setup

### 1. GitHub environment

In the repo: **Settings → Environments → New environment** → name it exactly `pypi`.

Optional but useful: require reviewers on that environment so “Run workflow”
still needs an approval before upload.

### 2. PyPI pending trusted publisher

Until the first successful publish, the project does not exist yet. Create a
**pending** publisher:

1. Sign in at https://pypi.org (account that should own `think-dsp`)
2. **Account settings → Publishing** (or https://pypi.org/manage/account/publishing/)
3. Under **Add a new pending publisher** → GitHub, set:
   - **PyPI project name:** `think-dsp`
   - **Owner:** `AllenDowney`
   - **Repository name:** `ThinkDSP`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
4. Save

The pending publisher creates the PyPI project on first successful workflow run.
It does **not** reserve the name until that run succeeds.

After the first release, the pending publisher becomes a normal publisher on the
project; you can manage it under the project’s **Publishing** settings.

### 3. Workflow file

`.github/workflows/publish.yml` is triggered only by `workflow_dispatch`
(Actions UI → **publish** → **Run workflow**).

## Each release

### 1. Bump version on `v2`

Edit `pyproject.toml`:

```toml
[tool.poetry]
version = "0.2.0"  # bump this
```

Commit and push to `v2` (or whatever branch you run the workflow from).

### 2. Run the workflow

1. https://github.com/AllenDowney/ThinkDSP/actions/workflows/publish.yml
2. **Run workflow**
3. Choose branch `v2` (until Option E cutover)
4. Confirm; if the `pypi` environment has required reviewers, approve the deploy

### 3. Verify

- https://pypi.org/project/think-dsp/
- `pip install think-dsp`
- `import thinkdsp`

### 4. Optional git tag

Tags are documentation only with this setup (they do **not** publish):

```bash
git tag v0.2.0
git push origin v0.2.0
```

## Local build check (optional)

Still useful before clicking Run workflow:

```bash
conda activate ThinkDSP
rm -rf dist/ build/
python -m build
python -m twine check dist/*
```

Do **not** need `twine upload` once Trusted Publishing works.

## Fallback: twine + API token

If Trusted Publishing is unavailable, use a **user-scoped** API token for the
first upload (project-scoped tokens cannot create a new project), then a
project-scoped token afterward. See https://pypi.org/manage/account/token/

```bash
python -m twine upload dist/*
```

## Current configuration

- **Package Name (PyPI):** `think-dsp`
- **Import Name:** `thinkdsp`
- **Current Version:** check `pyproject.toml`
- **Publish trigger:** `workflow_dispatch` only
- **Trusted publisher workflow:** `publish.yml`
- **GitHub environment:** `pypi`

## References

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [Creating a project through OIDC](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
- [TestPyPI](https://test.pypi.org/)
