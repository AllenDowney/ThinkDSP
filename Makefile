PROJECT_NAME = ThinkDSP

.PHONY: help create_environment create_environment_dev delete_environment \
	update_environment update_environment_dev clean lint format tests

help:
	@echo "Available targets:"
	@echo "  create_environment      - Create conda environment from environment.yml"
	@echo "  create_environment_dev  - Create dev environment (base + editable think-dsp)"
	@echo "  delete_environment      - Remove conda environment"
	@echo "  update_environment      - Update environment from environment.yml (with --prune)"
	@echo "  update_environment_dev  - Update dev environment"
	@echo "  clean                   - Remove temporary files and caches"
	@echo "  lint                    - Run linters (flake8, black --check)"
	@echo "  format                  - Format code with black"
	@echo "  tests                   - Run tests with pytest"

## Set up Python environment
create_environment:
	mamba env create -f environment.yml
	@echo ""
	@echo ">>> Environment created successfully!"
	@echo ">>> Activate with: conda activate $(PROJECT_NAME)"

## Create dev environment (base + editable install of local think-dsp)
create_environment_dev: create_environment
	mamba env update -f environment-dev.yml --name $(PROJECT_NAME)
	@echo ""
	@echo ">>> Dev environment created successfully!"
	@echo ">>> Activate with: conda activate $(PROJECT_NAME)"
	@echo ">>> Local package is editable (-e .). CI uses: pip install -r requirements-dev.txt"

## Delete environment
delete_environment:
	mamba env remove --name $(PROJECT_NAME)
	@echo ">>> Environment $(PROJECT_NAME) removed"

## Update environment
update_environment:
	mamba env update -f environment.yml --name $(PROJECT_NAME) --prune
	@echo ">>> Environment updated successfully!"

## Update dev environment
update_environment_dev: update_environment
	mamba env update -f environment-dev.yml --name $(PROJECT_NAME) --prune
	@echo ">>> Dev environment updated successfully!"

## Run tests
tests:
	@echo "Running tests..."
	pytest tests/
	@echo ""
	@echo "Running notebook tests..."
	cd nb && pytest --nbmake chap0[2346789].ipynb
	@echo ">>> Tests complete!"

## Run linters (check only)
lint:
	@echo "Running linters..."
	flake8 . --config pyproject.toml || true
	black --check --config pyproject.toml . || true
	@echo ">>> Linting complete!"

## Format code
format:
	@echo "Formatting code..."
	black --config pyproject.toml .
	@echo ">>> Formatting complete!"

## Clean temporary files and caches
clean:
	@echo "Cleaning temporary files and caches..."
	-find . -type f -name "*.py[co]" -delete
	-find . -type d -name "__pycache__" -delete
	-find . -type d -name ".pytest_cache" -delete
	-find . -type d -name ".ipynb_checkpoints" -delete
	-rm -rf .ruff_cache/
	-rm -rf .mypy_cache/
	-rm -rf build/
	-rm -rf dist/
	-rm -rf *.egg-info
	@echo ">>> Cleanup complete!"
