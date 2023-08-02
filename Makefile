PROJECT_NAME = ThinkDSP
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python



## Set up Python environment
create_environment:
	conda create -y --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


## Install dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt


tests:
	# looks like we can't test chapters with interactives
	cd code; pytest --nbmake chap0[2346789].ipynb
