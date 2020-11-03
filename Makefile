# s3 directory for DVC (data)
PROJECT_NAME = ml-deep-learning-final

IMAGE_TAG=0.1.0
ECR_URL=193567999519.dkr.ecr.us-east-1.amazonaws.com

# Save pyenv as local variable in Makefile
PYENV := $(shell command pyenv --version 2> /dev/null)

# Specify python kernel name used for Jupyter Notebook
PROJECT_KERNEL := dl_final

# Specify whether kernel already created
KERNEL_EXISTS := "$(shell command poetry run jupyter kernelspec list | grep $(PROJECT_KERNEL) 2> /dev/null)"

# Specify whether poetry already installed local package
LOCAL_EXISTS := "$(shell command poetry run pip freeze | grep adapticons)"

# Declaring all phony targets (avoid filename/directory collision)
.PHONY: help checkenv init_project download_data create_kernel start_lab lint test build_inference_image build_training_image local_train sagemaker_train


#========= Dependencies - No need to call directly ==========#
#============================================================#

# Make sure that a .python-version file exists in this directory. Dependency of init_project
checkenv:
ifndef PYENV
	$(error "make sure pyenv is installed and is accessible in your path, (usually by adding to PATH variable in bash_profile, zshrc, or other locations based on your platform) See: https://github.com/pyenv/pyenv#installation for the installation insructions.")
endif
ifndef PYENV_SHELL
	$(error "Add 'pyenv init' to your shell to enable shims and autocompletion, (usually by adding to your bash_profile, zshrc, or other locations based on your platform)")
endif
	@echo Detected $(PYENV)
	pyenv install --skip-existing

#============================================================#
#============================================================#


#================ Recipes for CLI Consumption ===============#
#============================================================#

help:
	@echo "Step X) [DO NOT USE] Run: make init_project"
	@echo "	Prepare development environment, ONLY USED WHEN CREATING NEW PROJECT. Initializes a poetry package and pre-commit"
	@echo "Step 1) Run: make download_data"
	@echo " Download data using DVC from s3 remote - just for processed datasets, not raw"
	@echo "Step 2) Run: make install_dependencies"
	@echo "	Install dependencies listed in pyproject.toml"
	@echo "Step 3) Run: make create_kernel"
	@echo "	Creates a kernel using poetry virtualenv - used by Jupyter Lab server"
	@echo "Step 4) Run: make start_lab"
	@echo "	Starts up a Jupyter Lab server for local experimentation/analysis"
	@echo "Step 5) Run: make build_inference_image"
	@echo " Build and push the inference image for the development environment"
	@echo "Step 6) Run: make test"
	@echo " Runs tests"
	@echo "Step 7) Run: make lint_project"
	@echo " Format with black and isort, lint with flake8 and mypy"
	@echo " "
	@echo "Running macro test: 'make run_macro_test'"
	@echo "Training on AWS Sagemaker: 'make sagemaker_train'"
	@echo "Training on local: 'make local_train'"
	@echo " "
	@exit 0

# Initializing a new poetry project and DVC - DONT RUN THIS, JUST FOR VERY 1ST TIME
init_project: checkenv
	poetry init
	pre-commit install

# Install dependencies listed in pyproject.toml file
install_dependencies:
	poetry install --no-root

# Create a poetry virtualenv based kernel for jupyter consumption
create_kernel:
ifdef KERNEL_EXISTS
	yes | poetry run jupyter kernelspec uninstall $(PROJECT_KERNEL)
endif
	poetry run python -m ipykernel install --user --name=$(PROJECT_KERNEL)

# Start a jupyterlab server
start_lab:
	poetry run jupyter lab

# Build and push the inference image for the development environment
build_inference_image:
	./build_inference.sh --environment development --push

# Perform all necessary formatting on source code
lint:
	yes | poetry run isort
	poetry run black .
	poetry run flake8 .
	poetry run mypy .

# Test the package
test:
	PYTHONPATH=. poetry run python -m pytest tests


# Build the training docker image - push to ECR
build_training_image:
	./build_training.sh --push

# Run training locally
local_train:
	poetry run python -m adapticons.modeling.standard.train --input-path data/cli_input --output-path tmp_models


#============================================================#
#============================================================#
