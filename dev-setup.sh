#!/bin/zsh
poetry env use 3.11 # Create the virtual environment if it does not exist
source $(poetry env info --path)/bin/activate # Activate and enter the virtual environment
poetry install --with=dev # Install dev dependencies
pre-commit install --install-hooks --overwrite -t pre-push # Set up pre-commit hooks
