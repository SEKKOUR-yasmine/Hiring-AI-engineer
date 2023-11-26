#!/bin/bash

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -e .

# install pre-commit hooks
pre-commit install
