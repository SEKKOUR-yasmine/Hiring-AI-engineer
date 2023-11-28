#!/bin/bash

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install pre-commit hooks
pre-commit install
