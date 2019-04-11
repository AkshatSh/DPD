#!/bin/bash

# create virtual env
DIR_ENV="./env"

if [ ! -d "$DIR_ENV" ]; then
  echo "Creating virtual environment ..."
  python3 -m venv env
  source env/bin/activate
  pip install -r deps/requirements.txt
  deactivate
else
  echo "Using existing virtual environment .. "
fi

echo "Finished setup..."