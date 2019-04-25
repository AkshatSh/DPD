#!/bin/bash

# create virtual env
# DIR_ENV="./env"

# if [ ! -d "$DIR_ENV" ]; then
#   echo "Creating virtual environment ..."
#   python3 -m venv env
#   # source env/bin/activate
#   pip install -r deps/requirements.txt
# else
#   echo "Using existing virtual environment .. "
#   # source env/bin/activate
# fi
pip install -r deps/requirements.txt

# update spacy corpi
echo "Downloading spaCy en..."
python -m spacy download en
echo "Finished spaCy en"

# setup NLTK corpi
echo "Downloading NLTK Corpi..."
python scripts/setup_nltk.py
echo "Finished NLTK Corpi"

echo "Finished setup"