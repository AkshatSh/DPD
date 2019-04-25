#!/bin/bash
set -e

# download glove files

# set up experiment data folder if it does not exist
EXPERIMENT_DATA_DIR='./experiments/data'
if [ ! -d "$EXPERIMENT_DATA_DIR" ]; then
  echo "No experiment data dir found, creating..."
  mkdir "./experiments/data/"
else
  echo "Experiment data dir exists..."
fi

# check if glove data exists
# if not download glove data
GLOVE_DIR="$EXPERIMENT_DATA_DIR/glove.6B"
if [ ! -d "$GLOVE_DIR" ]; then
  echo "No glove data dir found, downloading..."
  cd "experiments"
  cd "data"
  GLOVE_ZIP_NAME="./glove.6B.zip"
  if [ ! - d "$GLOVE_ZIP_NAME" ]; then
    echo "No zip found, downloading zip..."
    wget http://nlp.stanford.edu/data/glove.6B.zip
  else
    echo "zip found, skipping download..."
  fi
  mkdir "glove.6B"
  mv "glove.6B glove.6b/glove.6B.zip"
  cd "glove.6B"
  unzip glove.6B.zip
  rm -r glove.6B.zip
  cd ".."
  cd ".."
  cd ".."
else
  echo "GLOVE data dir exists..."
fi
