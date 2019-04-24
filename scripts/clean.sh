#!/bin/bash
set -e

# clean logs

LOG_DIR="./logs"

if [ ! -d "$LOG_DIR" ]; then
  echo "No log dir found..."
else
  echo "Cleaning log dir ..."
  rm -rf logs/
fi