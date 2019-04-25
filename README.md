# DPD (Data Programming By Demonstration)

[![Build Status](https://travis-ci.com/AkshatSh/DPD.svg?branch=master)](https://travis-ci.com/AkshatSh/DPD)

[Project Website](https://akshatsh.github.io/DPD/)

## Project Description

The aim of this project is to figure out: how can we augment our training data through weak supervision in an active learning loop to gain more out of our data with little annotation.

## Setup

```bash

$ conda create --name dpd --file deps/conda_requirements.txt
# Setting up conda env

$ bash scripts/setup.sh
# Setting up ...

$ pip freeze > deps/requirements.txt
# Save pip state

$ conda list --export | deps/conda_requirements.txt
# save conda env state

```