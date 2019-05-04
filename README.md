# DPD (Data Programming By Demonstration)

[![Build Status](https://travis-ci.com/AkshatSh/DPD.svg?branch=master)](https://travis-ci.com/AkshatSh/DPD)

[Project Website](https://akshatsh.github.io/DPD/)

## Project Description

The aim of this project is to figure out: how can we augment our training data through weak supervision in an active learning loop to gain more out of our data with little annotation.

## Project Structure

```bash
├── __pycache__
├── allennlp_active_train.py
├── allennlp_train.py
├── args.py
├── constants.py
├── dataset
│   ├── __init__.py
│   ├── __pycache__
│   ├── bio_dataloader.py
│   ├── bio_dataset.py
│   └── fields
│       ├── __init__.py
│       ├── __pycache__
│       ├── float_field.py
│       └── int_field.py
├── heuristics
│   ├── __init__.py
│   ├── __pycache__
│   └── random_heuristic.py
├── models
│   ├── __init__.py
│   ├── __pycache__
│   ├── allennlp_crf.py
│   ├── allennlp_crf_tagger.py
│   ├── crf.py
│   ├── elmo_bilstm_crf.py
│   ├── embedder
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── cached_text_field_embedder.py
│   │   ├── elmo.py
│   │   ├── glove_embedding.py
│   │   ├── glove_utils.py
│   │   └── ner_elmo.py
│   └── weighted_crf.py
├── oracles
│   ├── __init__.py
│   ├── __pycache__
│   ├── gold_oracle.py
│   └── oracle.py
├── training
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── average_tag_f1.py
│   │   └── tag_f1.py
│   └── trainer.py
├── utils
│   ├── __init__.py
│   ├── __pycache__
│   ├── dataset_utils.py
│   ├── logger.py
│   ├── logging_utils.py
│   ├── model_utils.py
│   ├── save_file.py
│   ├── saving_utils.py
│   ├── tensor_utils.py
│   └── utils.py
└── weak_supervision
    ├── __init__.py
    ├── __pycache__
    ├── adverserial_functions
    ├── bio_converter.py
    ├── collator
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── collate_utils.py
    │   ├── collator.py
    │   ├── intersection_collator.py
    │   ├── metal_collator.py
    │   ├── snorkel_collator.py
    │   └── union_collator.py
    ├── context_window_functions
    ├── contextual_functions
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── cwr_linear.py
    │   └── utils.py
    ├── dictionary_functions
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── glove_knn.py
    │   ├── glove_linear.py
    │   ├── keyword_match_function.py
    │   ├── phrase_match_function.py
    │   └── utils.py
    ├── pos_functions
    ├── tree_functions
    ├── types.py
    ├── weak_data.py
    └── weak_function.py
```

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