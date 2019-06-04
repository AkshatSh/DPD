#!/bin/bash
set -e
python dpd/allennlp_active_train.py --num_epochs 10 --model_name cadec/no_weak/trial_0 --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --num_epochs 10 --model_name cadec/no_weak/trial_1 --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --num_epochs 10 --model_name cadec/no_weak/trial_2 --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --num_epochs 10 --model_name cadec/no_weak/trial_3 --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --num_epochs 10 --model_name cadec/no_weak/trial_4 --cuda --cache --sample_strategy top_k
