#!/bin/bash
set -e
python dpd/allennlp_active_train.py --num_epochs 5 --model_name cached/no_weak/trial_0 --cuda --cache
python dpd/allennlp_active_train.py --num_epochs 5 --model_name cached/no_weak/trial_1 --cuda --cache
python dpd/allennlp_active_train.py --num_epochs 5 --model_name cached/no_weak/trial_2 --cuda --cache
python dpd/allennlp_active_train.py --num_epochs 5 --model_name cached/no_weak/trial_3 --cuda --cache
python dpd/allennlp_active_train.py --num_epochs 5 --model_name cached/no_weak/trial_4 --cuda --cache
