#!/bin/bash
set -e
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.01 --model_name 0.01/linear/trial_0 --use_weak --cuda --use_weak_fine_tune
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.01 --model_name 0.01/knn/trial_0 --use_weak --cuda --use_weak_fine_tune
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.01 --model_name 0.01/keyword/trial_0 --use_weak --cuda --use_weak_fine_tune
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.01 --model_name 0.01/linear/trial_1 --use_weak --cuda --use_weak_fine_tune
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.01 --model_name 0.01/knn/trial_1 --use_weak --cuda --use_weak_fine_tune
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.01 --model_name 0.01/keyword/trial_1 --use_weak --cuda --use_weak_fine_tune
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.01 --model_name 0.01/linear/trial_2 --use_weak --cuda --use_weak_fine_tune
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.01 --model_name 0.01/knn/trial_2 --use_weak --cuda --use_weak_fine_tune
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.01 --model_name 0.01/keyword/trial_2 --use_weak --cuda --use_weak_fine_tune
