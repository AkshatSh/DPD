#!/bin/bash
set -e
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.1 --model_name 0.1/linear/trial_0 --use_weak --cuda
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.1 --model_name 0.1/knn/trial_0 --use_weak --cuda
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.1 --model_name 0.1/keyword/trial_0 --use_weak --cuda
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.1 --model_name 0.1/linear/trial_1 --use_weak --cuda
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.1 --model_name 0.1/knn/trial_1 --use_weak --cuda
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.1 --model_name 0.1/keyword/trial_1 --use_weak --cuda
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.1 --model_name 0.1/linear/trial_2 --use_weak --cuda
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.1 --model_name 0.1/knn/trial_2 --use_weak --cuda
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.1 --model_name 0.1/keyword/trial_2 --use_weak --cuda
