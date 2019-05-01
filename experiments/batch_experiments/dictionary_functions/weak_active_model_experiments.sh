#!/bin/bash
set -e
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/linear/trial_0 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/knn/trial_0 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/keyword/trial_0 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/linear/trial_1 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/knn/trial_1 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/keyword/trial_1 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/linear/trial_2 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/knn/trial_2 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/keyword/trial_2 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/linear/trial_3 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/knn/trial_3 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/keyword/trial_3 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function linear --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/linear/trial_4 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function knn --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/knn/trial_4 --use_weak --cuda --cache --sample_strategy top_k
python dpd/allennlp_active_train.py --weak_function keyword --num_epochs 5 --weak_weight 0.01 --model_name cached/weighted/0.01/keyword/trial_4 --use_weak --cuda --cache --sample_strategy top_k
