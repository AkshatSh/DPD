#!/bin/bash
set -e
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/union/0.01/trial_0 --use_weak --cuda --cache --weak_collator union
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/intersection/0.01/trial_0 --use_weak --cuda --cache --weak_collator intersection
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/union/0.01/trial_1 --use_weak --cuda --cache --weak_collator union
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/intersection/0.01/trial_1 --use_weak --cuda --cache --weak_collator intersection
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/union/0.01/trial_2 --use_weak --cuda --cache --weak_collator union
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/intersection/0.01/trial_2 --use_weak --cuda --cache --weak_collator intersection
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/union/0.01/trial_3 --use_weak --cuda --cache --weak_collator union
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/intersection/0.01/trial_3 --use_weak --cuda --cache --weak_collator intersection
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/union/0.01/trial_4 --use_weak --cuda --cache --weak_collator union
python dpd/allennlp_active_train.py --weak_function linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name collation/weighted/intersection/0.01/trial_4 --use_weak --cuda --cache --weak_collator intersection
