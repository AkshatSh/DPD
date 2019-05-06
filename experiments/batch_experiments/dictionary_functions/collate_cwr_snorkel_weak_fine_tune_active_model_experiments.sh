#!/bin/bash
set -e
python dpd/allennlp_active_train.py --weak_function cwr_knn cwr_linear linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name cwr_collation/fine_tune/metal/0.01/trial_0 --use_weak --use_weak_fine_tune --cuda --cache --sample_strategy top_k --weak_collator metal
python dpd/allennlp_active_train.py --weak_function cwr_knn cwr_linear linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name cwr_collation/fine_tune/metal/0.01/trial_1 --use_weak --use_weak_fine_tune --cuda --cache --sample_strategy top_k --weak_collator metal
python dpd/allennlp_active_train.py --weak_function cwr_knn cwr_linear linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name cwr_collation/fine_tune/metal/0.01/trial_2 --use_weak --use_weak_fine_tune --cuda --cache --sample_strategy top_k --weak_collator metal
python dpd/allennlp_active_train.py --weak_function cwr_knn cwr_linear linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name cwr_collation/fine_tune/metal/0.01/trial_3 --use_weak --use_weak_fine_tune --cuda --cache --sample_strategy top_k --weak_collator metal
python dpd/allennlp_active_train.py --weak_function cwr_knn cwr_linear linear knn keyword --num_epochs 5 --weak_weight 0.01 --model_name cwr_collation/fine_tune/metal/0.01/trial_4 --use_weak --use_weak_fine_tune --cuda --cache --sample_strategy top_k --weak_collator metal
