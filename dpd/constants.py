import os
import sys

# Special Tokens 
UNKNOWN_TOKEN = '<UNK>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
SPECIAL_TOKENS = [UNKNOWN_TOKEN, START_TOKEN, END_TOKEN, PAD_TOKEN]

# Conll2003 Dataset Paths
CONLL2003_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'conll2003', 'en/')
CONLL2003_TRAIN = os.path.join(CONLL2003_DIR, 'train.txt')
CONLL2003_TEST = os.path.join(CONLL2003_DIR, 'test.txt')
CONLL2003_VALID = os.path.join(CONLL2003_DIR, 'valid.txt')


# CADEC Dataset Paths
CADEC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cadec')
CADEC_TRAIN = os.path.join(CADEC_DIR, 'cadec_train_post_conll.txt')
CADEC_VALID = os.path.join(CADEC_DIR, 'cadec_valid_post_conll.txt')