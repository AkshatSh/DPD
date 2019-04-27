import os
import sys
from enum import Enum
import nltk
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

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
CADEC_VALID_ORIGINAL = os.path.join(CADEC_DIR, 'cadec_valid_post_conll.txt')
CADEC_TEST = os.path.join(CADEC_DIR, 'cadec_valid_test_split_post_conll.txt')
CADEC_VALID = os.path.join(CADEC_DIR, 'cadec_valid_valid_split_post_conll.txt')
# CADEC_TEST = os.path.join(CADEC_DIR, 'cadec_valid_post_conll.txt')
# CADEC_VALID = os.path.join(CADEC_DIR, 'cadec_valid_final_post_conll.txt')
# CADEC_TRAIN = os.path.join(CADEC_DIR, 'cadec_train_final_post_conll.txt')

# GLOVE constants
GLOVE_DIMS = [50, 100, 200, 300]
GLOVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'data', 'glove.6B')
def _construct_glove_files(dims: int) -> str:
    return os.path.join(
        GLOVE_DIR,
        f'glove.6B.{dims}d.txt'
    )

# access glove files by GLOVE_FILES[dims]
# e.g. GLOVE_FILES[300] will return the glove file for 300 dimension GLOVE
GLOVE_FILES = {d: _construct_glove_files(d) for d in GLOVE_DIMS}
DEFAULT_GLOVE_DIM = 300

# embedding space similarity algorithm
class SimilarityAlgorithm(Enum):
    L2Distance = 1
    CosineSimilarity = 2

class DictionaryFunctionLinear(Enum):
    LOGISTIC_REGRESSION = 1
    SVM_LINEAR = 2
    SVM_RBF = 3
    SVM_QUADRATIC = 4