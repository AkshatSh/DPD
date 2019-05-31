import os
import sys
from enum import Enum
import spacy
from spacy.tokens import Doc
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

# SAVE DIR
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', '.saved')
CADEC_ELMo = os.path.join(SAVE_DIR, 'cadec_elmo.tmp')
CADEC_NER_ELMo = os.path.join(SAVE_DIR, 'cadec_ner_elmo.tmp')
CADEC_BERT = os.path.join(SAVE_DIR, 'cadec_bert.tmp')
CONLL_NER_ELMo = os.path.join(SAVE_DIR, 'conll_ner_elmo.tmp')
CONLL_ELMo = os.path.join(SAVE_DIR, 'conll_elmo.tmp')
CONLL_BERT = os.path.join(SAVE_DIR, 'conll_bert.tmp')

ELMo_file = dict(
    cadec=CADEC_ELMo,
    conll=CONLL_ELMo,
)

NER_ELMo_file = dict(
    cadec=CADEC_NER_ELMo,
    conll=CONLL_NER_ELMo,
)

BERT_file = dict(
    cadec=CADEC_BERT,
    conll=CONLL_BERT,
)

# ELMo constants
ELMO_OPTIONS_FILE = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
ELMO_WEIGHT_FILE = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'


# spaCy constants
SPACY_NLP = spacy.load('en_core_web_sm')
CADEC_SPACY = os.path.join(SAVE_DIR, 'cadec_spacy.tmp')
CONLL_SPACY = os.path.join(SAVE_DIR, 'conll_spacy.tmp')
SPACY_POS = ['UNK', 'ADJ','ADP','PUNCT','ADV','AUX','SYM','INTJ','CCONJ','X','NOUN','DET','PROPN','NUM','VERB','PART','PRON','SCONJ',]
SPACY_POS_INDEX = {pos: i for i, pos in enumerate(SPACY_POS)}

SPACY_file = dict(
    cadec=CADEC_SPACY,
    conll=CONLL_SPACY,
)

# if set to true, expects list strs as input to NLP in spacy
USE_TOKEN_TOKENIZER = True
def token_tokenizer(tokens):
    return Doc(SPACY_NLP.vocab, tokens)

if USE_TOKEN_TOKENIZER:
    SPACY_NLP.tokenizer = token_tokenizer

# MEMORY MANAGEMENT
MAX_RETRY: int = 3
MEMORY_WAIT: float = 7.