from enum import Enum

from .keyword_match_function import KeywordMatchFunction
from .glove_knn import GlovekNNFunction
from .glove_linear import GloveLinearFunction
from .phrase_match_function import PhraseMatchFunction

DICTIONARY_FUNCTION_IMPL = {
    'keyword': KeywordMatchFunction,
    'knn': GlovekNNFunction,
    'linear': GloveLinearFunction,
}