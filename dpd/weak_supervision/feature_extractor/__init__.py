from .feature_extractor import FeatureExtractor
from .spacy_feature_extractor import SpaCyFeatureExtractor
from .feature_utils import FeatureCollator, FeaturePadder
from .word_feature_extractor import WordFeatureExtractor
from .glove_feature_extractor import GloVeFeatureExtractor
from .cached_cwr_extractor import CWRFeatureExtractor

FEATURE_EXTRACTOR_IMPL = {
    'word': WordFeatureExtractor,
    'glove': GloVeFeatureExtractor,
    'cwr': CWRFeatureExtractor,
}