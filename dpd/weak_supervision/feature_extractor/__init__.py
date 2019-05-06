from .feature_extractor import FeatureExtractor
from .spacy_feature_extractor import SpaCyFeatureExtractor
from .feature_utils import FeatureCollator, FeaturePadder
from .word_feature_extractor import WordFeatureExtractor

FEATURE_EXTRACTOR_IMPL = {
    'word': WordFeatureExtractor,
}