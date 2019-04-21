import warnings
from typing import (
    Dict,
    List,
    Union,
    Any,
)

import torch
from torch import nn
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


class CachedTextFieldEmbedder(nn.Module):
    '''
    Given an AllenNLP text field embedder, assumes the embedder to be frozen
    and caches the result so it never needs to be recomputed based on dataset id
    and entry id

    Can also be used for futher analysis of an embedding space through TSNE, PCA, and UMAP Projections
    '''
    pass