from typing import (
    List,
)

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from .allennlp_models import (
    ELMoCrfTagger,
    BERTCrfTagger,
    ELMoLinearTagger,
    ELMoLinearTransformer,
    ELMoCRFTransformer,
    ELMoRNNMTL,
)

from .multitask_tagger import MultiTaskTagger

from .statistical_models import (
    construct_linear_classifier,
    LinearType,
)

def build_model(
    model_type: str,
    vocab: Vocabulary,
    hidden_dim: int,
    class_labels: List[str],
    cached: bool,
) -> Model:
    model_kwargs = dict(
        vocab=vocab,
        hidden_dim=hidden_dim,
        class_labels=class_labels,
        cached=cached,
    )

    if model_type == 'ELMo_bilstm_crf':
        return ELMoCrfTagger(**model_kwargs)
    elif model_type == 'BERT_bilstm_crf':
        return BERTCrfTagger(**model_kwargs)
    elif model_type == 'ELMo_linear':
        return ELMoLinearTagger(**model_kwargs)
    elif model_type == 'ELMo_linear_transformer':
        return ELMoLinearTransformer(**model_kwargs)
    elif model_type == 'ELMo_crf_transformer':
        return ELMoCRFTransformer(**model_kwargs)
    elif model_type == 'ELMo_rnn_mtl':
        return ELMoRNNMTL(**model_kwargs)
    else:
        raise Exception(f'Unknown model type {model_type}')