from typing import (
    List,
)

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from .allennlp_models import (
    ELMoCrfTagger,
    BERTCrfTagger,
    ELMoLinearTagger,
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
    if model_type == 'ELMo_bilstm_crf':
        return ELMoCrfTagger(
            vocab=vocab,
            hidden_dim=hidden_dim,
            class_labels=class_labels,
            cached=cached,
        )
    elif model_type == 'BERT_bilstm_crf':
        return BERTCrfTagger(
            vocab=vocab,
            hidden_dim=hidden_dim,
            class_labels=class_labels,
            cached=cached,
        )
    elif model_type == 'ELMo_linear':
        return ELMoLinearTagger(
            vocab=vocab,
            hidden_dim=hidden_dim,
            class_labels=class_labels,
            cached=cached,
        )
    else:
        raise Exception(f'Unknown model type {model_type}')