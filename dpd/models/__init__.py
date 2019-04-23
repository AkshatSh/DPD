from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from .allennlp_crf import ELMoCrfTagger

def build_model(
    model_type: str,
    vocab: Vocabulary,
    hidden_dim: int,
    class_labels: List[str],
) -> Model:
    if model_type == 'ELMo_lstm_crf':
        return ELMoCrfTagger(
            vocab=vocab,
            hidden_dim=hidden_dim,
            class_labels=class_labels,
        )
    else:
        raise Exception(f'Unknown model type {model_type}')