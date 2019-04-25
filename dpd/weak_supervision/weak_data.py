from typing import (
    List,
    Dict,
    Tuple,
)

from allennlp.models import Model

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction
from dpd.weak_supervision.dictionary_functions import (
    KeywordMatchFunction,
)

EntryDataType = Dict[str, object]
DatasetType = List[EntryDataType]

def build_weak_data(
    train_data: DatasetType,
    unlabeled_corpus: UnlabeledBIODataset,
    model: Model,
    weight: float = 1.0,
) -> DatasetType:
    '''
    This constructs a weak dataset

    input:
        - ``train_data`` DatasetType
            the training data along with annotations that we can use
            to generate our heuristics
        -  ``unlabeled_corpus`` UnlabeldBIODataset
            the unlabeled corpus to run these heuristics on
        - ``model`` Model
            the model that can be used to label this corpus
    output:
        ``DatasetType``
            the weak dataset that can be used along side training
    '''
    function: WeakFunction = KeywordMatchFunction(unlabeled_corpus.binary_class)
    function.train(train_data)
    annotated_corpus = function.evaluate(unlabeled_corpus)
    for item in annotated_corpus:
        item['weight'] = weight
    return annotated_corpus