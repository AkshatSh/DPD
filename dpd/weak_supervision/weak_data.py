from typing import (
    List,
    Dict,
    Tuple,
    Union,
)

import logging

from allennlp.models import Model

from dpd.dataset import UnlabeledBIODataset
from dpd.weak_supervision import WeakFunction, BIOConverter
from dpd.weak_supervision.dictionary_functions import (
    KeywordMatchFunction,
    GlovekNNFunction,
    GloveLinearFunction,
    DICTIONARY_FUNCTION_IMPL,
)

from dpd.weak_supervision.collator import (
    COLLATOR_IMPLEMENTATION,
)

from dpd.weak_supervision.types import (
    EntryDataType,
    DatasetType,
)

def build_weak_data(
    train_data: DatasetType,
    unlabeled_corpus: UnlabeledBIODataset,
    model: Model,
    weight: float = 1.0,
    function_types: List[str] = ['linear'],
    collator_type: str = 'union',
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
        - ``weight`` float
            the weight to give each instance during training
        - ``function_type`` str
            the type of weak function to use
    output:
        ``DatasetType``
            the weak dataset that can be used along side training
    '''
    functions: List[WeakFunction] = [DICTIONARY_FUNCTION_IMPL[f](unlabeled_corpus.binary_class) for f in function_types]
    collator = COLLATOR_IMPLEMENTATION[collator_type](positive_label=unlabeled_corpus.binary_class)
    bio_converter = BIOConverter(binary_class=unlabeled_corpus.binary_class)
    print(f'using weak functions: {functions}')
    logging.info(f'using weak functions: {functions}')
    annotated_corpi = []
    for function in functions:
        function.train(train_data)
        annotated_corpus = function.evaluate(unlabeled_corpus)
        annotated_corpi.append(annotated_corpus)
    fin_annotated_corpus = collator.collate(annotated_corpi)
    bio_corpus = bio_converter.convert(fin_annotated_corpus)
    for i, item in enumerate(bio_corpus):
        item['weight'] = weight
    return bio_corpus