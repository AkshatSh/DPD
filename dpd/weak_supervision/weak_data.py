from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Optional,
)

import logging

from allennlp.models import Model
from allennlp.data import Vocabulary

from dpd.dataset import UnlabeledBIODataset
from dpd.models.embedder import CachedTextFieldEmbedder
from dpd.weak_supervision import WeakFunction, BIOConverter
from dpd.weak_supervision.dictionary_functions import (
    KeywordMatchFunction,
    GlovekNNFunction,
    GloveLinearFunction,
    DICTIONARY_FUNCTION_IMPL,
)

from dpd.weak_supervision.contextual_functions import (
    CONTEXTUAL_FUNCTIONS_IMPL,
)

from dpd.weak_supervision.collator import (
    COLLATOR_IMPLEMENTATION,
)

from dpd.weak_supervision.context_window_functions import (
    WINDOW_FUNCITON_IMPL
)

from dpd.weak_supervision.feature_extractor import FEATURE_EXTRACTOR_IMPL, FeatureCollator

from dpd.weak_supervision.types import (
    EntryDataType,
    DatasetType,
)

def build_weak_data(
    train_data: DatasetType,
    unlabeled_corpus: UnlabeledBIODataset,
    model: Model,
    vocab: Vocabulary,
    weight: float = 1.0,
    function_types: List[str] = ['linear'],
    collator_type: str = 'union',
    contextual_word_embeddings: Optional[List[CachedTextFieldEmbedder]] = None,
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
    dict_functions: List[WeakFunction] = []
    cwr_functions: List[WeakFunction] = []
    window_functions: List[WeakFunction] = []
    for f in function_types:
        if f in DICTIONARY_FUNCTION_IMPL:
            dict_functions.append(DICTIONARY_FUNCTION_IMPL[f](unlabeled_corpus.binary_class))
        elif f in CONTEXTUAL_FUNCTIONS_IMPL and contextual_word_embeddings is not None:
            for contextual_word_embedding in contextual_word_embeddings:
                cwr_functions.append(CONTEXTUAL_FUNCTIONS_IMPL[f](unlabeled_corpus.binary_class, contextual_word_embedding))
        elif f.startswith('context_window'):
            # format
            # context_window-{window}-{extractor}-{collator}
            window, extractor, collator = f.split('-')[1:]
            window = int(window)
            for constructor in WINDOW_FUNCITON_IMPL.values():
                if extractor == 'cwr':
                    for embedder in contextual_word_embeddings:
                        window_functions.append(
                            constructor(
                                positive_label=unlabeled_corpus.binary_class,
                                context_window=window,
                                feature_extractor=FEATURE_EXTRACTOR_IMPL[extractor](vocab=vocab, embedder=embedder),
                                feature_summarizer=FeatureCollator.get(collator),
                            )
                        )
                else:
                    window_functions.append(
                        constructor(
                            positive_label=unlabeled_corpus.binary_class,
                            context_window=window,
                            feature_extractor=FEATURE_EXTRACTOR_IMPL[extractor](vocab=vocab),
                            feature_summarizer=FeatureCollator.get(collator),
                        )
                    )

    collator = COLLATOR_IMPLEMENTATION[collator_type](positive_label=unlabeled_corpus.binary_class)
    bio_converter = BIOConverter(binary_class=unlabeled_corpus.binary_class)
    print(f'using {len(dict_functions) + len(cwr_functions) + len(window_functions)} weak functions ({len(dict_functions)} dict, {len(cwr_functions)} cwr, {len(window_functions)} window)')
    logging.info(f'using weak functions: {dict_functions}')
    annotated_corpi = []

    basic_functions: List[WeakFunction] = dict_functions + window_functions

    for function in basic_functions:
        # contextual word representation functions
        function.train(train_data)
        annotated_corpus = function.evaluate(unlabeled_corpus)
        annotated_corpi.append(annotated_corpus)
    
    for function in cwr_functions:
        # Contextual word representation functions
        function.train(train_data, unlabeled_corpus.dataset_id)
        annotated_corpus = function.evaluate(unlabeled_corpus)
        annotated_corpi.append(annotated_corpus)

    fin_annotated_corpus = collator.collate(annotated_corpi)
    bio_corpus = bio_converter.convert(fin_annotated_corpus)
    for i, item in enumerate(bio_corpus):
        item['weight'] = weight
    return bio_corpus