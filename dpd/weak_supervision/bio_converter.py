from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys
import numpy as np

from dpd.constants import STOP_WORDS

from .utils import is_negative, is_positive

from .types import (
    AnnotatedDataType,
)

class BIOConverter(object):
    def __init__(
        self,
        binary_class: str,
    ):
        self.binary_class = binary_class
    
    @classmethod
    def get_prob_labels(
        cls,
        predictions: List[str],
        probabilities: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        if probabilities is not None:
            return probabilities
        
        def _get_probability(label: str) -> np.ndarray:
            arr = np.zeros((1,2))
            if is_negative(label):
                arr[:, 0] = 1.
            else:
                arr[:, 1] = 1.
            return arr
        
        return list(map(_get_probability, predictions))
    
    def stop_word_heuristic(
        self,
        sentence: List[str],
        predictions: List[str],
        class_tag: str,
        probabilities: Optional[List[np.ndarray]] = None,
    ) -> List[str]:
        probabilities = BIOConverter.get_prob_labels(predictions, probabilities)
        proc_pred = list(predictions) # create copy
        start_stop_words = None
        contains_stop_word = False
        stop_word_ranges: List[Tuple[int, int]] = []
        probability_label: List[Optional[np.ndarray]] = []
        prob_label: Optional[np.ndarray] = None
        for i, (w, p_i) in enumerate(zip(sentence, predictions)):
            if p_i == class_tag:
                if start_stop_words is None:
                    start_stop_words = i
                contains_stop_word = True
                prob_label = None if probabilities is None else probabilities[i]
            elif w in STOP_WORDS and start_stop_words is None:
                start_stop_words = i
            elif w not in STOP_WORDS and start_stop_words is not None:
                if contains_stop_word:
                    stop_word_ranges.append((start_stop_words, i))
                    probability_label.append(prob_label)
                start_stop_words = None
                contains_stop_word = False
        if contains_stop_word and start_stop_words is not None:
            stop_word_ranges.append((start_stop_words, len(predictions)))
        for (start_pos, end_pos) in stop_word_ranges:
            for i in range(start_pos, end_pos):
                proc_pred[i] = class_tag
        return proc_pred
    
    def convert_to_bio(
        self,
        sentence: List[str],
        predictions: List[str],
        class_tag: str,
        probabilities: List[np.ndarray],
    ) -> Tuple[List[str], Optional[List[np.ndarray]]]:
        proc_pred = list(predictions) # create copy
        proc_probs = list(probabilities) # create copy
        for i, (w, p_i, prob_i) in enumerate(zip(sentence, predictions, probabilities)):
            # 0 => O, 1 => B-ADR, 2 => I-ADR
            new_prob = np.zeros((1,3))
            new_prob[:, 0], new_prob[:, 1] = prob_i[:, 0], prob_i[:, 1]
            if i == 0:
                if p_i == class_tag:
                    proc_pred[i] = f'B-{class_tag}'
            elif p_i == class_tag:
                prev_tag = predictions[i - 1]
                if prev_tag == 'O':
                    proc_pred[i] = f'B-{class_tag}'
                else:
                    proc_pred[i] = f'I-{class_tag}'
                    new_prob[:, 2] = new_prob[:, 1]
                    new_prob[:, 1] = 0.
            proc_probs[i] = new_prob
        return proc_pred, proc_probs
    
    def convert(
        self,
        annotated_corpus: AnnotatedDataType,
    ) -> AnnotatedDataType:
        '''
        convert the annotations to BIO

        input:
            - annotated_corpus ``AnnotatedDataType``
                the annotated corpus to be converted to BIO encoding
        output:
            - annotations are in BIO format
        '''
        annotated_data = []
        for data_entry in annotated_corpus:
            data_entry = data_entry.copy()
            heuristic_output = data_entry['output']
            prob_labels = BIOConverter.get_prob_labels(
                predictions=heuristic_output,
                probabilities=data_entry['prob_labels'] if 'prob_labels' in data_entry else None,
            )

            # heuristic_output = self.stop_word_heuristic(
            #     sentence=data_entry['input'],
            #     predictions=heuristic_output,
            #     class_tag=self.binary_class,
            #     probabilities=prob_labels,
            # )

            data_entry['output'], data_entry['prob_labels'] = self.convert_to_bio(
                data_entry['input'],
                heuristic_output,
                self.binary_class,
                probabilities=prob_labels,
            )

            annotated_data.append(data_entry)
        return annotated_data