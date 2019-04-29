from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

from .types import (
    AnnotatedDataType,
)

class BIOConverter(object):
    def __init__(
        self,
        binary_class: str,
    ):
        self.binary_class = binary_class
    
    def convert_to_bio(self, sentence: List[str], predictions: List[str], class_tag: str) -> List[str]:
        proc_pred = list(predictions) # create copy
        for i, (w, p_i) in enumerate(zip(sentence, predictions)):
            if i == 0:
                if p_i == class_tag:
                    proc_pred[i] = f'B-{class_tag}'
                continue
            
            if p_i == class_tag:
                prev_tag = predictions[i - 1]
                if prev_tag == 'O':
                    proc_pred[i] = f'B-{class_tag}'
                else:
                    proc_pred[i] = f'I-{class_tag}'
        return proc_pred
    
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
            data_entry['output'] = self.convert_to_bio(
                data_entry['input'],
                data_entry['output'],
                self.binary_class,
            )
            annotated_data.append(data_entry)
        return annotated_data