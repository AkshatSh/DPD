from typing import (
    List,
    Tuple,
    Dict,
    Iterator,
    Optional,
)

import os
import sys
from tqdm import tqdm

import torch
import numpy as np
import allennlp

from dpd.weak_supervision import AnnotatedDataType

class Collator(object):
    def __init__(
        self,
    ):
        pass
    
    def _collate_fn(self, outputs: List[List[str]]) -> List[str]:
        raise NotImplementedError()

    def collate(
        self,
        annotations: List[AnnotatedDataType],
        should_verify: bool = False,
    ) -> AnnotatedDataType:
        '''
        args:
            ``annotations``: List[AnnotatedDataType]
                given a series of annotations, collate them into a single
                series of annotations per instance
        '''
        if should_verify:
            # make sure the annotations are in the
            # proper format
            Collator.verify_annotations(annotations)

        output = []
        for i, ann_inst in tqdm(enumerate(zip(*annotations))):
            inputs = [inst['input'] for inst in ann_inst]
            outputs = [inst['output'] for inst in ann_inst]

            # gather one of the inputs
            coll_output = self._collate_fn(outputs)
            new_entry = {'input': inputs[0], 'output': coll_output}
            output.append(new_entry)
        return output
    
    @classmethod
    def verify_input(cls, inputs: List[List[str]]) -> bool:
        for i, inp in enumerate(inputs):
            '''
            verify all inputs are of the same length
            '''
            if i == 0:
                continue
            if inputs[i] != inputs[i - 1]:
                return False
        return True
    
    @classmethod
    def verify_output(cls, outputs: List[List[str]]) -> bool:
        '''
        verify all outputs are of the same length
        '''
        for i, out in enumerate(outputs):
            if i == 0:
                continue
            if len(outputs[i]) != len(outputs[i - 1]):
                return False
        return True
    
    @classmethod
    def verify_annotations(cls, annotations: List[AnnotatedDataType]):
        '''
        Verify all the annotations of the same length
        '''
        for ann_inst in tqdm(zip(*annotations)):
            inputs = [inst['input'] for inst in ann_inst]
            outputs = [inst['output'] for inst in ann_inst]
            assert cls.verify_input(inputs)
            assert cls.verify_output(outputs)
