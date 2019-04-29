from typing import (
    List,
    Tuple,
    Dict,
    Iterator,
    Optional,
)

import os
import sys
import tqdm as tqdm

import torch
import numpy as np
import allennlp

from dpd.weak_supervision import AnnotatedDataType

class Collator(object):
    def __init__(
        self,
    ):
        pass
    
    def collate(
        self,
        annotations: List[AnnotatedDataType],
    ) -> AnnotatedDataType:
        '''
        args:
            ``annotations``: List[AnnotatedDataType]
                given a series of annotations, collate them into a single
                series of annotations per instance
        '''
        raise NotImplementedError()
    
    @classmethod
    def verify_input(cls, inputs) -> bool:
        for i, inp in enumerate(inputs):
            '''
            verify all inputs are of the same length
            '''
            if i == 0:
                continue
            if inp[i] != inp[i - 1]:
                return False
        return True
    
    @classmethod
    def verify_output(cls, outputs) -> bool:
        '''
        verify all outputs are of the same length
        '''
        for i, out in enumerate(outputs):
            if i == 0:
                continue
            if len(out[i]) != len(out[i - 1]):
                return False
        return True
    
    @classmethod
    def verify_annotations(cls, annotations):
        '''
        Verify all the annotations of the same length
        '''
        for ann_inst in tqdm(zip(*annotations)):
            inputs = [inst['input'] for inst in ann_inst]
            outputs = [inst['output'] for inst in ann_inst]
            assert cls.verify_input(inputs)
            assert cls.verify_output(outputs)
