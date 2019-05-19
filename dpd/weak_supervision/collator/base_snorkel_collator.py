from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys
import logging

import torch
import allennlp
import numpy as np
import scipy
from scipy import sparse
from tqdm import tqdm

from dpd.utils import TensorList

from ..types import (
    AnnotatedDataType,
    AnnotationType,
)

from .collator import Collator
from ..utils import NEGATIVE_LABEL, ABSTAIN_LABEL, is_negative, is_positive, is_abstain

class BaseSnorkelCollator(Collator):
    def __init__(
        self,
        positive_label: str,
        class_cardinality: int = 2,
        num_epochs: int = 500,
        log_train_every: int = 50,
        seed: int = 123,
    ):
        self.positive_label = positive_label
        self.class_cardinality = class_cardinality
        self.num_epochs = num_epochs
        self.log_train_every = log_train_every
        self.seed = seed
        self.label_model = LabelModel(k=self.class_cardinality, seed=seed)
    
    @classmethod
    def get_snorkel_index(cls, tag: str) -> int:
        if is_positive(tag):
            return 2
        elif is_negative(tag):
            return 1
        else:
            return 0

    def get_tag(self, index: int) -> str:
        if index == 1:
            return self.positive_label
        else:
            return NEGATIVE_LABEL
    
    def get_index(self, prob: np.ndarray) -> str:
        assert prob.shape == (2,)
        return prob.argmax()
    
    def collate_np(self, annotations) -> Tuple[np.ndarray, List[str], List[int]]:
        output_arrs: List[np.ndarray] = []
        words_list: List[str] = []
        id_to_labels: Dict[int, Tuple[int, int]] = {}
        num_funcs = len(annotations)
        for i, ann_inst in tqdm(enumerate(zip(*annotations))):
            ids = [inst['id'] for inst in ann_inst]
            inputs = [inst['input'] for inst in ann_inst]
            outputs = [inst['output'] for inst in ann_inst]
            input_len = len(inputs[0])
            entry_id = ids[0]

            # output arr = (sentence x num_labels)
            output_arr = np.zeros((input_len, num_funcs))
            for i, output in enumerate(outputs):
                for j, out_j in enumerate(output):
                    output_arr[j, i] = BaseSnorkelCollator.get_snorkel_index(out_j)
            
            label_start = len(words_list)
            for word_i, word in enumerate(inputs[0]):
                words_list.append(word)
            output_arrs.append(output_arr)
            label_end = len(words_list)
            id_to_labels[entry_id] = (label_start, label_end)
        output_res = np.concatenate(output_arrs, axis=0)
        return output_res, words_list, id_to_labels
    
    def train_label_model(
        self,
        collated_labels: np.ndarray,
        descriptions: Optional[List[str]],
        train_data_np: Optional[np.ndarray],
    ):
        raise NotImplementedError()
    
    def get_probabilistic_labels(self, collated_labels: np.ndarray) -> np.ndarray:
        sparse_labels = sparse.csr_matrix(collated_labels)
        return self.label_model.predict_proba(sparse_labels)

    def convert_to_tags(
        self,
        train_probs: np.ndarray,
        word_list: List[str],
        id_to_labels: Dict[int, Tuple[int, int]],
    ) -> List[AnnotatedDataType]:
        output = []
        for entry_id, (label_start, label_end) in id_to_labels.items():
            words = word_list[label_start:label_end]
            prob_labels = train_probs[label_start:label_end]
            label_ids = prob_labels.argmax(axis=1)
            labels = [self.get_tag(i) for i in label_ids]
            output.append({
                'id': entry_id,
                'input': words,
                'output': labels,
            })
        return output
    
    def collate(
        self,
        annotations: List[AnnotatedDataType],
        should_verify: bool = False,
        descriptions: Optional[List[str]] = None,
        train_data: Optional[AnnotatedDataType] = None
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
        
        train_data_np = None
        if train_data:
            # if train data specified, will be used by Snorkel to estimate class balanc
            train_data_np, word_lists, id_to_labels = self.collate_np([train_data])
            train_data_np = train_data_np.astype(int)
            train_data_np = train_data_np.reshape(-1)
        collate_np, word_lists, id_to_labels = self.collate_np(annotations)
        self.train_label_model(collated_labels=collate_np, descriptions=descriptions, train_data_np=train_data_np)
        y_train_probs = self.get_probabilistic_labels(collated_labels=collate_np,)
        tags = self.convert_to_tags(y_train_probs, word_list=word_lists, id_to_labels=id_to_labels)
        return tags