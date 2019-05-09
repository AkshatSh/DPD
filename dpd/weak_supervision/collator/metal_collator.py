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
from metal.label_model import LabelModel
from metal.analysis import lf_summary
from tqdm import tqdm

from ..types import (
    AnnotatedDataType,
    AnnotationType,
)

from .collator import Collator
from .collate_utils import bio_negative, bio_positive, NEGATIVE_LABEL

logger = logging.getLogger(name=__name__)

class SnorkeMeTalCollator(Collator):
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
        if bio_positive(tag):
            return 2
        else:
            return 1

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
                    output_arr[j, i] = SnorkeMeTalCollator.get_snorkel_index(out_j)
            
            label_start = len(words_list)
            for word_i, word in enumerate(inputs[0]):
                words_list.append(word)
            output_arrs.append(output_arr)
            label_end = len(words_list)
            id_to_labels[entry_id] = (label_start, label_end)
        output_res = np.concatenate(output_arrs, axis=0)
        return output_res, words_list, id_to_labels
    
    def train_label_model(self, collated_labels: np.ndarray):
        sparse_labels = sparse.csr_matrix(collated_labels)
        logger.debug(lf_summary(sparse_labels))
        self.label_model.train_model(
            sparse_labels,
            n_epochs=self.num_epochs,
            log_train_every=self.log_train_every,
        )
    
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
        collate_np, word_lists, id_to_labels = self.collate_np(annotations)
        self.train_label_model(collated_labels=collate_np)
        y_train_probs = self.get_probabilistic_labels(collated_labels=collate_np,)
        tags = self.convert_to_tags(y_train_probs, word_list=word_lists, id_to_labels=id_to_labels)
        return tags