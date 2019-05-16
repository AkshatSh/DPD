from typing import (
    List,
    Tuple,
    Dict,
    Set,
)

import os
import random

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy

from dpd.dataset import UnlabeledBIODataset
from dpd.models.embedder import CachedTextFieldEmbedder, SentenceEmbedder

class ClusteringHeuristic(object):
    def __init__(
        self,
        cwr: CachedTextFieldEmbedder,
        dataset: UnlabeledBIODataset,
        *args,
        **kwargs,
    ):
        self.cwr = cwr
        self.sentence_embedder = SentenceEmbedder(cwr)
    
    def evaluate(
        self,
        unlabeled_corpus: UnlabeledBIODataset,
        sample_size: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        '''
        evaluate the random heuristic on every item and return the
        weights associated with the unlabeled corpus

        input:
            ``unlabeled_corpus`` UnlabeledBIODataset
                the unlabeled corpus to evaluate this heuristic on
        output:
            ``torch.Tensor``
                get the weighted unlabeled corpus
        '''
        index = SentenceEmbedder.build_index(self.sentence_embedder, unlabeled_corpus)
        index_np: np.ndarray = index.numpy()
        model = AgglomerativeClustering(linkage="average", affinity="cosine", n_clusters=sample_size)
        model.fit(index_np)
        distr = torch.zeros((len(unlabeled_corpus),))
        for i in range(sample_size):
            cluster = list(filter(lambda item: item[1] == i, enumerate(model.labels_)))
            index, cluster_label = random.sample(cluster, 1)[0]
            distr[index] = 1.0

        return F.softmax(distr, dim=0)
