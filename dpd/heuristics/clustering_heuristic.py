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
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster.hierarchy

from dpd.dataset import UnlabeledBIODataset
from dpd.models.embedder import CachedTextFieldEmbedder, SentenceEmbedder
from dpd.utils import TensorList

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

    @classmethod
    def get_agg_clusters(index: TensorList, n_cluster: int) -> AgglomerativeClustering:
        '''
        Runs agglomerative clustering on the `index` TensorList passed in
        '''
        index_np: np.ndarray = index.numpy()
        model = AgglomerativeClustering(linkage="average", affinity="cosine", n_clusters=n_cluster, compute_full_tree=True)
        model.fit(index_np)
        return model
    
    @classmethod
    def get_kmeans_clusters(cls, index: TensorList, n_cluster: int) -> KMeans:
        '''
        Runs kmeans clustering on the `index` TensorList passed in
        '''
        index_np: np.ndarray = index.numpy()
        model = KMeans(n_clusters=n_cluster, random_state=0)
        model.fit(index_np)
        return model

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
        index: TensorList = SentenceEmbedder.build_index(self.sentence_embedder, unlabeled_corpus)
        model = ClusteringHeuristic.get_kmeans_clusters(index, n_cluster=sample_size)
        distr = torch.zeros((len(unlabeled_corpus),))
        for i in range(sample_size):
            cluster = list(filter(lambda item: item[1] == i, enumerate(model.labels_)))
            index, cluster_label = random.sample(cluster, 1)[0]
            distr[index] = 1.0

        return F.softmax(distr, dim=0)
