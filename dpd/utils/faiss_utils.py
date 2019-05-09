from typing import (
    Optional,
)

import os
import sys
import faiss

from dpd.constants import SimilarityAlgorithm

class PickleFaiss(object):
    def __init__(
        self,
        faiss_index_name: Optional[str] = 'faiss_index',
        index_np_name: Optional[str] = 'index_np',
        embedding_space_dims_name: Optional[str] = 'embedding_space_dims',
        similarity_algorithm_name: Optional[str] = 'similarity_algorithm'
    ):
        self.faiss_index_name = faiss_index_name
        self.index_np_name = index_np_name
        self.embedding_space_dims_name = embedding_space_dims_name
        self.similarity_algorithm_name = similarity_algorithm_name
    
    '''
    FAISS index is not picklable, so lets overrwrite it with a picklable state
    '''
    def __getstate__(self):
        state = self.__dict__.copy()
        state[state['faiss_index_name']] = None
        return state

    def __setstate__(self, newstate):
        embedding_space_dims = newstate[newstate['embedding_space_dims_name']]
        similarity_algorithm = newstate[newstate['similarity_algorithm_name']]
        index_np = newstate[newstate['index_np_name']]
        faiss_index = faiss.IndexFlatIP(embedding_space_dims)
        if similarity_algorithm == SimilarityAlgorithm.CosineSimilarity:
            # normalize with L2 as a proxy for cosine search
            faiss.normalize_L2(index_np)
        faiss_index.add(index_np)
        newstate[newstate['faiss_index_name']] = faiss_index
        self.__dict__.update(newstate)