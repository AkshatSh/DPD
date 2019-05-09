from typing import (
    List,
    Tuple,
    Dict,
)

from collections import Counter
import numpy as np
import torch
import faiss

from dpd.constants import (
    DEFAULT_GLOVE_DIM,
    SimilarityAlgorithm,
)

from dpd.utils import PickleFaiss

from .glove_utils import (
    load_glove,
    EmbeddingSpaceType,
    EmbeddingType,
)

class GloVeWordEmbeddingIndex(PickleFaiss):
    _INSTANCE = None

    @classmethod
    def instance(cls):
        '''
        maintain singleton object to conserve memory

        all methods are reading, this should be multiprocessing safe
        '''
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(
                embedding_space=load_glove(DEFAULT_GLOVE_DIM),
                embedding_space_dims=DEFAULT_GLOVE_DIM,
                similarity_algorithm=SimilarityAlgorithm.CosineSimilarity,
            )

        return cls._INSTANCE

    '''
    Build a FAISS index for an EmbeddingSpaceType
    object
    
    Its a nice wrapper around the faiss index, to allow easily searching
    and converting vectors to words and vice versa
    '''
    def __init__(
        self,
        embedding_space: EmbeddingSpaceType,
        embedding_space_dims: int,
        similarity_algorithm: SimilarityAlgorithm,
    ):
        super(GloVeWordEmbeddingIndex, self).__init__(
            faiss_index_name='faiss_index',
            index_np_name='index_np',
            embedding_space_dims_name='embedding_space_dims',
            similarity_algorithm_name='similarity_algorithm',
        )

        self.embedding_space = embedding_space
        self.embedding_space_dims = embedding_space_dims
        self.similarity_algorithm = similarity_algorithm
        self.index_np, self.word_to_index, self.index_to_word = (
            GloVeWordEmbeddingIndex.build_index(
                embedding_space,
                embedding_space_dims,
            )
        )
        
        # for FAISS we need float32 instead of float64
        self.index_np = self.index_np.astype('float32')
        
        self.faiss_index = faiss.IndexFlatIP(embedding_space_dims)
        if similarity_algorithm == SimilarityAlgorithm.CosineSimilarity:
            # normalize with L2 as a proxy for cosine search
            faiss.normalize_L2(self.index_np)
        self.faiss_index.add(self.index_np)
    
    def find_similar(
        self,
        query_np: np.ndarray,
        k: int,
        remove_first_row: bool = True,
    ) -> Counter:
        '''
        given a query retreive similar words
        
        input:
            - ``query_np`` np.ndarray
                The query to search the embedding space for
        output:
            - ``Counter``
                The count of kNN results
        '''
        query_np = query_np.astype('float32')
        distances, indexes = self.faiss_index.search(query_np, k)
        
        if remove_first_row:
            first_row = indexes[:, 0]
            similar_words_i = indexes[:, 0:]
        else:
            similar_words_i = indexes
        
        similar_words_i = similar_words_i.flatten()
        
        similar_words = Counter()
        for word_index in similar_words_i:
            similar_words[self.index_to_word[word_index]] += 1
        return similar_words
    
    def _phrase_embedding(
        self,
        phrase: List[str],
    ) -> np.ndarray:
        '''
        compute an embedding representation given a list of words
        input:
            - ``phrase`` List[str]
                
        output:
            - ``np.ndarray``
                the embedding for the phrase
        '''
        phrase_embedding = np.zeros((self.embedding_space_dims,))
        for w in phrase:
            embedding_index = self.get_embedding_index(w)
            if w not in STOP_WORDS and embedding_index > 0:
                continue
            embedding_vec = self.index_np[embedding_index]
            phrase_embedding += embedding_vec
        phrase_embedding /= len(phrase)
        return phrase_embedding
    
    def find_similar_phrases(
        self,
        query: List[List[str]],
        k: int = 5,
    ) -> Counter:
        query_vecs = [self._phrase_embedding(q) for q in query]
        query_np = np.array(query_vecs)
        similar_words = self.find_similar(query_np, k, remove_first_row=False)
        for phrase in query:
            for word in phrase:
                del similar_words[word]
        return similar_words

    def get_embedding(
        self,
        word: str,
    ) -> np.ndarray:
        '''
        Retrieving the embedding for a specific word
        '''
        embedding_i = self.get_embedding_index(word)
        return self.index_np[embedding_i]

    def find_similar_words(
        self,
        query: List[str],
        k: int = 5,
    ) -> Counter:
        '''
        Using the specified search algorithm and the query passed in, the method
        returns similar words
        
        The algorithm builds a query of embedding vectors by retrieving the cached embedding vectors
        Then uses the `similarity search` specified in the constructor to find similary queries
        Finally the indexes are converted to words and a list of query words is retrieved and ranked
        by ocurrence.
        
        input:
            - ``query``: List[str]
                a list of all the query words, this should be the dictionary (or a subset) that
                we are augmenting
            - ``k``: int,
                the number of instances to search over (the k in kNN)
            - ``result_size``: Optional[int]
                if specified will limit the results to be of the result size
        output:
            - ``similar words`` Counter[str]
                a counting occurence of all the words retrieved from the query
        '''
        embedding_indicies_list = list(set([self.get_embedding_index(w) for w in query]))
        embedding_indicies = list(filter(lambda x: x > 0, embedding_indicies_list))
        embedding_indicies = np.array(embedding_indicies)
        
        query_np = self.index_np[embedding_indicies]
        
        similar_words = self.find_similar(query_np, k)
        for word in query:
            del similar_words[word]
        return similar_words
    
    def get_embedding_index(self, word: str) -> np.ndarray:
        if word not in self.word_to_index:
            word = 'UNK'
        return self.word_to_index[word]

    @classmethod
    def build_index(
        cls,
        embedding_space: EmbeddingSpaceType,
        embedding_space_dims: int,
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        '''
        Builds 3 objects specified in the output, meant for searching in the
        embedding space
        
        input:
            - ``embedding_space``: EmbeddingSpaceType
                this is the embedding space mapping keys to embeddings, we use this
                to create a nice wrapper around FAISS to enable fast searching
            - ``embedding_space_dims``: int
                the number of dimensions in each embedding
        output Tuple of 3 object:
            - ``index_np`` np.ndarray
                shape: (len(embedding_space), embedding space dimensions)
                this contains the entire index of the embedding space in a continous numpy
                ndarray for searching
            - ``word_to_index`` Dict[str, int]
                maps each word to the associated index in the index_np
            - ``index_to_word`` Dict[int, str]
                maps each index to the associated word
        '''
        word_to_index: Dict[str, int] = {'UNK': 0}
        index_to_word: Dict[int, str] = {0: 'UNK'}
        for word in embedding_space:
            word_to_index[word] = len(word_to_index)
            index_to_word[
                word_to_index[word]
            ] = word
        
        index_np = np.ndarray((len(word_to_index), embedding_space_dims))

        # first dimension is UNK
        index_np[0] = np.zeros((embedding_space_dims,))
        for word, embedding in embedding_space.items():
            word_i = word_to_index[word]
            index_np[word_i] = embedding
        
        return index_np, word_to_index, index_to_word