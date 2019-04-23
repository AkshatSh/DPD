from typing import (
    List,
    Tuple,
    Dict,
)

import tqdm
import numpy as np
from dpd.constants import GLOVE_FILES

EmbeddingType = np.ndarray
EmbeddingSpaceType = Dict[str, EmbeddingType]

def load_glove(dims: int) -> EmbeddingSpaceType:
    '''
    Given a number of dimensions load the embedding space for the associated GLOVE embedding
    
    Input: ``dims``: int the number of dimensions to use
    
    Output: ``EmbeddingSpace`` EmbeddingSpaceType, the entire embedding space embedded in the file
    '''
    glove_file = GLOVE_FILES[dims]
    with open(glove_file, 'r') as f:
        embedding_space = {}
        for line in tqdm(f):
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            embedding_space[word] = embedding
    return embedding_space