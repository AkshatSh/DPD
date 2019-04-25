from typing import (
    List,
    Tuple,
    Dict,
)

import tqdm
import numpy as np
from tqdm import tqdm
from dpd.constants import GLOVE_FILES
from dpd.utils import get_num_lines

EmbeddingType = np.ndarray
EmbeddingSpaceType = Dict[str, EmbeddingType]

def load_glove(dims: int) -> EmbeddingSpaceType:
    '''
    Given a number of dimensions load the embedding space for the associated GLOVE embedding
    
    Input: ``dims``: int the number of dimensions to use
    
    Output: ``EmbeddingSpace`` EmbeddingSpaceType, the entire embedding space embedded in the file
    '''
    glove_file = GLOVE_FILES[dims]
    embedding_space = {}
    file_len = get_num_lines(glove_file)
    with open(glove_file, 'r') as f:
        for line in tqdm(f, total=file_len):
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            embedding_space[word] = embedding
    return embedding_space