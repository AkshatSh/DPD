from typing import (
    List,
    Tuple,
    Dict,
)

import os

from dpd.dataset import BIODataset

# (id, sentence)
QueryType = Tuple[int, List[str]]
# (id, sentence, tags)
QueryResultType = Dict[str, object]

class Oracle(object):
    '''
    Abstract class for an oracle
    '''
    def __init__(
        self,
        train_data: BIODataset,
    ):
        pass
    
    def get_query(self, query: QueryType) -> QueryResultType:
        raise NotImplementedError()