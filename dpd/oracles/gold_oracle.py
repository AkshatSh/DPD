from typing import (
    List,
    Tuple,
    Dict,
)

import os

from .oracle import (
    Oracle,
    QueryType,
    QueryResultType,
)

from dpd.dataset import BIODataset

class GoldOracle(Oracle):
    def __init__(
        self,
        train_data: BIODataset,
    ):
        self.train_data = train_data
        self.lookup : Dict[str, Tuple[List[str], List[str]]] = {}
        for entry in self.train_data:
            s_id, s_input, s_output = entry['id'], entry['input'], entry['output']
            self.lookup[s_id] = (s_input, s_output)
    
    def get_query(self, query: QueryType) -> QueryResultType:
        s_id, sentence = query
        query_input, query_tags = self.lookup[s_id]

        # verify the correct thing was retrieved
        assert query_input == sentence

        return {
            'id': s_id,
            'input': sentence,
            'output': query_tags,
            'weight': 1.0,
        }
