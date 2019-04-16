from typing import (
    List,
    Tuple,
    Dict,
    Callable,
)

from .dataset.bio_dataset import (
    BIODataset,
)

OracleQueryType = Tuple[int, List[str]]

class Oracle(object):
    '''
    An abstract class for the Oracle interface
    '''
    def __init__(self):
        raise NotImplementedError()

    def get_label(self, inp: OracleQueryType):
        raise NotImplementedError()

class SimulatedOracle(Oracle):
    '''
    Consturcts a simulated oracle that always returns the ground truth label
    '''
    def __init__(
        self, 
        dataset: BIODataset,
    ):
        self.ground_truth = dataset
    
    def get_label(self, inp: OracleQueryType):
        i, input_string = inp
        oracle_answer = self.ground_truth.data[i]
        s_id, truth_sentence, truth_output = oracle_answer['id'], oracle_answer['input'], oracle_answer['output']

        # sanity check to make sure 
        if (input_string != truth_sentence):
            print('{} vs {}'.format(input_string, truth_sentence))
            assert(False)

        return s_id, input_string, truth_output
    