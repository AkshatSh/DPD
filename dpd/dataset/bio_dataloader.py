from typing import (
    List,
    Tuple,
    Dict,
    Callable,
    Iterator,
)

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token

from .bio_dataset import BIODataset
from .fields import IntField, FloatField

class BIODatasetReader(DatasetReader):
    """
    Dataset Reader for BIO tags
    """
    def __init__(
        self,
        bio_dataset: BIODataset,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.bio_dataset = bio_dataset

    def text_to_instance(self, dataset_id: int, tokens: List[Token], tags: List[str] = None, entry_id: int = None, entry_weight: float = 1.0) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {
            'sentence': sentence_field,
            'dataset_id': IntField(dataset_id),
            'weight': FloatField(entry_weight),
        }

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
        
        if entry_id is not None:
            id_field = IntField(entry_id)
            fields['entry_id'] = id_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        for instance in self.bio_dataset.data:
            sentence = instance['input']
            tags = instance['output'] if 'output' in instance else None
            entry_id = instance['id']
            entry_weight = instance['weight']
            yield self.text_to_instance(self.bio_dataset.dataset_id, [Token(word) for word in sentence], tags, entry_id, entry_weight)