from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.tensorboard_writer import TensorboardWriter

from dpd.dataset.bio_dataset import BIODataset
from dpd.dataset.bio_dataloader import BIODatasetReader
from dpd.constants import (
    CONLL2003_TRAIN,
    CONLL2003_VALID,
    CADEC_TRAIN,
    CADEC_VALID,
    CADEC_TEST,
)

from dpd.models.embedder.ner_elmo import NERElmoTokenEmbedder
from dpd.models import build_model


def setup_reader(d_id: int, file_name: str, binary_class: str) -> DatasetReader:
    bio_dataset = BIODataset(
        dataset_id=d_id,
        file_name=file_name,
        binary_class=binary_class,
    )
    bio_dataset.parse_file()
    return BIODatasetReader(
        bio_dataset=bio_dataset,
        token_indexers={
            'tokens': ELMoTokenCharactersIndexer(),
        }
    )

train_reader = setup_reader(0, CADEC_TRAIN, 'ADR')
valid_reader = setup_reader(1, CADEC_VALID, 'ADR')

train_dataset = train_reader.read(cached_path(CONLL2003_TRAIN))
validation_dataset = valid_reader.read(cached_path(CONLL2003_VALID))

EMBEDDING_DIM = 1024
HIDDEN_DIM = 1024

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

model = build_model(
    model_type='ELMo_bilstm_crf',
    vocab=vocab,
    hidden_dim=HIDDEN_DIM,
    class_labels=['B-ADR', 'I-ADR'],
    cached=True,
    dataset_name='cadec',
)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
iterator = BucketIterator(batch_size=1, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    patience=10,
    num_epochs=25,
    cuda_device=cuda_device,
    serialization_dir='/tmp/akshats/dpd/models',
)
trainer._tensorboard = TensorboardWriter(
    get_batch_num_total=lambda: trainer._batch_num_total,
    serialization_dir=None,
    summary_interval=100,
    histogram_interval=None,
    should_log_parameter_statistics=False,
    should_log_learning_rate=False
)
metrics = trainer.train()
print(metrics)
iterator = BucketIterator(
    batch_size=4,
    sorting_keys=[("sentence", "num_tokens")],
)
test_reader = setup_reader(2, CADEC_TEST, 'ADR')
iterator.index_with(vocab)
instances = test_reader.read('temp.txt')
metrics = evaluate(model, instances, iterator, cuda_device, "")
print(metrics)
