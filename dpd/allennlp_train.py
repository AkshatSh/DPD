from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, CrfTagger
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from dpd.dataset.bio_dataset import BIODataset
from dpd.dataset.bio_dataloader import BIODatasetReader
from dpd.constants import (
    CONLL2003_TRAIN,
    CONLL2003_VALID,
)

class LstmTagger(Model):
    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        vocab: Vocabulary,
    ) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self._f1_metric = SpanBasedF1Measure(
            vocab,
            tag_namespace="labels",
            label_encoding="BIO",
        )

        self._verbose_metrics = False

    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            self._f1_metric(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            "accuracy": self.accuracy.get_metric(reset),
        }

        f1_dict = self._f1_metric.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update({
                x: y for x, y in f1_dict.items() if
                "overall" in x
            })
        
        return metrics_to_return

class CrfLstmTagger(Model):
    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        vocab: Vocabulary,
    ) -> None:
        super().__init__(vocab)
        self.model = CrfTagger(
            vocab,
            word_embeddings,
            lstm,
            label_encoding='BIO',
            calculate_span_f1=True,
            # constrain_crf_decoding=True,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model(tokens=sentence, tags=labels)
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.model.get_metrics()


def setup_reader(file_name: str, binary_class: str) -> DatasetReader:
    bio_dataset = BIODataset(
        file_name=file_name,
        binary_class=binary_class,
    )

    bio_dataset.parse_file()

    return BIODatasetReader(
        bio_dataset=bio_dataset,
    )

train_reader = setup_reader(CONLL2003_TRAIN, 'PER')
valid_reader = setup_reader(CONLL2003_VALID, 'PER')

train_dataset = train_reader.read(cached_path(CONLL2003_TRAIN))
validation_dataset = valid_reader.read(cached_path(CONLL2003_VALID))

EMBEDDING_DIM = 300
HIDDEN_DIM = 512

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, bidirectional=True, batch_first=True))

# model = LstmTagger(word_embeddings, lstm, vocab)

model = CrfLstmTagger(
    word_embeddings, lstm, vocab
)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)


for i in range(10):
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        patience=10,
        num_epochs=1,
        cuda_device=cuda_device,
    )
    metrics = trainer.train()
    # print(metrics)

predictor = SentenceTaggerPredictor(model, dataset_reader=reader)


