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

from dpd.dataset.bio_dataset import BIODataset
from dpd.dataset.bio_dataloader import BIODatasetReader
from dpd.models.allennlp_crf_tagger import CrfTagger
from dpd.training.metrics import TagF1
from dpd.constants import (
    CONLL2003_TRAIN,
    CONLL2003_VALID,
    CADEC_TRAIN,
    CADEC_VALID,
)

from dpd.models.embedder.ner_elmo import NERElmoTokenEmbedder

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
        # self._f1_metric = TagF1(
        #     vocab=vocab,
        #     class_labels=['B-PER', 'I-PER']
        # )

        self._span_f1 = SpanBasedF1Measure(
            vocab,
            tag_namespace="labels",
            label_encoding="BIO",
        )

        # self._negative_f1_metric = TagF1(
        #     vocab=vocab,
        #     class_labels=['O'],
        # )

        self._verbose_metrics = False

    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        # dataset_id: torch.Tensor,
        labels: torch.Tensor = None,
        # entry_id: torch.Tensor = None,
        # weight: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            accuracy_args = (tag_logits, labels, mask)
            self.accuracy(tag_logits, labels, mask)
            # self._f1_metric(*accuracy_args)
            self._span_f1(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            "accuracy": self.accuracy.get_metric(reset),
        }

        # f1_dict = self._f1_metric.get_metric(reset=reset)
        # metrics_to_return.update(f1_dict)

        span_f1_dict = self._span_f1.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(span_f1_dict)
        else:
            metrics_to_return.update({
                x: y for x, y in span_f1_dict.items() if
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
            verbose_metrics=False,
        )

        # self._tag_f1_metric = TagF1(
        #     vocab=vocab,
        #     class_labels=['B-ADR', 'I-ADR']
        # )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        dataset_id: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor = None,
        entry_id: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        model_out = self.model(
            tokens=sentence,
            tags=labels,
        )

        # if labels is not None:
        #     tag_logits = model_out['logits']
        #     mask = model_out['mask']
        #     self._tag_f1_metric(tag_logits, labels, mask)

        return model_out
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.model.get_metrics(reset)
        # metrics.update(self._tag_f1_metric.get_metric(reset))
        return metrics


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
HIDDEN_DIM = 512


# token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
#                             embedding_dim=EMBEDDING_DIM)
# word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
# vocab = Vocabulary()
elmo_embedder = NERElmoTokenEmbedder()
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

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

optimizer = optim.SGD(model.parameters(), lr=.01, weight_decay=1e-4)
iterator = BucketIterator(batch_size=1, sorting_keys=[("sentence", "num_tokens")])
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