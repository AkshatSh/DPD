from typing import (
    List,
    Tuple,
    Dict,
    Callable,
)

# PyTorch imports
import torch
from torch import nn

# AllenNLP imports
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.data.token_indexers import PretrainedBertIndexer


# local imports
from dpd.constants import CADEC_NER_ELMo, CADEC_BERT
from dpd.utils import H5SaveFile
from .crf_tagger import CrfTagger
from .embedder import NERElmoTokenEmbedder, CachedTextFieldEmbedder
from .linear_tagger import LinearTagger


'''
This is a wrapper for the CRF models to be used in the AllenNLP pipeline
'''
class ELMoCrfTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
        cached: bool,
    ) -> None:
        super().__init__(vocab)
        elmo_embedder = NERElmoTokenEmbedder()
        self.vocab = vocab
        self.word_embeddings = BasicTextFieldEmbedder(
            {"tokens": elmo_embedder},
        )

        if cached:
            self.word_embeddings = CachedTextFieldEmbedder(
                text_field_embedder=self.word_embeddings,
            )

            self.word_embeddings.setup_cache(dataset_id=0)
            self.word_embeddings.setup_cache(dataset_id=1)

            self.word_embeddings.load(save_file=H5SaveFile(CADEC_NER_ELMo))

        self.seq2seq_model = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                elmo_embedder.get_output_dim(),
                hidden_dim,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.model = CrfTagger(
            vocab,
            self.word_embeddings,
            self.seq2seq_model,
            label_encoding='BIO',
            calculate_span_f1=True,
            constrain_crf_decoding=True,
            verbose_metrics=False,
            class_labels=class_labels,
            cached_embeddings=cached,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        dataset_id: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor = None,
        entry_id: torch.Tensor = None,
        prob_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        model_out = self.model(
            tokens=sentence,
            tags=labels,
            weight=weight,
            dataset_id=dataset_id,
            entry_id=entry_id,
        )

        return model_out
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.model.get_metrics(reset)
        return metrics

class BERTCrfTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
        cached: bool,
    ) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        bert_embedder = PretrainedBertEmbedder(
                pretrained_model="bert-base-uncased",
                top_layer_only=True, # conserve memory
        )

        self.word_embeddings = BasicTextFieldEmbedder(
            {"tokens": bert_embedder},
            allow_unmatched_tokens=True,
        )

        if cached:
            self.word_embeddings = CachedTextFieldEmbedder(
                text_field_embedder=self.word_embeddings,
            )
            self.word_embeddings.load(save_file=H5SaveFile(CADEC_BERT))

        self.seq2seq_model = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                bert_embedder.get_output_dim(),
                hidden_dim,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.model = CrfTagger(
            vocab,
            self.word_embeddings,
            self.seq2seq_model,
            label_encoding='BIO',
            calculate_span_f1=True,
            constrain_crf_decoding=True,
            verbose_metrics=False,
            class_labels=class_labels,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        dataset_id: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor = None,
        entry_id: torch.Tensor = None,
        prob_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        model_out = self.model(
            tokens=sentence,
            tags=labels,
            weight=weight,
        )

        return model_out
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.model.get_metrics(reset)
        return metrics

class ELMoLinearTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
        cached: bool,
    ) -> None:
        super().__init__(vocab)
        elmo_embedder = NERElmoTokenEmbedder()
        self.vocab = vocab
        self.word_embeddings = BasicTextFieldEmbedder(
            {"tokens": elmo_embedder},
        )

        if cached:
            self.word_embeddings = CachedTextFieldEmbedder(
                text_field_embedder=self.word_embeddings,
            )

            self.word_embeddings.setup_cache(dataset_id=0)
            self.word_embeddings.setup_cache(dataset_id=1)

            self.word_embeddings.load(save_file=H5SaveFile(CADEC_NER_ELMo))

        self.seq2seq_model = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                elmo_embedder.get_output_dim(),
                hidden_dim,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.model = LinearTagger(
            vocab,
            self.word_embeddings,
            self.seq2seq_model,
            label_encoding='BIO',
            calculate_span_f1=True,
            verbose_metrics=False,
            class_labels=class_labels,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        dataset_id: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor = None,
        entry_id: torch.Tensor = None,
        prob_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        model_out = self.model(
            sentence=sentence,
            tags=labels,
            weight=weight,
            prob_labels=prob_labels,
        )

        return model_out
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.model.get_metrics(reset)
        return metrics

class BERTLinearTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
        cached: bool,
    ) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        bert_embedder = PretrainedBertEmbedder(
                pretrained_model="bert-base-uncased",
                top_layer_only=True, # conserve memory
        )

        self.word_embeddings = BasicTextFieldEmbedder(
            {"tokens": bert_embedder},
            allow_unmatched_tokens=True,
        )

        if cached:
            self.word_embeddings = CachedTextFieldEmbedder(
                text_field_embedder=self.word_embeddings,
            )

            self.word_embeddings.setup_cache(dataset_id=0)
            self.word_embeddings.setup_cache(dataset_id=1)

            self.word_embeddings.load(save_file=H5SaveFile(CADEC_NER_ELMo))

        self.seq2seq_model = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(
                elmo_embedder.get_output_dim(),
                hidden_dim,
                bidirectional=True,
                batch_first=True,
            ),
        )

        self.model = LinearTagger(
            vocab,
            self.word_embeddings,
            self.seq2seq_model,
            label_encoding='BIO',
            calculate_span_f1=True,
            verbose_metrics=False,
            class_labels=class_labels,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        dataset_id: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor = None,
        entry_id: torch.Tensor = None,
        prob_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        model_out = self.model(
            sentence=sentence,
            tags=labels,
            weight=weight,
            prob_labels=prob_labels,
        )

        return model_out
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.model.get_metrics(reset)
        return metrics


class ELMoLinearTransformer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
        cached: bool,
    ) -> None:
        super().__init__(vocab)
        elmo_embedder = NERElmoTokenEmbedder()
        self.vocab = vocab
        self.word_embeddings = BasicTextFieldEmbedder(
            {"tokens": elmo_embedder},
        )

        if cached:
            self.word_embeddings = CachedTextFieldEmbedder(
                text_field_embedder=self.word_embeddings,
            )

            self.word_embeddings.setup_cache(dataset_id=0)
            self.word_embeddings.setup_cache(dataset_id=1)

            self.word_embeddings.load(save_file=H5SaveFile(CADEC_NER_ELMo))

        self.seq2seq_model = StackedSelfAttentionEncoder(
            input_dim=self.word_embeddings.get_output_dim(),
            hidden_dim=hidden_dim,
            projection_dim=hidden_dim,
            feedforward_hidden_dim=hidden_dim,
            num_layers=4,
            num_attention_heads=4,
            use_positional_encoding=True,
        )

        self.model = LinearTagger(
            vocab,
            self.word_embeddings,
            self.seq2seq_model,
            label_encoding='BIO',
            calculate_span_f1=True,
            verbose_metrics=False,
            class_labels=class_labels,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        dataset_id: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor = None,
        entry_id: torch.Tensor = None,
        prob_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        model_out = self.model(
            sentence=sentence,
            tags=labels,
            weight=weight,
            prob_labels=prob_labels,
        )

        return model_out
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.model.get_metrics(reset)
        return metrics

class ELMoCRFTransformer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        hidden_dim: int,
        class_labels: List[str],
        cached: bool,
    ) -> None:
        super().__init__(vocab)
        elmo_embedder = NERElmoTokenEmbedder()
        self.vocab = vocab
        self.word_embeddings = BasicTextFieldEmbedder(
            {"tokens": elmo_embedder},
        )

        if cached:
            self.word_embeddings = CachedTextFieldEmbedder(
                text_field_embedder=self.word_embeddings,
            )

            self.word_embeddings.setup_cache(dataset_id=0)
            self.word_embeddings.setup_cache(dataset_id=1)

            self.word_embeddings.load(save_file=H5SaveFile(CADEC_NER_ELMo))

        self.seq2seq_model = StackedSelfAttentionEncoder(
            input_dim=self.word_embeddings.get_output_dim(),
            hidden_dim=hidden_dim,
            projection_dim=hidden_dim,
            feedforward_hidden_dim=hidden_dim,
            num_layers=4,
            num_attention_heads=4,
            use_positional_encoding=True,
        )

        self.model = CrfTagger(
            vocab,
            self.word_embeddings,
            self.seq2seq_model,
            label_encoding='BIO',
            calculate_span_f1=True,
            constrain_crf_decoding=True,
            verbose_metrics=False,
            class_labels=class_labels,
        )
    
    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        dataset_id: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor = None,
        entry_id: torch.Tensor = None,
        prob_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        model_out = self.model(
            sentence=sentence,
            tags=labels,
            weight=weight,
            prob_labels=prob_labels,
        )

        return model_out
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.model.get_metrics(reset)
        return metrics