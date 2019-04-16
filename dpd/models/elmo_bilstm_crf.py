import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.elmo import Elmo, batch_to_ids

from .crf import CRF
from embedder.elmo import FrozenELMo, ELMo

from ner import constants

class ELMo_BiLSTM_CRF(CRF):
    '''
    This model is a BiLSTM CRF for Named Entity Recognition, this involes a Bidirectional 
    LSTM to compute the features for an input sentence and then convert the computed features 
    to a tag sequence through a tag decoding CRF (Conditional Random Field)
    '''
    def __init__(self, vocab, tag_set, hidden_dim, batch_size, freeze_elmo: bool=True):
        super(ELMo_BiLSTM_CRF, self).__init__(vocab, tag_set, 1024, hidden_dim, batch_size)
        self.embedding_dim = 1024 # elmo embedding size
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.tag_set = tag_set
        self.tag_set_size = len(tag_set)
        self.batch_size = batch_size
        self.elmo = FrozenELMo.instance() if freeze_elmo else ELMo()

        # Bidirectional LSTM for computing the CRF features 
        # Note: there is an output vector for each direction of the LSTM
        #   hence the hidden dimension is // 2, so that the output of the LSTM 
        #   is size hidden dimension due to the concatenation of the forward and backward
        #   LSTM pass
        self.lstm = nn.LSTM(
            self.embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Project LSTM outputs to the taget set space
        self.tag_projection = nn.Linear(hidden_dim, len(self.tag_set))

        self.hidden = self.init_hidden(batch_size, 'cpu')
    
    def init_hidden(self, batch_size, device):
        '''
        Initialize the hidden dimensions of the LSTM outputs
        '''
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
            torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
        )
    
    def compute_lstm_features(self, sentence, sentence_chars, mask):
        '''
        Given an input encoded sentence, compute the LSTM features
        for the CRF 

        Essentially run the Bidirectional LSTM and embedder
        '''
        device = sentence.device
        self.hidden = self.init_hidden(sentence.shape[0], device)
        long_sentence = sentence.long()

        embeddings = self.elmo(sentence_chars)
        embeded_sentence = embeddings['elmo_representations'][0]

        # embeded_sentence is now (batch_size x max_length x embedding dim)

        lstm_output, self.hidden = self.lstm(embeded_sentence, self.hidden)

        # lstm output is now (batch_size x max_length x hidden_dim)
        features = self.tag_projection(lstm_output)

        # features is now (batch_size x max_length x tag_set size)
        return features