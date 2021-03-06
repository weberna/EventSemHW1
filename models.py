import torch 
import torch.nn as nn
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RnnEncoder(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int) -> None:
        super(RnnEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self.output_dim = self.hidden_dim = output_dim
        self.rnn = nn.GRU(self._embedding_dim, self.hidden_dim, 1, bidirectional=False)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor): 
        """
        Params:
            Tokens (Tensor[batch, maxlength, dim]) : the input embeddings
            lengths (Tensor[batch])
        Outputs:
            encoded states (Tensor[batch, output_size])
        """

        packed_input = pack_padded_sequence(tokens, lengths.cpu().numpy())
        packed_input=tokens
        self.rnn.flatten_parameters()
        _, last_state = self.rnn(packed_input) #[1, batch, hiddensize]
        last_state=last_state.squeeze(dim=0)
        return last_state

class PredictionModel(nn.Module):
    def __init__(self, inputs_dim, sentence_vocab_size, predarg_vocab_size):
        super(PredictionModel, self).__init__()
        self.text_embeddings = nn.Embedding(sentence_vocab_size, inputs_dim)
        self.text_encoder = RnnEncoder(inputs_dim, inputs_dim)
        self.pred_arg_embeddings = nn.Embedding(predarg_vocab_size, inputs_dim)
        self.outputlayer = nn.Linear(3*inputs_dim, 1)

    def forward(self, input):
        encoded_text = self.text_encoder(self.text_embeddings(input.sentence[0]), input.sentence[1])
        pred_head = self.pred_arg_embeddings(input.pred_head)
        arg_head = self.pred_arg_embeddings(input.arg_head)
        concat = torch.cat([encoded_text, pred_head, arg_head], dim=1)
        return self.outputlayer(concat).squeeze()


