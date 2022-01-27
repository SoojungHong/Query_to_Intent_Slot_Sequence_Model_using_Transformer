#!/usr/bin/env python
# coding: utf-8
# __author__: jorg.frese@here.com


import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class LSTMClassifierMiniBatch(nn.Module):
    def __init__(self, params, **kwargs):
        super(LSTMClassifierMiniBatch, self).__init__()
        self.__dict__.update(params)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.word_embeddings = nn.Embedding.from_pretrained(self.PT_WEIGHTS, freeze=True)
        self.lstm = nn.LSTM(self.EMBEDDING_DIM, self.HIDDEN_DIM, num_layers=self.NLAYERS, bidirectional=True, dropout=self.DROPOUT, batch_first=True)
        self.hidden2label = nn.Linear(self.HIDDEN_DIM, self.LABEL_SIZE)

    def init_hidden(self, nlayers, batch_size):
        return (torch.zeros(2*nlayers, batch_size, self.HIDDEN_DIM),
                torch.zeros(2*nlayers, batch_size, self.HIDDEN_DIM))

    def forward(self, input_batch):
        self.hidden = self.init_hidden(self.NLAYERS, input_batch[0].size(0))
        embeds = self.word_embeddings(input_batch[0])
        x_packed = pack_padded_sequence(embeds, input_batch[2], batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(x_packed, self.hidden)
        y = self.hidden2label(ht[-1])
        return y
    

class LSTMClassifierMiniBatchNoPT(nn.Module):
    def __init__(self, params):
        super(LSTMClassifierMiniBatchNoPT, self).__init__()
        self.__dict__.update(params)
        self.word_embeddings = nn.Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM)
        self.lstm = nn.LSTM(self.EMBEDDING_DIM, self.HIDDEN_DIM, num_layers=self.NLAYERS, bidirectional=True, dropout=self.DROPOUT, batch_first=True)
        self.hidden2label = nn.Linear(self.HIDDEN_DIM, self.LABEL_SIZE)

    def init_hidden(self, nlayers, batch_size):
        return (torch.zeros(2*nlayers, batch_size, self.HIDDEN_DIM),
                torch.zeros(2*nlayers, batch_size, self.HIDDEN_DIM))

    def forward(self, input_batch):
        self.hidden = self.init_hidden(self.NLAYERS, input_batch[0].size(0))
        embeds = self.word_embeddings(input_batch[0])
        x_packed = pack_padded_sequence(embeds, input_batch[2], batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(x_packed, self.hidden)
        y = self.hidden2label(ht[-1])
        return y
