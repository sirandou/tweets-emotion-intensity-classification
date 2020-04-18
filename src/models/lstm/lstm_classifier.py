import torch
from torch.nn.utils.rnn import pack_padded_sequence

from src.constants.constant import DEVICE

"""
Code adapted from https://github.com/claravania/lstm-pytorch/blob/master/model.py
"""
class LSTMClassifier(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim, n_hidden, n_out, pretrained_vec, dropout=0.2, bidirectional=True):
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.bidirectional = bidirectional
    
    self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
    self.embedding.weight.data.copy_(pretrained_vec)
    self.embedding.weight.requires_grad = False
    
    self.lstm = torch.nn.LSTM(self.embedding_dim, self.n_hidden, bidirectional=bidirectional)
    self.dropout_layer = torch.nn.Dropout(p=dropout)
    self.linear = torch.nn.Linear(self.n_hidden, self.n_out)
    self.softmax = torch.nn.LogSoftmax(dim=1)
        
  def forward(self, batch, lengths):
    self.hidden = self.init_hidden(batch.size(-1))
    embeds = self.embedding(batch)
    packed_input = pack_padded_sequence(embeds, lengths)
    _, (ht, _) = self.lstm(packed_input, self.hidden)
    output = self.dropout_layer(ht[-1])
    output = self.linear(output)
    output = self.softmax(output)
    return output
    
  def init_hidden(self, batch_size): 
    if self.bidirectional:
      return (
          torch.autograd.Variable(torch.randn(2, batch_size, self.n_hidden)).to(DEVICE), 
          torch.autograd.Variable(torch.randn(2, batch_size, self.n_hidden)).to(DEVICE)
      )
    return (
      torch.autograd.Variable(torch.randn(1, batch_size, self.n_hidden)).to(DEVICE), 
      torch.autograd.Variable(torch.randn(1, batch_size, self.n_hidden)).to(DEVICE)
    )