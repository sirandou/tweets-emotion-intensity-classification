from transformers import BertModel
from torch import nn
import torch

class BertHybridClassifier(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertHybridClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(772, 4)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, tokens, extra_segments, masks=None):
    _, pooled_output = self.bert(tokens, attention_mask=masks)
    third_tensor = torch.cat((pooled_output, extra_segments), 1)
    dropout_output = self.dropout(third_tensor)
    linear_output = self.linear(dropout_output)
    proba = self.softmax(linear_output)
    return proba