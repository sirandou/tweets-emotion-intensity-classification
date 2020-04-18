from transformers import BertModel
from torch import nn

class BertForUncasedClassification(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertForUncasedClassification, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 4)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, tokens, extra_segments, masks=None):
    _, pooled_output = self.bert(tokens, attention_mask=masks)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    proba = self.softmax(linear_output)
    return proba