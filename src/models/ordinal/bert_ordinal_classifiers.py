from transformers import BertModel
from torch import nn


############ 1
class BertForUncasedClassification_1(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertForUncasedClassification_1, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 2)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, tokens, extra_segments, masks=None):
    _, pooled_output = self.bert(tokens, attention_mask=masks)
    # third_tensor = torch.cat((pooled_output, extra_segments), 1)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    proba = self.softmax(linear_output)
    return proba

class BertForCasedClassification_1(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertForCasedClassification_1, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 2)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, tokens, extra_segments, masks=None):
    _, pooled_output = self.bert(tokens, attention_mask=masks)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    proba = self.softmax(linear_output)
    return proba

################2
class BertForUncasedClassification_2(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertForUncasedClassification_2, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 2)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, tokens, extra_segments, masks=None):
    _, pooled_output = self.bert(tokens, attention_mask=masks)
    # third_tensor = torch.cat((pooled_output, extra_segments), 1)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    proba = self.softmax(linear_output)
    return proba

class BertForCasedClassification_2(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertForCasedClassification_2, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 2)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, tokens, extra_segments, masks=None):
    _, pooled_output = self.bert(tokens, attention_mask=masks)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    proba = self.softmax(linear_output)
    return proba

##############3
class BertForUncasedClassification_3(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertForUncasedClassification_3, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 2)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, tokens, extra_segments, masks=None):
    _, pooled_output = self.bert(tokens, attention_mask=masks)
    # third_tensor = torch.cat((pooled_output, extra_segments), 1)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    proba = self.softmax(linear_output)
    return proba

class BertForCasedClassification_3(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertForCasedClassification_3, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 2)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, tokens, extra_segments, masks=None):
    _, pooled_output = self.bert(tokens, attention_mask=masks)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    proba = self.softmax(linear_output)
    return proba