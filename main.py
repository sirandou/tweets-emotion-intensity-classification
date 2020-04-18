import os
import optparse
from torchtext import vocab

from src.data.save_data import save_cleaned_data
from src.models.lstm.train.lstm_train import train_lstm
from src.models.lstm.train.bilstm_train import train_bilstm
from src.models.bert.train.train_hybrid import train_hybrid
from src.models.bert.train.train_individual import train_cased, train_uncased

from src.models.ordinal.train.train_ordinal_anger import train_ordinal_anger
from src.models.ordinal.train.train_ordinal_fear import train_ordinal_fear
from src.models.ordinal.train.train_ordinal_joy import train_ordinal_joy
from src.models.ordinal.train.train_ordinal_sadness import train_ordinal_sadness

from src.models.bert.evaluate.evaluation import evaluate_uncased

from src.models.bert.predict.predict_uncased import predict_uncased

from src.models.bert.bert_hybrid import BertHybridClassifier

"""
Generate the cleaned data from the data available
"""
save_cleaned_data()

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("--train", dest='is_train', default=False, help="Training")
    optparser.add_option("--train-model", dest='train_model', default='bert_uncased', help="lstm/bilstm/bert_uncased/bert_cased/bert_hybrid/bert_ordinal")
    (opts, _) = optparser.parse_args()

    if opts.is_train:
      if opts.train_model == 'lstm':
        train_lstm()
      elif opts.train_model == 'bilstm':
        train_bilstm()
      elif opts.train_model == 'bert_uncased':
        train_uncased()
      elif opts.train_model == 'bert_cased':
        train_cased()
      elif opts.train_model == 'bert_hybrid':
        train_hybrid()
      elif opts.train_model == 'bert_ordinal':
        train_ordinal_anger()
        train_ordinal_fear()
        train_ordinal_joy()
        train_ordinal_sadness()
    else:
      predict_uncased()
      evaluate_uncased()



