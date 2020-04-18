import pandas as pd
from src.data.load_cleaned_data import load_uncased_data
from src.models.lstm.train.train import train
from src.models.lstm.lstm_classifier import LSTMClassifier

"""
Train the LSTM Models
"""
def train_bilstm_anger():
  vocab_size, trainds, valds, testds, traindl, valdl, testdl = load_uncased_data('anger')

  final_stats = []
  for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for hidden_size in [32, 64, 96, 128]:
      for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        model = LSTMClassifier(
          vocab_size, 
          200, 
          hidden_size, 
          4, 
          trainds.fields['Clean_Tweet'].vocab.vectors,
          dropout=dropout,
          bidirectional=True
        )
        training_stats = train(model, traindl, valdl, lr=lr, hidden_size=hidden_size, dropout=dropout)
        final_stats.extend(training_stats)
  pd.DataFrame(final_stats).to_csv('./data/models/bilstm/results/anger.csv')


def train_bilstm_fear():
  vocab_size, trainds, valds, testds, traindl, valdl, testdl = load_uncased_data('fear')

  final_stats_fear = []
  for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for hidden_size in [32, 64, 96, 128]:
      for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        model = LSTMClassifier(
          vocab_size, 
          200, 
          hidden_size, 
          4, 
          trainds.fields['Clean_Tweet'].vocab.vectors,
          dropout=dropout,
          bidirectional=True
        )
        training_stats = train(model, traindl, valdl, lr=lr, hidden_size=hidden_size, dropout=dropout)
        final_stats_fear.extend(training_stats)
  pd.DataFrame(final_stats_fear).to_csv('./data/models/bilstm/results/fear.csv')

def train_bilstm_sadness():
  vocab_size, trainds, valds, testds, traindl, valdl, testdl = load_uncased_data('sadness')
  final_stats_sadness = []
  for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for hidden_size in [32, 64, 96, 128]:
      for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        model = LSTMClassifier(
          vocab_size, 
          200, 
          hidden_size, 
          4, 
          trainds.fields['Clean_Tweet'].vocab.vectors,
          dropout=dropout,
          bidirectional=False
        )
        training_stats = train(model, traindl, valdl, lr=lr, hidden_size=hidden_size, dropout=dropout)
        final_stats_sadness.extend(training_stats)
  pd.DataFrame(final_stats_sadness).to_csv('./data/models/bilstm/results/sadness.csv')

def train_bilstm_joy():
  vocab_size, trainds, valds, testds, traindl, valdl, testdl = load_uncased_data('joy')

  final_stats_joy = []
  for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for hidden_size in [32, 64, 96, 128]:
      for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        model = LSTMClassifier(
          vocab_size, 
          200, 
          hidden_size, 
          4, 
          trainds.fields['Clean_Tweet'].vocab.vectors,
          dropout=dropout,
          bidirectional=False
        )
        training_stats = train(model, traindl, valdl, lr=lr, hidden_size=hidden_size, dropout=dropout)
        final_stats_joy.extend(training_stats)
  pd.DataFrame(final_stats_joy).to_csv('./data/models/bilstm/results/joy.csv')

def train_bilstm():
  train_bilstm_anger()
  train_bilstm_fear()
  train_bilstm_sadness()
  train_bilstm_joy()