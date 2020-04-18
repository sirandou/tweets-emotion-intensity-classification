import os
import pandas as pd
from nltk.tokenize import TweetTokenizer
from torchtext import data, vocab

from src.data.create_dataloader import create_dataloader

"""
Code adapted from https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
"""
def load_uncased_data(emotion):
  tweet_tokenizer = TweetTokenizer()
  
  def tokenize_tweets(tweet):
    return tweet_tokenizer.tokenize(tweet)
  
  txt_field = data.Field(sequential=True, tokenize=tokenize_tweets, include_lengths=True, use_vocab=True)
  label_field = data.LabelField(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

  train_val_fields = [
    ('ID', None),
    ('Tweet', None),
    ('Affect Dimension', None),
    ('Intensity Class', None),
    ('Intensity', label_field),
    ('Clean_Tweet_Cased', None),
    ('Clean_Tweet', txt_field)
  ]

  trainds, valds, testds = data.TabularDataset.splits(
    path='./data/cleaned/' + emotion, 
    format='CSV', 
    train='train.csv', 
    validation='dev.csv',
    test='test.csv',
    fields=train_val_fields, 
    skip_header=True
  )

  traindl, valdl, testdl = data.BucketIterator.splits(
    datasets=(trainds, valds, testds),
    batch_sizes=(32,32,32),
    sort_key=lambda x: len(x.Clean_Tweet),
    device=None,
    sort_within_batch=True, 
    repeat=False
  )

  vec = vocab.Vectors('./data/models/glove/glove.twitter.27B.200d.txt')
  txt_field.build_vocab(trainds, valds, max_size=100000, vectors=vec)
  label_field.build_vocab(trainds)

  vocab_size = len(txt_field.vocab)
  
  return vocab_size, trainds, valds, testds, traindl, valdl, testdl

def get_bert_data_loader(emotion, uncased=True):
  train_path = os.path.join('data', 'cleaned', emotion, 'train.csv')
  dev_path = os.path.join('data', 'cleaned', emotion, 'dev.csv')
  test_path = os.path.join('data', 'cleaned', emotion, 'test.csv')

  train_df = pd.read_csv(train_path)
  dev_df = pd.read_csv(dev_path)
  test_df = pd.read_csv(test_path)

  train_dataloader = create_dataloader(train_df, uncased=uncased)
  dev_dataloader = create_dataloader(dev_df, uncased=uncased, batch=len(dev_df))
  test_dataloader = create_dataloader(test_df, uncased=uncased, batch=len(test_df))

  return (train_dataloader, dev_dataloader, test_dataloader)

def get_all_bert_data_loader(uncased=True):

  train_df = []
  dev_df = []
  test_df = []

  for emotion in ['anger', 'fear', 'sadness', 'joy']:
    train_path = os.path.join('data', 'cleaned', emotion, 'train.csv')
    dev_path = os.path.join('data', 'cleaned', emotion, 'dev.csv')
    test_path = os.path.join('data', 'cleaned', emotion, 'test.csv')

    train_df.append(pd.read_csv(train_path))
    dev_df.append(pd.read_csv(dev_path))
    test_df.append(pd.read_csv(test_path))
  
  train = pd.concat(train_df)
  dev = pd.concat(dev_df)
  test = pd.concat(test_df)

  train_dataloader = create_dataloader(train, uncased=uncased)
  dev_dataloader = create_dataloader(dev, uncased=uncased, batch=len(dev))
  test_dataloader = create_dataloader(test, uncased=uncased, batch=len(test))

  return (train_dataloader, dev_dataloader, test_dataloader)