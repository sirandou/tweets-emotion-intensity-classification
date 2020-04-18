import pandas as pd
import re
import emoji
from sklearn.model_selection import train_test_split
import os

from src.constants.constant import SEED_VAL

def clean_tweets(in_text):
  # regex from https://regexr.com/36fcc
  url_re = re.compile(r'(http|ftp|https)://([\w+?\.\w+])+([a-zA-Z0-9\~\!\@\#\$\%\^\&\*\(\)_\-\=\+\\\/\?\.\:\;\'\,]*)?')
  in_text = url_re.sub("",in_text)

  # remove screen name references
  screen_name_re = re.compile(r'(@\w+)')
  in_text = screen_name_re.sub("", in_text)

  # hashtags
  in_text = in_text.replace('#', '')

  # remove RTs
  in_text = in_text.replace("RT","")

  # strip text of extra spaces , and keep one space between each word
  in_text = " ".join(in_text.split())

  return in_text

def prepare_df(df):
  df['Intensity'] = df['Intensity Class'].str[0].apply(int)
  df['Clean_Tweet_Cased'] = df['Tweet'].apply(clean_tweets).apply(emoji.demojize).str.replace(':', '')
  df['Clean_Tweet'] = df['Clean_Tweet_Cased'].str.lower()
  return df

def load_raw_data(train_path, dev_path):
  df = pd.read_csv(train_path, sep='\t')
  cleaned_df = prepare_df(df)

  train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=SEED_VAL)

  dev_df = pd.read_csv(dev_path, sep='\t')
  dev_df = prepare_df(dev_df)

  return train_df, dev_df, test_df

def save_anger_data():
  train_path = os.path.join('data', 'raw', 'train', 'anger.txt')
  dev_path = os.path.join('data', 'raw', 'dev', 'anger.txt')
  train_df, dev_df, test_df = load_raw_data(train_path, dev_path)

  dir = os.path.join('data', 'cleaned', 'anger')
  os.mkdir(dir)
  
  train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
  dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
  test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)

def save_fear_data():
  train_path = os.path.join('data', 'raw', 'train', 'fear.txt')
  dev_path = os.path.join('data', 'raw', 'dev', 'fear.txt')
  train_df, dev_df, test_df = load_raw_data(train_path, dev_path)

  dir = os.path.join('data', 'cleaned', 'fear')
  os.mkdir(dir)
  
  train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
  dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
  test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)

def save_sadness_data():
  train_path = os.path.join('data', 'raw', 'train', 'sadness.txt')
  dev_path = os.path.join('data', 'raw', 'dev', 'sadness.txt')
  train_df, dev_df, test_df = load_raw_data(train_path, dev_path)

  dir = os.path.join('data', 'cleaned', 'sadness')
  os.mkdir(dir)
  
  train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
  dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
  test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)

def save_joy_data():
  train_path = os.path.join('data', 'raw', 'train', 'joy.txt')
  dev_path = os.path.join('data', 'raw', 'dev', 'joy.txt')
  train_df, dev_df, test_df = load_raw_data(train_path, dev_path)

  dir = os.path.join('data', 'cleaned', 'joy')
  os.mkdir(dir)
  
  train_df.to_csv(os.path.join(dir, 'train.csv'), index=False)
  dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False)
  test_df.to_csv(os.path.join(dir, 'test.csv'), index=False)

def save_cleaned_data():
  if not os.path.exists('./data/cleaned/anger'):
    save_anger_data()

  if not os.path.exists('./data/cleaned/fear'):
    save_fear_data()

  if not os.path.exists('./data/cleaned/sadness'):
    save_sadness_data()

  if not os.path.exists('./data/cleaned/joy'):
    save_joy_data()


  
