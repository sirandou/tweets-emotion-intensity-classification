import torch
import random
import numpy as np
from src.constants.constant import SEED_VAL
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler

random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

bert_tokenizer_uncased = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_tokenizer_cased = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def create_dataloader(df, uncased=False, batch=32):
  input_ids = []
  attention_masks = []
  anger = []
  fear = []
  joy = []
  sadness = []
  intensity = []
  vec = []

  bert_tokenizer = bert_tokenizer_cased
  column = 'Clean_Tweet_Cased'
  if uncased:
    bert_tokenizer = bert_tokenizer_uncased
    column = 'Clean_Tweet'
  for _, row in df.iterrows():
    encoded_dict = bert_tokenizer.encode_plus(
      row[column],
      add_special_tokens = True,
      max_length = 128,
      pad_to_max_length = True,
      return_attention_mask = True,
      return_tensors = 'pt',
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    anger.append(row['Affect Dimension'] == 'anger')
    fear.append(row['Affect Dimension'] == 'fear')
    joy.append(row['Affect Dimension'] == 'joy')
    sadness.append(row['Affect Dimension'] == 'sadness')

    vec.append([row['Affect Dimension'] == 'anger', row['Affect Dimension'] == 'fear', row['Affect Dimension'] == 'sadness', row['Affect Dimension'] == 'joy'])

    intensity.append(row['Intensity'])
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  intensity = torch.tensor(intensity)
  anger = torch.tensor(anger)
  fear = torch.tensor(fear)
  joy = torch.tensor(joy)
  sadness = torch.tensor(sadness)
  vec = torch.tensor(vec)
  vec = vec.type(torch.FloatTensor)

  dataset = TensorDataset(input_ids, attention_masks, anger, fear, joy, sadness, vec, intensity)

  dataloader = DataLoader(
    dataset,
    sampler = RandomSampler(dataset),
    batch_size = batch
  )
  return dataloader