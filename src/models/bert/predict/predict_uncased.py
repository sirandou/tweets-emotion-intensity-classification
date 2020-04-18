from src.data.load_cleaned_data import get_bert_data_loader
from src.models.bert.bert_uncased_classifier import BertForUncasedClassification
import pandas as pd
import os
import torch
from transformers import BertTokenizer
from src.evaluate.evaluation import evaluate_PerEmotion

def predict(df, model, tokenizer, column):
  model.eval()
  final_data = []
  for row in df.iterrows():
    final_row = row[1]
    encoded_dict = tokenizer.encode_plus(
      row[1][column],
      add_special_tokens = True,
      max_length = 128,
      pad_to_max_length = True,
      return_attention_mask = True,
      return_tensors = 'pt',
    )
    with torch.no_grad():
      probas  = model(encoded_dict['input_ids'], encoded_dict['attention_mask'])
      output = torch.max(probas, 1)[1]
      final_row['Predicted Intensity'] = int(output)

    final_data.append(final_row)
  return pd.DataFrame(final_data)

def get_data(emotion):
  train_path = os.path.join('data', 'cleaned', emotion, 'train.csv')
  dev_path = os.path.join('data', 'cleaned', emotion, 'dev.csv')
  test_path = os.path.join('data', 'cleaned', emotion, 'test.csv')

  train_df = pd.read_csv(train_path)
  dev_df = pd.read_csv(dev_path)
  test_df = pd.read_csv(test_path)

  return train_df, dev_df, test_df

def predict_uncased_anger(dropout=0.3):
  train_df, dev_df, test_df = get_data('anger')
  bert_tokenizer_uncased = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  model = BertForUncasedClassification(dropout)
  model.load_state_dict(torch.load('./data/models/uncased-bert/models/anger', map_location='cpu'))
  predicted_train_df = predict(train_df, model, bert_tokenizer_uncased, 'Clean_Tweet')
  predicted_dev_df = predict(dev_df, model, bert_tokenizer_uncased, 'Clean_Tweet')
  predicted_test_df = predict(test_df, model, bert_tokenizer_uncased, 'Clean_Tweet')

  predicted_train_df.to_csv('./data/models/uncased-bert/output/anger_train_out.csv')
  predicted_dev_df.to_csv('./data/models/uncased-bert/output/anger_dev_out.csv')
  predicted_test_df.to_csv('./data/models/uncased-bert/output/anger_test_out.csv')

def predict_uncased_fear(dropout=0.1):
  train_df, dev_df, test_df = get_data('fear')
  bert_tokenizer_uncased = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  model = BertForUncasedClassification(dropout)
  model.load_state_dict(torch.load('./data/models/uncased-bert/models/fear', map_location='cpu'))
  predicted_train_df = predict(train_df, model, bert_tokenizer_uncased, 'Clean_Tweet')
  predicted_dev_df = predict(dev_df, model, bert_tokenizer_uncased, 'Clean_Tweet')
  predicted_test_df = predict(test_df, model, bert_tokenizer_uncased, 'Clean_Tweet')

  predicted_train_df.to_csv('./data/models/uncased-bert/output/fear_train_out.csv')
  predicted_dev_df.to_csv('./data/models/uncased-bert/output/fear_dev_out.csv')
  predicted_test_df.to_csv('./data/models/uncased-bert/output/fear_test_out.csv')

def predict_uncased_joy(dropout=0.1):
  train_df, dev_df, test_df = get_data('joy')
  bert_tokenizer_uncased = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  model = BertForUncasedClassification(dropout)
  model.load_state_dict(torch.load('./data/models/uncased-bert/models/joy', map_location='cpu'))
  predicted_train_df = predict(train_df, model, bert_tokenizer_uncased, 'Clean_Tweet')
  predicted_dev_df = predict(dev_df, model, bert_tokenizer_uncased, 'Clean_Tweet')
  predicted_test_df = predict(test_df, model, bert_tokenizer_uncased, 'Clean_Tweet')

  predicted_train_df.to_csv('./data/models/uncased-bert/output/joy_train_out.csv')
  predicted_dev_df.to_csv('./data/models/uncased-bert/output/joy_dev_out.csv')
  predicted_test_df.to_csv('./data/models/uncased-bert/output/joy_test_out.csv')

def predict_uncased_sadness(dropout=0.2):
  train_df, dev_df, test_df = get_data('sadness')
  bert_tokenizer_uncased = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  model = BertForUncasedClassification(dropout)
  model.load_state_dict(torch.load('./data/models/uncased-bert/models/sadness', map_location='cpu'))
  predicted_train_df = predict(train_df, model, bert_tokenizer_uncased, 'Clean_Tweet')
  predicted_dev_df = predict(dev_df, model, bert_tokenizer_uncased, 'Clean_Tweet')
  predicted_test_df = predict(test_df, model, bert_tokenizer_uncased, 'Clean_Tweet')

  predicted_train_df.to_csv('./data/models/uncased-bert/output/sadness_train_out.csv')
  predicted_dev_df.to_csv('./data/models/uncased-bert/output/sadness_dev_out.csv')
  predicted_test_df.to_csv('./data/models/uncased-bert/output/sadness_test_out.csv')



def predict_uncased():
  predict_uncased_anger()
  predict_uncased_fear()
  predict_uncased_sadness()
  predict_uncased_joy()