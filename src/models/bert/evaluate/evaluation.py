import pandas as pd
import torch
from src.evaluate.evaluation import evaluate_PerEmotion

def evaluate_anger_uncased():
  print('\n\nAnger')
  train_df = pd.read_csv('./data/models/uncased-bert/output/anger_train_out.csv')
  dev_df = pd.read_csv('./data/models/uncased-bert/output/anger_dev_out.csv')
  test_df = pd.read_csv('./data/models/uncased-bert/output/anger_test_out.csv')

  train_outputs = torch.Tensor(train_df['Predicted Intensity'].tolist())
  train_intensities = torch.Tensor(train_df['Intensity'].tolist())
  train_pear, train_pear_some, train_kappa, train_kappa_some = evaluate_PerEmotion(train_intensities, train_outputs)
  train_results = {
    'Pearson': train_pear,
    'Pearson SE': train_pear_some,
    'Kappa': train_kappa,
    'Kappa SE': train_kappa_some
  }

  dev_outputs = torch.Tensor(dev_df['Predicted Intensity'].tolist())
  dev_intensities = torch.Tensor(dev_df['Intensity'].tolist())
  dev_pear, dev_pear_some, dev_kappa, dev_kappa_some = evaluate_PerEmotion(dev_intensities, dev_outputs)
  dev_results = {
    'Pearson': dev_pear,
    'Pearson SE': dev_pear_some,
    'Kappa': dev_kappa,
    'Kappa SE': dev_kappa_some
  }

  test_outputs = torch.Tensor(test_df['Predicted Intensity'].tolist())
  test_intensities = torch.Tensor(test_df['Intensity'].tolist())
  test_pear, test_pear_some, test_kappa, test_kappa_some = evaluate_PerEmotion(test_intensities, test_outputs)
  test_results = {
    'Pearson': test_pear,
    'Pearson SE': test_pear_some,
    'Kappa': test_kappa,
    'Kappa SE': test_kappa_some
  }

  print('Train Results', train_results)
  print('Dev Results', dev_results)
  print('Test Results', test_results)

  return train_results, dev_results, test_results

def evaluate_fear_uncased():
  print('\n\nFear')
  train_df = pd.read_csv('./data/models/uncased-bert/output/fear_train_out.csv')
  dev_df = pd.read_csv('./data/models/uncased-bert/output/fear_dev_out.csv')
  test_df = pd.read_csv('./data/models/uncased-bert/output/fear_test_out.csv')

  train_outputs = torch.Tensor(train_df['Predicted Intensity'].tolist())
  train_intensities = torch.Tensor(train_df['Intensity'].tolist())
  train_pear, train_pear_some, train_kappa, train_kappa_some = evaluate_PerEmotion(train_intensities, train_outputs)
  train_results = {
    'Pearson': train_pear,
    'Pearson SE': train_pear_some,
    'Kappa': train_kappa,
    'Kappa SE': train_kappa_some
  }

  dev_outputs = torch.Tensor(dev_df['Predicted Intensity'].tolist())
  dev_intensities = torch.Tensor(dev_df['Intensity'].tolist())
  dev_pear, dev_pear_some, dev_kappa, dev_kappa_some = evaluate_PerEmotion(dev_intensities, dev_outputs)
  dev_results = {
    'Pearson': dev_pear,
    'Pearson SE': dev_pear_some,
    'Kappa': dev_kappa,
    'Kappa SE': dev_kappa_some
  }

  test_outputs = torch.Tensor(test_df['Predicted Intensity'].tolist())
  test_intensities = torch.Tensor(test_df['Intensity'].tolist())
  test_pear, test_pear_some, test_kappa, test_kappa_some = evaluate_PerEmotion(test_intensities, test_outputs)
  test_results = {
    'Pearson': test_pear,
    'Pearson SE': test_pear_some,
    'Kappa': test_kappa,
    'Kappa SE': test_kappa_some
  }

  print('Train Results', train_results)
  print('Dev Results', dev_results)
  print('Test Results', test_results)

  return train_results, dev_results, test_results

def evaluate_sadness_uncased():
  print('\n\nSadness')
  train_df = pd.read_csv('./data/models/uncased-bert/output/sadness_train_out.csv')
  dev_df = pd.read_csv('./data/models/uncased-bert/output/sadness_dev_out.csv')
  test_df = pd.read_csv('./data/models/uncased-bert/output/sadness_test_out.csv')

  train_outputs = torch.Tensor(train_df['Predicted Intensity'].tolist())
  train_intensities = torch.Tensor(train_df['Intensity'].tolist())
  train_pear, train_pear_some, train_kappa, train_kappa_some = evaluate_PerEmotion(train_intensities, train_outputs)
  train_results = {
    'Pearson': train_pear,
    'Pearson SE': train_pear_some,
    'Kappa': train_kappa,
    'Kappa SE': train_kappa_some
  }

  dev_outputs = torch.Tensor(dev_df['Predicted Intensity'].tolist())
  dev_intensities = torch.Tensor(dev_df['Intensity'].tolist())
  dev_pear, dev_pear_some, dev_kappa, dev_kappa_some = evaluate_PerEmotion(dev_intensities, dev_outputs)
  dev_results = {
    'Pearson': dev_pear,
    'Pearson SE': dev_pear_some,
    'Kappa': dev_kappa,
    'Kappa SE': dev_kappa_some
  }

  test_outputs = torch.Tensor(test_df['Predicted Intensity'].tolist())
  test_intensities = torch.Tensor(test_df['Intensity'].tolist())
  test_pear, test_pear_some, test_kappa, test_kappa_some = evaluate_PerEmotion(test_intensities, test_outputs)
  test_results = {
    'Pearson': test_pear,
    'Pearson SE': test_pear_some,
    'Kappa': test_kappa,
    'Kappa SE': test_kappa_some
  }

  print('Train Results', train_results)
  print('Dev Results', dev_results)
  print('Test Results', test_results)

  return train_results, dev_results, test_results

def evaluate_joy_uncased():
  print('\n\nJoy')
  train_df = pd.read_csv('./data/models/uncased-bert/output/joy_train_out.csv')
  dev_df = pd.read_csv('./data/models/uncased-bert/output/joy_dev_out.csv')
  test_df = pd.read_csv('./data/models/uncased-bert/output/joy_test_out.csv')

  train_outputs = torch.Tensor(train_df['Predicted Intensity'].tolist())
  train_intensities = torch.Tensor(train_df['Intensity'].tolist())
  train_pear, train_pear_some, train_kappa, train_kappa_some = evaluate_PerEmotion(train_intensities, train_outputs)
  train_results = {
    'Pearson': train_pear,
    'Pearson SE': train_pear_some,
    'Kappa': train_kappa,
    'Kappa SE': train_kappa_some
  }

  dev_outputs = torch.Tensor(dev_df['Predicted Intensity'].tolist())
  dev_intensities = torch.Tensor(dev_df['Intensity'].tolist())
  dev_pear, dev_pear_some, dev_kappa, dev_kappa_some = evaluate_PerEmotion(dev_intensities, dev_outputs)
  dev_results = {
    'Pearson': dev_pear,
    'Pearson SE': dev_pear_some,
    'Kappa': dev_kappa,
    'Kappa SE': dev_kappa_some
  }

  test_outputs = torch.Tensor(test_df['Predicted Intensity'].tolist())
  test_intensities = torch.Tensor(test_df['Intensity'].tolist())
  test_pear, test_pear_some, test_kappa, test_kappa_some = evaluate_PerEmotion(test_intensities, test_outputs)
  test_results = {
    'Pearson': test_pear,
    'Pearson SE': test_pear_some,
    'Kappa': test_kappa,
    'Kappa SE': test_kappa_some
  }

  print('Train Results', train_results)
  print('Dev Results', dev_results)
  print('Test Results', test_results)

  return train_results, dev_results, test_results

def evaluate_uncased():
  anger_train_results, anger_dev_results, anger_test_results = evaluate_anger_uncased()
  fear_train_results, fear_dev_results, fear_test_results = evaluate_fear_uncased()
  sadness_train_results, sadness_dev_results, sadness_test_results = evaluate_sadness_uncased()
  joy_train_results, joy_dev_results, joy_test_results = evaluate_joy_uncased()

  macro_average_train = {
    'Pearson': (anger_train_results['Pearson'] + fear_train_results['Pearson'] + sadness_train_results['Pearson'] + joy_train_results['Pearson']) / 4,
    
    'Pearson SE': (anger_train_results['Pearson SE'] + fear_train_results['Pearson SE'] + sadness_train_results['Pearson SE'] + joy_train_results['Pearson SE']) / 4,
    
    'Kappa': (anger_train_results['Kappa'] + fear_train_results['Kappa'] + sadness_train_results['Kappa'] + joy_train_results['Kappa']) / 4,
    
    'Kappa SE': (anger_train_results['Kappa SE'] + fear_train_results['Kappa SE'] + sadness_train_results['Kappa SE'] + joy_train_results['Kappa SE']) / 4
  }

  macro_average_dev = {
    'Pearson': (anger_dev_results['Pearson'] + fear_dev_results['Pearson'] + sadness_dev_results['Pearson'] + joy_dev_results['Pearson']) / 4,
    
    'Pearson SE': (anger_dev_results['Pearson SE'] + fear_dev_results['Pearson SE'] + sadness_dev_results['Pearson SE'] + joy_dev_results['Pearson SE']) / 4,
    
    'Kappa': (anger_dev_results['Kappa'] + fear_dev_results['Kappa'] + sadness_dev_results['Kappa'] + joy_dev_results['Kappa']) / 4,
    
    'Kappa SE': (anger_dev_results['Kappa SE'] + fear_dev_results['Kappa SE'] + sadness_dev_results['Kappa SE'] + joy_dev_results['Kappa SE']) / 4
  }

  macro_average_test = {
    'Pearson': (anger_test_results['Pearson'] + fear_test_results['Pearson'] + sadness_test_results['Pearson'] + joy_test_results['Pearson']) / 4,
    
    'Pearson SE': (anger_test_results['Pearson SE'] + fear_test_results['Pearson SE'] + sadness_test_results['Pearson SE'] + joy_test_results['Pearson SE']) / 4,
    
    'Kappa': (anger_test_results['Kappa'] + fear_test_results['Kappa'] + sadness_test_results['Kappa'] + joy_test_results['Kappa']) / 4,
    
    'Kappa SE': (anger_test_results['Kappa SE'] + fear_test_results['Kappa SE'] + sadness_test_results['Kappa SE'] + joy_test_results['Kappa SE']) / 4
  }

  print('\n\nOverall')
  print('Train', macro_average_train)
  print('Dev', macro_average_dev)
  print('Test', macro_average_test)
