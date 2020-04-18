import time
import torch
import numpy as np
from transformers import AdamW
from src.constants.constant import DEVICE
from src.util.util import format_time
from src.evaluate.evaluation import evaluate_PerEmotion
from src.models.bert.bert_uncased_classifier import BertForUncasedClassification
from src.models.bert.bert_cased_classifier import BertForCasedClassification
from src.data.load_cleaned_data import get_bert_data_loader
import pandas as pd

def train_model(model, name, train_dataloader, validation_dataloader, filepath='', lr=2e-5, EPOCHS=7, BATCH_SIZE=1, weight_decay=0.9, eps=1e-7):
  training_stats = []
  torch.cuda.empty_cache()
  model = model.to(DEVICE)
  optimizer = AdamW(model.parameters(), lr = lr, weight_decay=weight_decay, eps=eps)
  
  loss_func = torch.nn.NLLLoss()
  training_loss_history = []
  val_loss_history = []
  for epoch_num in range(EPOCHS):
    t0 = time.time()
    model.train()
    total_train_loss = 0
    for step_num, batch_data in enumerate(train_dataloader):
      input_ids, attention_masks, anger, fear, joy, sadness, vec, intensity = tuple(t for t in batch_data)
      
      input_ids = input_ids.to(DEVICE)
      attention_masks = attention_masks.to(DEVICE)
      anger = anger.to(DEVICE)
      fear = fear.to(DEVICE)
      joy = joy.to(DEVICE)
      sadness = sadness.to(DEVICE)
      intensity = intensity.to(DEVICE)
      vec = vec.to(DEVICE)
      
      model.zero_grad()
      
      probas  = model(input_ids, attention_masks)
      loss = loss_func(probas, intensity)

      total_train_loss += loss.item()
      loss.backward()
      torch.torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      # scheduler.step()
      print('Epoch: ', epoch_num + 1)
      print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_dataloader) / BATCH_SIZE, total_train_loss / (step_num + 1)))
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    dropout = model.dropout.p
    parameter_tuned = '_lr_' + str(lr) + '_dropout_' + str(dropout) + '_weight_decay_' + str(weight_decay) +'_eps_' + str(eps)
    model_save_name = filepath + name + '_epoch_' + str(epoch_num) + parameter_tuned + '.pt'
    torch.save(model.state_dict(), model_save_name)
    training_time = format_time(time.time() - t0)
    model.eval()

    total_pearson = 0
    total_pearson_some = 0
    total_kappa = 0
    total_kappa_some = 0
    
    total_eval_loss = 0

    for batch_data in validation_dataloader:
      input_ids, attention_masks, anger, fear, joy, sadness, vec, intensity = tuple(t for t in batch_data)
      
      input_ids = input_ids.to(DEVICE)
      attention_masks = attention_masks.to(DEVICE)
      anger = anger.to(DEVICE)
      fear = fear.to(DEVICE)
      joy = joy.to(DEVICE)
      sadness = sadness.to(DEVICE)
      intensity = intensity.to(DEVICE)
      vec = vec.to(DEVICE)

      with torch.no_grad():
        probas  = model(input_ids, attention_masks)
        output = torch.max(probas, 1)[1]

        loss = loss_func(probas, intensity)
        
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        output = output.detach().cpu()
        intensity = intensity.to('cpu')

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.

        pear, pear_some, kappa, kappa_some = evaluate_PerEmotion(intensity, output)
        
        total_pearson += pear
        total_pearson_some += pear_some
        total_kappa += kappa
        total_kappa_some += kappa_some

    # Report the final accuracy for this validation run.
    avg_pearson = total_pearson / len(validation_dataloader)
    avg_some_pearson = total_pearson_some / len(validation_dataloader)
    avg_kappa = total_kappa / len(validation_dataloader)
    avg_some_kappa = total_kappa_some / len(validation_dataloader)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    val_time = format_time(time.time() - t0)
    
    training_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    
    # Record all statistics from this epoch.
    training_stats.append({
      'epoch': epoch_num + 1,
      'Training Loss': avg_train_loss,
      'Valid. Loss': avg_val_loss,
      'Pearson': avg_pearson,
      'Pearson Some': avg_some_pearson,
      'Kappa': avg_kappa,
      'Kappa Some': avg_some_kappa,
      'Learning Rate': lr,
      'Weight Decay': weight_decay,
      'Dropout': dropout,
      'Epsilon': eps,
      'Training Time': training_time,
      'Validation Time': val_time
    })
  return training_stats, training_loss_history, val_loss_history, parameter_tuned

def train_individual_uncased_anger():
  train, dev, _ = get_bert_data_loader('anger')
  final_training_stats = []

  for d in [0.1, 0.2, 0.3]:
    for w in [0.8, 0.85, 0.9, 0.95]:
      for e in [1e-06, 1e-07, 1e-08]:
        for lr in [2e-5, 3e-5, 5e-5]:
          uncased_model = BertForUncasedClassification(dropout=d)
          uncased_trained, _, _, _ = train_model(
            uncased_model,
            'anger_uncased',
            train,
            dev,
            filepath='./models/anger_uncased/',
            lr=lr,
            eps=e,
            weight_decay=w
          )

          final_training_stats.extend(uncased_trained)

  pd.DataFrame(final_training_stats).to_csv('./data/models/uncased-bert/results/anger/results.csv')

def train_individual_uncased_fear():
  train, dev, _ = get_bert_data_loader('fear')
  final_training_stats = []

  for d in [0.1, 0.2, 0.3]:
    for w in [0.8, 0.85, 0.9, 0.95]:
      for e in [1e-06, 1e-07, 1e-08]:
        for lr in [2e-5, 3e-5, 5e-5]:
          uncased_model = BertForUncasedClassification(dropout=d)
          uncased_trained, _, _, _ = train_model(
            uncased_model,
            'fear_uncased',
            train,
            dev,
            filepath='./models/fear_uncased/',
            lr=lr,
            eps=e,
            weight_decay=w
          )

          final_training_stats.extend(uncased_trained)

  pd.DataFrame(final_training_stats).to_csv('./data/models/uncased-bert/results/fear/results.csv')

def train_individual_uncased_sadness():
  train, dev, _ = get_bert_data_loader('sadness')
  final_training_stats = []

  for d in [0.1, 0.2, 0.3]:
    for w in [0.8, 0.85, 0.9, 0.95]:
      for e in [1e-06, 1e-07, 1e-08]:
        for lr in [2e-5, 3e-5, 5e-5]:
          uncased_model = BertForUncasedClassification(dropout=d)
          uncased_trained, _, _, _ = train_model(
            uncased_model,
            'sadness_uncased',
            train,
            dev,
            filepath='./models/sadness_uncased/',
            lr=lr,
            eps=e,
            weight_decay=w
          )

          final_training_stats.extend(uncased_trained)

  pd.DataFrame(final_training_stats).to_csv('./data/models/uncased-bert/results/sadness/results.csv')

def train_individual_uncased_joy():
  train, dev, _ = get_bert_data_loader('joy')
  final_training_stats = []

  for d in [0.1, 0.2, 0.3]:
    for w in [0.8, 0.85, 0.9, 0.95]:
      for e in [1e-06, 1e-07, 1e-08]:
        for lr in [2e-5, 3e-5, 5e-5]:
          uncased_model = BertForUncasedClassification(dropout=d)
          uncased_trained, _, _, _ = train_model(
            uncased_model,
            'joy_uncased',
            train,
            dev,
            filepath='./models/joy_uncased/',
            lr=lr,
            eps=e,
            weight_decay=w
          )

          final_training_stats.extend(uncased_trained)

  pd.DataFrame(final_training_stats).to_csv('./data/models/uncased-bert/results/joy/results.csv')

def train_uncased():
  train_individual_uncased_anger()
  train_individual_uncased_fear()
  train_individual_uncased_sadness()
  train_individual_uncased_joy()

def train_individual_cased_anger():
  train, dev, _ = get_bert_data_loader('anger', uncased=False)
  final_training_stats = []

  for d in [0.1, 0.2, 0.3]:
    for w in [0.8, 0.85, 0.9, 0.95]:
      for e in [1e-06, 1e-07, 1e-08]:
        for lr in [2e-5, 3e-5, 5e-5]:
          cased_model = BertForCasedClassification(dropout=d)
          cased_trained, _, _, _ = train_model(
            cased_model,
            'anger_cased',
            train,
            dev,
            filepath='./models/anger_cased/',
            lr=lr,
            eps=e,
            weight_decay=w
          )
          final_training_stats.extend(cased_trained)
  pd.DataFrame(final_training_stats).to_csv('./data/models/cased-bert/results/anger/results.csv')

def train_individual_cased_fear():
  train, dev, _ = get_bert_data_loader('fear', uncased=False)
  final_training_stats = []

  for d in [0.1, 0.2, 0.3]:
    for w in [0.8, 0.85, 0.9, 0.95]:
      for e in [1e-06, 1e-07, 1e-08]:
        for lr in [2e-5, 3e-5, 5e-5]:
          cased_model = BertForCasedClassification(dropout=d)
          cased_trained, _, _, _ = train_model(
            cased_model,
            'fear_cased',
            train,
            dev,
            filepath='./models/fear_cased/',
            lr=lr,
            eps=e,
            weight_decay=w
          )
          final_training_stats.extend(cased_trained)
  pd.DataFrame(final_training_stats).to_csv('./data/models/cased-bert/results/fear/results.csv')

def train_individual_cased_sadness():
  train, dev, _ = get_bert_data_loader('sadness', uncased=False)
  final_training_stats = []

  for d in [0.1, 0.2, 0.3]:
    for w in [0.8, 0.85, 0.9, 0.95]:
      for e in [1e-06, 1e-07, 1e-08]:
        for lr in [2e-5, 3e-5, 5e-5]:
          cased_model = BertForCasedClassification(dropout=d)
          cased_trained, _, _, _ = train_model(
            cased_model,
            'sadness_cased',
            train,
            dev,
            filepath='./models/sadness_cased/',
            lr=lr,
            eps=e,
            weight_decay=w
          )
          final_training_stats.extend(cased_trained)
  pd.DataFrame(final_training_stats).to_csv('./data/models/cased-bert/results/sadness/results.csv')

def train_individual_cased_joy():
  train, dev, _ = get_bert_data_loader('joy', uncased=False)
  final_training_stats = []

  for d in [0.1, 0.2, 0.3]:
    for w in [0.8, 0.85, 0.9, 0.95]:
      for e in [1e-06, 1e-07, 1e-08]:
        for lr in [2e-5, 3e-5, 5e-5]:
          cased_model = BertForCasedClassification(dropout=d)
          cased_trained, _, _, _ = train_model(
            cased_model,
            'joy_cased',
            train,
            dev,
            filepath='./models/joy_cased/',
            lr=lr,
            eps=e,
            weight_decay=w
          )
          final_training_stats.extend(cased_trained)
  pd.DataFrame(final_training_stats).to_csv('./data/models/cased-bert/results/joy/results.csv')

def train_cased():
  train_individual_cased_anger()
  train_individual_cased_fear()
  train_individual_cased_sadness()
  train_individual_cased_joy()