
import time
import torch
import numpy as np
from torch import nn

from transformers import AdamW

from src.constants.constant import DEVICE
from src.util.util import format_time
from src.evaluate.evaluation import evaluate_PerEmotion
from src.models.ordinal.bert_ordinal_classifiers import BertForUncasedClassification_1
from src.models.ordinal.bert_ordinal_classifiers import BertForUncasedClassification_2
from src.models.ordinal.bert_ordinal_classifiers import BertForUncasedClassification_3
from src.data.load_cleaned_data import get_bert_data_loader

import pandas as pd


def train(weight, model_num, model, train_dataloader, validation_dataloader, filepath='', lr=2e-5, EPOCHS=10, BATCH_SIZE=1):
  total_t0 = time.time()
  training_stats = []
  model = model.to(DEVICE)
  optimizer = AdamW(model.parameters(), lr = lr)
  
  weight = weight.to(DEVICE)
  loss_func = nn.NLLLoss(weight)
  loss_real = nn.NLLLoss()
  softmax = nn.LogSoftmax(dim=1)
  
  for epoch_num in range(EPOCHS):
    t0 = time.time()
    model.train()
    total_train_loss = 0
    for step_num, batch_data in enumerate(train_dataloader):
      input_ids, attention_masks, anger, fear, joy, sadness, vec, intensity = tuple(t for t in batch_data)
      
      ##ordinal
      o1=torch.tensor((intensity.numpy()>0).astype(int))
      o2=torch.tensor((intensity.numpy()>1).astype(int))
      o3=torch.tensor((intensity.numpy()>2).astype(int))
      
      if model_num == 1:
        o=o1
      if model_num == 2:
        o=o2
      if model_num == 3:
        o=o3
      ###

      input_ids = input_ids.to(DEVICE)
      attention_masks = attention_masks.to(DEVICE)
      anger = anger.to(DEVICE)
      fear = fear.to(DEVICE)
      joy = joy.to(DEVICE)
      sadness = sadness.to(DEVICE)
      intensity = intensity.to(DEVICE)
      vec = vec.to(DEVICE)
      o = o.to(DEVICE)

      model.zero_grad()
      
      probas  = model(input_ids, attention_masks)
      loss = loss_func(probas, o)

      total_train_loss += loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      # scheduler.step()
      print('Epoch: ', epoch_num + 1)
      print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_dataloader) / BATCH_SIZE, total_train_loss / (step_num + 1)))
    avg_train_loss = total_train_loss / len(train_dataloader)

    #model_save_name = filepath + '_epoch_' + str(epoch_num) + '_lr_' + str(lr) + '_' + str(model_num) + '.pt'
    #torch.save(model.state_dict(), model_save_name)
    training_time = format_time(time.time() - t0)
    model.eval()

    total_pearson = 0
    total_kappa = 0
    total_eval_loss_r = 0
    total_eval_loss_w = 0

    for batch_data in validation_dataloader:
      input_ids, attention_masks, anger, fear, joy, sadness, vec, intensity = tuple(t for t in batch_data)
      ##ordinal
      o1=torch.tensor((intensity.numpy()>0).astype(int))
      o2=torch.tensor((intensity.numpy()>1).astype(int))
      o3=torch.tensor((intensity.numpy()>2).astype(int))
      
      if model_num == 1:
        o=o1
      elif model_num ==2:
        o=o2
      else:
        o=o3
      ###
      input_ids = input_ids.to(DEVICE)
      attention_masks = attention_masks.to(DEVICE)
      anger = anger.to(DEVICE)
      fear = fear.to(DEVICE)
      joy = joy.to(DEVICE)
      sadness = sadness.to(DEVICE)
      intensity = intensity.to(DEVICE)
      vec = vec.to(DEVICE)
      o = o.to(DEVICE)

      with torch.no_grad():
        probas  = model(input_ids, attention_masks)
        output = torch.max(probas, 1)[1]

        lossr = loss_real(probas, o)
        lossw = loss_func(probas, o)
        # Accumulate the validation loss.
        total_eval_loss_r += lossr.item()
        total_eval_loss_w += lossw.item()

        output = output.detach().cpu()
        o = o.to('cpu')

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.

        pear, _ , kappa, _ = evaluate_PerEmotion(o, output)
        
        total_pearson += pear
        total_kappa += kappa

    # Report the final accuracy for this validation run.
    avg_pearson = total_pearson / len(validation_dataloader)    
    avg_kappa = total_kappa / len(validation_dataloader)

    # Calculate the average loss over all of the batches.
    avg_val_loss_r = total_eval_loss_r / len(validation_dataloader)
    avg_val_loss_w = total_eval_loss_w / len(validation_dataloader)

    val_time = format_time(time.time() - t0)
    
    # Record all statistics from this epoch.
    training_stats.append({
      'epoch': epoch_num + 1,
      'Training Loss on 1 ordinal': avg_train_loss,
      'Valid. Loss on 1 ordinal, real': avg_val_loss_r,
      'Valid. Loss on 1 ordinal, weighted': avg_val_loss_w,
      'Pearson on 1 ordinal': avg_pearson,
      'Kappa on 1 ordinal': avg_kappa,
      'Learning Rate': lr,
      'Training Time': training_time,
      'Validation Time': val_time
    })

    print(training_stats)

  return training_stats, model


def train_ordinal_fear():
  fear_train_dataloader_uncased, fear_val_dataloader_uncased, fear_test_dataloader_uncased = get_bert_data_loader('fear')
  fear_uncased_model_1 = BertForUncasedClassification_1()
  #fear_cased_model_1 = BertForCasedClassification_1()

  fear_uncased_model_2 = BertForUncasedClassification_2()
  #fear_cased_model_2 = BertForCasedClassification_2()

  fear_uncased_model_3 = BertForUncasedClassification_3()
  #fear_cased_model_3 = BertForCasedClassification_3()

  ### has to be based on the data distribution for each emotion
  class_weights1 = torch.tensor([1,2], dtype = torch.float)
  class_weights2 = torch.tensor([1,4], dtype = torch.float)
  class_weights3 = torch.tensor([1,10], dtype = torch.float)

  final_training_stats_au_1 = []
  for lr in [2e-5, 3e-5, 5e-5]:
    fear_uncased_model_1 = BertForUncasedClassification_1()

    uncased_fear_train_1, model1_fearUncased = train(
      class_weights1,  
      1,
      fear_uncased_model_1,
      fear_train_dataloader_uncased,
      fear_test_dataloader_uncased,
      filepath='fear_uncased_ordinal',
      lr=lr
    )

    final_training_stats_au_1.append(uncased_fear_train_1)
  pd.DataFrame(final_training_stats_au_1[0]).to_csv('./data/models/ordinal/fear/lr_2e-5_model1.csv')
  pd.DataFrame(final_training_stats_au_1[1]).to_csv('./data/models/ordinal/fear/lr_3e-5_model1.csv')
  pd.DataFrame(final_training_stats_au_1[2]).to_csv('./data/models/ordinal/fear/lr_5e-5_model1.csv')
 
  final_training_stats_au_2 = []
  for lr in [2e-5, 3e-5, 5e-5]:
    fear_uncased_model_2 = BertForUncasedClassification_2()

    uncased_fear_train_2, fear_uncased_model2 = train(
      class_weights2,  
      2,
      fear_uncased_model_2,
      fear_train_dataloader_uncased,
      fear_test_dataloader_uncased,
      filepath='fear_uncased_ordinal',
      lr=lr
    )

    final_training_stats_au_2.append(uncased_fear_train_2)

  pd.DataFrame(final_training_stats_au_2[0]).to_csv('./data/models/ordinal/fear/lr_2e-5_model2.csv')
  pd.DataFrame(final_training_stats_au_2[1]).to_csv('./data/models/ordinal/fear/lr_3e-5_model2.csv')
  pd.DataFrame(final_training_stats_au_2[2]).to_csv('./data/models/ordinal/fear/lr_5e-5_model2.csv')


  final_training_stats_au_3 = []
  for lr in [2e-5, 3e-5, 5e-5]:
    fear_uncased_model_3 = BertForUncasedClassification_3()

    uncased_fear_train_3, fear_uncased_model3 = train(
      class_weights3,  
      3,
      fear_uncased_model_3,
      fear_train_dataloader_uncased,
      fear_test_dataloader_uncased,
      filepath='fear_uncased_ordinal',
      lr=lr
    )

    final_training_stats_au_3.append(uncased_fear_train_3)

  pd.DataFrame(final_training_stats_au_3[0]).to_csv('./data/models/ordinal/fear/lr_2e-5_model3.csv')
  pd.DataFrame(final_training_stats_au_3[1]).to_csv('./data/models/ordinal/fear/lr_3e-5_model3.csv')
  pd.DataFrame(final_training_stats_au_3[2]).to_csv('./data/models/ordinal/fear/lr_5e-5_model3.csv')

