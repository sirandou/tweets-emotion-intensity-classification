import time
import torch
import numpy as np
from transformers import AdamW
from src.constants.constant import DEVICE
import datetime
from src.evaluate.evaluation import evaluate_PerEmotion
from src.models.bert.bert_hybrid import BertHybridClassifier
from src.data.load_cleaned_data import get_all_bert_data_loader
import pandas as pd

def train_model(model, train_dataloader, validation_dataloader, filepath='', lr=2e-5, EPOCHS=5, BATCH_SIZE=1, weight_decay=0.9, drop_out=0.1):
  training_stats = []
  torch.cuda.empty_cache()
  model = model.to(DEVICE)
  optimizer = AdamW(model.parameters(), lr = lr, weight_decay=weight_decay)
  loss_func = torch.nn.NLLLoss()
  
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

      probas  = model(input_ids, vec, attention_masks)
      loss = loss_func(probas, intensity)

      total_train_loss += loss.item()
      loss.backward()
      torch.torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      print('Epoch: ', epoch_num + 1)
      print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_dataloader) / BATCH_SIZE, total_train_loss / (step_num + 1)))
    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = str(datetime.timedelta(seconds=int(round(((time.time() - t0))))))
    model.eval()

    total_eval_loss = 0
    anger_emotion_intensity = []
    anger_emotion_output = []
    fear_emotion_intensity = []
    fear_emotion_output = []
    sadness_emotion_intensity = []
    sadness_emotion_output = []
    joy_emotion_intensity = []
    joy_emotion_output = []
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
        probas  = model(input_ids, vec, attention_masks)
        output = torch.max(probas, 1)[1]

        loss = loss_func(probas, intensity)

        total_eval_loss += loss.item()

        output = output.detach().cpu()
        intensity = intensity.to('cpu')
        
        for i, a in enumerate(anger.to('cpu')):
          if int(a) == 1:
            anger_emotion_intensity.append(intensity[i])
            anger_emotion_output.append(output[i])
        
        for i, a in enumerate(fear.to('cpu')):
          if int(a) == 1:
            fear_emotion_intensity.append(intensity[i])
            fear_emotion_output.append(output[i])
        
        for i, a in enumerate(sadness.to('cpu')):
          if int(a) == 1:
            sadness_emotion_intensity.append(intensity[i])
            sadness_emotion_output.append(output[i])
        
        for i, a in enumerate(joy.to('cpu')):
          if int(a) == 1:
            joy_emotion_intensity.append(intensity[i])
            joy_emotion_output.append(output[i])
    
    anger_emotion_intensity = torch.Tensor(anger_emotion_intensity)
    anger_emotion_output = torch.Tensor(anger_emotion_output)
    fear_emotion_intensity = torch.Tensor(fear_emotion_intensity)
    fear_emotion_output = torch.Tensor(fear_emotion_output)
    sadness_emotion_intensity = torch.Tensor(sadness_emotion_intensity)
    sadness_emotion_output = torch.Tensor(sadness_emotion_output)
    joy_emotion_intensity = torch.Tensor(joy_emotion_intensity)
    joy_emotion_output = torch.Tensor(joy_emotion_output)  
    
    pear_fear, pear_some_fear, kappa_fear, kappa_some_fear = evaluate_PerEmotion (fear_emotion_intensity, fear_emotion_output)
    pear_joy, pear_some_joy, kappa_joy, kappa_some_joy = evaluate_PerEmotion (joy_emotion_intensity, joy_emotion_output)
    pear_anger, pear_some_anger, kappa_anger, kappa_some_anger = evaluate_PerEmotion (anger_emotion_intensity, anger_emotion_output)
    pear_sadness, pear_some_sadness, kappa_sadness, kappa_some_sadness = evaluate_PerEmotion (sadness_emotion_intensity, sadness_emotion_output)

    pears = np.mean((pear_fear, pear_joy, pear_sadness, pear_anger))
    pears_some = np.mean((pear_some_fear, pear_some_joy, pear_some_sadness, pear_some_anger))
    kappa = np.mean((kappa_fear, kappa_joy, kappa_sadness, kappa_anger))
    kappa_some = np.mean((kappa_some_fear, kappa_some_joy, kappa_some_sadness, kappa_some_anger))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    val_time = str(datetime.timedelta(seconds=int(round(((time.time() - t0))))))

    # Record all statistics from this epoch.
    training_stats.append({
      'epoch': epoch_num + 1,
      'Training Loss': avg_train_loss,
      'Valid. Loss': avg_val_loss,
      
      'Learning Rate': lr,
      'Weight Decay': weight_decay,
      'Drop out Rate': drop_out,
      
      'Pearson': pears,
      'Pearson Some': pears_some,
      'Kappa': kappa,
      'Kappa Some': kappa_some,
      
      'Pearson (Anger)': pear_anger,
      'Pearson some (Anger)': pear_some_anger,
      'Kappa (Anger)': kappa_anger,
      'Kappa some (Anger)': kappa_some_anger,
      
      'Pearson (Fear)': pear_fear,
      'Pearson some (Fear)': pear_some_fear,
      'Kappa (Fear)': kappa_fear,
      'Kappa some (Fear)': kappa_some_fear,
      
      'Pearson (Joy)': pear_joy,
      'Pearson some (Joy)': pear_some_joy,
      'Kappa (Joy)': kappa_joy,
      'Kappa some (Joy)': kappa_some_joy,
      
      'Pearson (Sadness)': pear_sadness,
      'Pearson some (Sadness)': pear_some_sadness,
      'Kappa (Sadness)': kappa_sadness,
      'Kappa some (Sadness)': kappa_some_sadness,
      
      'Training Time': training_time,
      'Validation Time': val_time,
    })

  return training_stats

def train_hybrid():
  train, dev, _ = get_all_bert_data_loader()
  final_training_stats = []

  for drop_out in [0.1, 0.2, 0.3]:
    for lr in [2e-5, 3e-5, 5e-5]:
      for weight_decay in [0.8,0.9,0.95]:
        uncased_model = BertHybridClassifier(dropout=drop_out)
        uncased_trained = train_model(
          uncased_model,
          train,
          dev,
          filepath='uncased',
          lr=lr,
          weight_decay=weight_decay,
          drop_out=drop_out
        )
        final_training_stats.extend(uncased_trained)
  pd.DataFrame(final_training_stats).to_csv('./data/models/hybrid-bert/results/hybrid.csv')