import torch
import time
from src.constants.constant import DEVICE
from src.evaluate.evaluation import evaluate_PerEmotion
import datetime

def train(model, train_dataloader, validation_dataloader, filepath='', lr=2e-5, EPOCHS=10, BATCH_SIZE=1, dropout=0.1, hidden_size=20):
  training_stats = []
  model = model.to(DEVICE)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
  loss_func = torch.nn.NLLLoss()

  for epoch_num in range(EPOCHS):
    t0 = time.time()
    model.train()
    total_train_loss = 0
    for _, batch_data in enumerate(train_dataloader):
      tweet = batch_data.Clean_Tweet
      intensity = batch_data.Intensity.to(DEVICE)
      probas  = model(tweet[0].to(DEVICE), tweet[1].to(DEVICE))
      loss = loss_func(probas, intensity)
      total_train_loss += loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = str(datetime.timedelta(seconds=int(round(((time.time() - t0))))))
    model.eval()

    outputs = []
    intensities = []

    total_eval_loss = 0

    for batch_data in validation_dataloader:
      tweet = batch_data.Clean_Tweet
      intensity = batch_data.Intensity.to(DEVICE)
      with torch.no_grad():
        probas  = model(tweet[0].to(DEVICE), tweet[1].to(DEVICE))
        output = torch.max(probas, 1)[1]

        loss = loss_func(probas, intensity)
        
        total_eval_loss += loss.item()
        
        output = output.detach().cpu()
        intensity = intensity.to('cpu')
        
        outputs.extend(list(output.numpy()))
        intensities.extend(list(intensity.numpy()))

    outputs = torch.Tensor(outputs)
    intensities = torch.Tensor(intensities)

    pear, pear_some, kappa, kappa_some = evaluate_PerEmotion(intensities, outputs)
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    val_time = str(datetime.timedelta(seconds=int(round(((time.time() - t0))))))
    
    training_stats.append({
      'epoch': epoch_num + 1,
      'Training Loss': avg_train_loss,
      'Valid. Loss': avg_val_loss,
      'Pearson': pear,
      'Pearson Some': pear_some,
      'Kappa': kappa,
      'Kappa Some': kappa_some,
      'Learning Rate': lr,
      'Dropout': dropout,
      'Hidden Size': hidden_size,
      'Training Time': training_time,
      'Validation Time': val_time
    })

  return training_stats