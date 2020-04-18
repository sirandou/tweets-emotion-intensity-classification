import torch

SEED_VAL = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
