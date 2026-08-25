import torch
import torch.nn as nn
import torch.nn.functional as F

CANDIDATE_DIM = 384
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
OUTPUT_DIM = 128

class ProfTower(nn.Module):
  def __init__(self, vocab_dict, dropout=0.2):
    super().__init__()

    self.dense_layers = nn.Sequential(
        nn.Linear(CANDIDATE_DIM, HIDDEN_DIM_1),
        nn.LayerNorm(HIDDEN_DIM_1),
        nn.ReLU(),

        nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
        nn.LayerNorm(HIDDEN_DIM_2),
        nn.ReLU(),

        nn.Linear(HIDDEN_DIM_2, OUTPUT_DIM)
    )

  def forward(self, data):
    all_embs = ""
    out = self.dense_layers(all_embs)
    return F.normalize(out, p=2, dim=-1)


