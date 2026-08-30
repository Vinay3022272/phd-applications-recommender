import torch
import torch.nn as nn
import torch.nn.functional as F

CANDIDATE_DIM = 384
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
OUTPUT_DIM = 128

class ProfTower(nn.Module):
  def __init__(self, text_dim=384, cat_dim=160, dropout=0.2):
    super().__init__()

    input_dim = text_dim + cat_dim

    self.dense_layers = nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM_1),
        nn.LayerNorm(HIDDEN_DIM_1),
        nn.ReLU(),

        nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
        nn.LayerNorm(HIDDEN_DIM_2),
        nn.ReLU(),

        nn.Linear(HIDDEN_DIM_2, OUTPUT_DIM)
    )

  def forward(self, text_emb, cat_emb):
    all_embs = torch.cat([text_emb, cat_emb], dim=-1)
    out = self.dense_layers(all_embs)
    return F.normalize(out, p=2, dim=-1)


