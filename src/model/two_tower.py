import torch
import torch.nn as nn
import torch.nn.functional as F
from query_tower import PHDTower
from candidate_tower import ProfTower

class TwoTower(nn.Module):
  def __init__(self, vocab_dict, dropout=0.2):
    super().__init__()

    self.query_tower = PHDTower(vocab_dict, dropout)
    self.candidate_tower = ProfTower(vocab_dict, dropout)

    self.apply(self.init_weights_xavier)

  def init_weights_xavier(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def forward(self, data):
    query_embs = self.query_tower(data)
    candidate_embs = self.candidate_tower(data)
    return query_embs, candidate_embs


