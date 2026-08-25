import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class QueryNumercalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        out =  self.network(x)
        return F.normalize(out, p=2, dim=1)