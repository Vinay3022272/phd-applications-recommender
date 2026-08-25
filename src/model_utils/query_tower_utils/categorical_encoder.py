import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ids -> embedding (num_categories.X.32)
class QueryCategoricalEncoder(nn.Module):
    def __init__(self, categorical_maps):
        super().__init__()
        # Fixed typo: embeddigs -> embeddings
        self.embeddings = nn.ModuleDict() 

        for col, mapping in categorical_maps.items():
            num_categories = len(mapping)
            self.embeddings[col] = nn.Embedding(
                num_embeddings=num_categories,
                embedding_dim=32,
                padding_idx=0
            )

    def forward(self, categorical_inputs):
        outputs = []
        for col, embedding_encoder in self.embeddings.items():
            x = categorical_inputs[col]
            emb = embedding_encoder(x)
            outputs.append(emb)
            
        # Concatenate first
        cat_out = torch.cat(outputs, dim=1)
        
        # Apply L2 normalization to match the text and numerical encoders
        return F.normalize(cat_out, p=2, dim=1)