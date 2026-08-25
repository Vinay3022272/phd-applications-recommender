import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
HIDDEN_DIM=256

class CandidateTextEncoder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, output_dim=128):
        super().__init__()
        self.text_encoder = SentenceTransformer(MODEL_NAME)
        self.input_dim = self.text_encoder.get_embedding_dimension()

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM, output_dim)
        )

    def forward(self, texts):
        with torch.no_grad():
            embeddings = self.text_encoder.encode(
                texts,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        embeddings = embeddings.to(
            next(self.network.parameters()).device
        )
        out = self.network(embeddings)
        return F.normalize(out, p=2, dim=1)