import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_utils.query_tower_utils.text_encoder import TextEncoder
from model_utils.query_tower_utils.categorical_encoder import CategoricalEncoder
from model_utils.query_tower_utils.numerical_encoder import NumercalEncoder

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

with open(r"C:\Users\ps302\OneDrive\Desktop\Recommend\src\checkpoints\vocab_dict.json", "r") as file:
    vocab_dict = json.load(file)

print(vocab_dict)
categorical_maps = vocab_dict.get("categorical_maps")
categorical_tensors = vocab_dict.get("categorical_inputs")
num_cols = vocab_dict.get("numerical_columns")
numeric_tensor = vocab_dict.get("numeric_tensor")
texts = vocab_dict.get("texts")

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")


# Categorical Encoder
categorical_encoder = CategoricalEncoder(categorical_maps, embedding_dim=32).to(device)
categorical_encoder
categorical_embedding = categorical_encoder(categorical_tensors)
print(categorical_embedding.shape)


# Numerical Encoder
numerical_encoder = NumercalEncoder(input_dim=num_cols, output_dim=32).to(device)
numerical_encoder
numerical_embeddings = numerical_encoder(
    numeric_tensor
)
print(numerical_embeddings.shape)


# Text Encoder
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
text_encoder = TextEncoder(model_name=MODEL_NAME, output_dim=128)
texts = texts
text_embeddings = text_encoder(texts)

QUERY_DIM = 384
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
OUTPUT_DIM = 128

class PHDTower(nn.Module):
  def __init__(self, text_dim, categorical_dim, numeric_dim, dropout=0.2):
    super().__init__()

    input_dim = (text_dim + categorical_dim + numeric_dim)

    self.dense_layers = nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM_1),
        nn.LayerNorm(HIDDEN_DIM_1),
        nn.ReLU(),
        nn.Dropout(dropout),

        nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
        nn.LayerNorm(HIDDEN_DIM_2),
        nn.ReLU(),

        nn.Linear(HIDDEN_DIM_2, OUTPUT_DIM)
    )

    def forward(self, text_embedding, categorical_embedding, numeric_embedding):
      all_embs = torch.cat(
        [
          text_embedding, 
          categorical_embedding,
          numeric_embedding
        ],
        dim = 1
      )
      out = self.dense_layers(all_embs)
      return F.normalize(out, p=2, dim=1) #(batch, 128)


"""
query_tower = PHDTower(
    text_dim=384,
    categorical_dim=96,
    numeric_dim=32,
    embedding_dim=128
).to(device)


embeddings = query_tower(
    text_embedding=text_embeddings,
    categorical_embedding=categorical_embedding,
    numeric_embedding=numeric_embedding
)

print(embeddings.shape)

"""