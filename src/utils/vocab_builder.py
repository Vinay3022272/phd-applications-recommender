import numpy as np
import pandas as pd
import torch
import json
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

scholar_df = pd.DataFrame({
    "scholar_id": [1, 2, 3, 4],
    "research_interests": [
        "deep learning for flood prediction and hydrological modelling",
        "computer vision and medical image analysis",
        "structural health monitoring using machine learning",
        "remote sensing and climate change"
    ],
    "expertise": [
        "deep learning, hydrology, remote sensing",
        "computer vision, deep learning, medical imaging",
        "machine learning, structural engineering, sensors",
        "remote sensing, GIS, climate modelling"
    ],
    "department": ["Civil Engineering","Computer Science","Civil Engineering","Civil Engineering"],
    "university": ["IIT Kharagpur","IIT Delhi","IIT Bombay","IIT Kharagpur"],
    "country": ["India","India","India","India"],
    "publication_count": [  12, 25, 18, 30],
    "citation_count": [ 120, 450, 200, 600],
    "years_experience": [ 2, 5, 3, 7 ]
})

scholar_df

vocab_dict = defaultdict()
categorical_columns = ["department", "university", "country"]


# Categorical data preprocessing
categorical_maps = {}

for col in categorical_columns:
    values = scholar_df[col].fillna("UNKNOWN").astype(str)
    unique_values = sorted(values.unique())
    # mapping  = {value: idx for idx, value in enumerate(unique_values)}
    mapping = {value: idx for idx, value in enumerate(unique_values) }

    # Reserve 0 for UNKNOWN
    mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
    mapping["UNKNOWN"] = 0
    categorical_maps[col] = mapping

vocab_dict["categorical_maps"] = categorical_maps

import os
os.makedirs("src/data/processed/vocab_dict_data", exist_ok=True)
with open("src/data/processed/vocab_dict_data/categorical_maps.json", "w") as file:
    json.dump(categorical_maps, file, indent=4)

# Convert categories → IDs
categorical_ids = {}

for col in categorical_columns:
    mapping = categorical_maps[col]
    categorical_ids[col] = (scholar_df[col].fillna("UNKNOWN").astype(str).map(lambda x: mapping.get(x, 0)).values)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Convert categorical IDs to tensors
categorical_tensors = {}

for col in categorical_columns:
    mapping = categorical_ids[col]
    categorical_tensors[col] = torch.tensor( categorical_ids[col], dtype=torch.long,device=device)

vocab_dict["categorical_inputs"] = {col: tensor.tolist() for col, tensor in categorical_tensors.items()}


# Textual data preprocessing
scholar_df["research_text"] = ( "Research interests: " + scholar_df["research_interests"].fillna("") + ". Expertise: " + scholar_df["expertise"].fillna(""))
texts = scholar_df["research_text"].tolist()
vocab_dict["texts"] = texts

 
# Numerical data preprocessing
numerical_columns = ["publication_count", "citation_count", "years_experience"]
scaler = StandardScaler()
numerical_values = scaler.fit_transform( scholar_df[numerical_columns])

numeric_tensors = torch.tensor( numerical_values, dtype=torch.float32, device=device)
vocab_dict["numeric_tensors"] = numeric_tensors.tolist()
vocab_dict["num_cols"] = len(numerical_columns) 

with open("src/checkpoints/vocab_dict.json", "w") as file:
    json.dump(vocab_dict, file, indent=4)

