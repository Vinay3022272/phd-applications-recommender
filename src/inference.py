"""
Corrected Inference Script for Two-Tower PhD-Professor Recommendation Model.

FIX APPLIED:
- Builds a DEDUPLICATED professor candidate pool from the training dataset
  instead of using all 8520 training-pair rows (which have duplicate professor
  entries and cause index → prof_id mismatch).
- Maps top-k indices back to actual professor IDs correctly.
"""

import os
import sys
import json
import torch
import pandas as pd

# Fix Windows console encoding for unicode characters in data
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\ps302\OneDrive\Desktop\Recommend\src"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "two_tower_checkpoint.pt")
DATASET_PATH = os.path.join(BASE_DIR, "data", "final_dataset", "labeled_training_dataset.pt")
PHD_JSON_PATH = os.path.join(BASE_DIR, "data", "processed", "phd", "sop", "gpt_extracted_data.json")
PROF_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "prof", "extracted_04_prof_data.csv")

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Load Model ───────────────────────────────────────────────────────────────
from model.two_tower import TwoTower

loaded_model = TwoTower(
    phd_text_dim=384,
    phd_cat_dim=160,
    phd_num_dim=32,
    prof_text_dim=384,
    prof_cat_dim=160,
    output_dim=128,
    dropout=0.2
).to(device)

checkpoint = torch.load(CKPT_PATH, map_location=device)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()
print(f"Model loaded from: {CKPT_PATH}")


# ── Load Dataset ─────────────────────────────────────────────────────────────
loaded_dataset = torch.load(DATASET_PATH, map_location=device)
print(f"Loaded {len(loaded_dataset['label'])} training pairs")


# ── Build Deduplicated Professor Candidate Index ─────────────────────────────
def build_unique_candidate_index(dataset):
    """
    Extracts a UNIQUE set of professor candidates from the training pair dataset.

    The training dataset has 8520 rows of (scholar, professor) pairs.
    Each professor appears multiple times (paired with different scholars).
    This function deduplicates by prof_id and returns one embedding per professor.

    Returns:
        unique_prof_ids:  List[int]           – actual professor IDs
        unique_prof_text: Tensor (N_unique, 384) – text embeddings
        unique_prof_cat:  Tensor (N_unique, 160) – categorical embeddings
    """
    prof_ids_all = dataset["prof_id"]
    prof_text_all = dataset["prof_text_emb"]
    prof_cat_all = dataset["prof_cat_emb"]

    # Find the first occurrence index for each unique professor ID
    seen = set()
    first_occurrence_indices = []
    for i, pid in enumerate(prof_ids_all.tolist()):
        if pid not in seen:
            seen.add(pid)
            first_occurrence_indices.append(i)

    idx_tensor = torch.tensor(first_occurrence_indices)

    unique_prof_ids = prof_ids_all[idx_tensor].tolist()
    unique_prof_text = prof_text_all[idx_tensor]
    unique_prof_cat = prof_cat_all[idx_tensor]

    print(f"Deduplicated candidates: {len(prof_ids_all)} pairs -> {len(unique_prof_ids)} unique professors")

    return unique_prof_ids, unique_prof_text, unique_prof_cat


unique_prof_ids, unique_prof_text, unique_prof_cat = build_unique_candidate_index(loaded_dataset)


# ── Inference: Top-K Recommendations ────────────────────────────────────────
@torch.no_grad()
def get_top_k_recommendations(model, dataset, unique_prof_ids, unique_prof_text, unique_prof_cat, scholar_row_idx=0, top_k=10):
    """
    Computes top-K professor recommendations for a given scholar.

    Args:
        model: Trained TwoTower model in eval mode
        dataset: The loaded training dataset dict
        unique_prof_ids: Deduplicated list of professor IDs
        unique_prof_text: Deduplicated professor text embeddings
        unique_prof_cat: Deduplicated professor categorical embeddings
        scholar_row_idx: Row index in the training dataset to use as query
        top_k: Number of recommendations to return

    Returns:
        scholar_id: int
        recommendations: list of (rank, prof_id, score) tuples
    """
    model.eval()

    # 1. Compute UNIQUE candidate embeddings
    prof_embs = model.candidate_tower(
        unique_prof_text.to(device),
        unique_prof_cat.to(device)
    )  # Shape: (N_unique, 128)

    # 2. Compute query embedding for the selected scholar
    phd_text = dataset["phd_text_emb"][scholar_row_idx].unsqueeze(0).to(device)
    phd_cat = dataset["phd_cat_emb"][scholar_row_idx].unsqueeze(0).to(device)
    phd_num = dataset["phd_num_emb"][scholar_row_idx].unsqueeze(0).to(device)
    query_emb = model.query_tower(phd_text, phd_cat, phd_num)  # Shape: (1, 128)

    # 3. Cosine similarity against UNIQUE candidates
    similarities = torch.matmul(query_emb, prof_embs.T).squeeze(0)  # Shape: (N_unique,)
    top_k_scores, top_k_indices = torch.topk(similarities, k=min(top_k, len(similarities)))

    # 4. Map indices → actual professor IDs (CORRECT mapping)
    scholar_id = dataset["scholar_id"][scholar_row_idx].item()
    recommendations = []
    for rank, (idx, score) in enumerate(
        zip(top_k_indices.cpu().tolist(), top_k_scores.cpu().tolist()), start=1
    ):
        prof_id = unique_prof_ids[idx]  # ← Correct: index into deduplicated list
        recommendations.append((rank, prof_id, score))

    return scholar_id, recommendations


# ── Display Recommendations with Profile Details ─────────────────────────────
def display_recommendations(scholar_id, recommendations):
    """Prints recommendations and looks up scholar/professor profiles."""

    # Load PhD scholar profile
    scholar_info = {}
    if os.path.exists(PHD_JSON_PATH):
        with open(PHD_JSON_PATH, "r", encoding="utf-8") as f:
            phd_data = json.load(f)
        phd_map = {item["scholar_id"]: item for item in phd_data if "scholar_id" in item}
        scholar_info = phd_map.get(scholar_id, {})

    # Load Professor profiles
    prof_df = None
    if os.path.exists(PROF_CSV_PATH):
        prof_df = pd.read_csv(PROF_CSV_PATH)

    # Print scholar profile
    print("=" * 70)
    print(f"[SCHOLAR] PROFILE [ID: {scholar_id}]")
    print(f"   Name: {scholar_info.get('name', 'N/A')}")
    print(f"   University: {scholar_info.get('university', 'N/A')}")
    print(f"   Department: {scholar_info.get('department', 'N/A')}")
    print(f"   Research Interests: {scholar_info.get('research_interests', 'N/A')}")
    print("=" * 70)

    # Print top-K recommendations
    print(f"\n[PROFESSOR] TOP-{len(recommendations)} RECOMMENDED PROFESSORS:")
    print("-" * 70)

    rec_prof_ids = [r[1] for r in recommendations]

    if prof_df is not None and "id" in prof_df.columns:
        for rank, prof_id, score in recommendations:
            row = prof_df[prof_df["id"] == prof_id]
            if not row.empty:
                name = row.iloc[0].get("name", "N/A")
                dept = row.iloc[0].get("department", "N/A")
                expertise = row.iloc[0].get("expertise", "N/A")
                # Truncate long expertise strings for readability
                if isinstance(expertise, str) and len(expertise) > 100:
                    expertise = expertise[:100] + "..."
            else:
                name, dept, expertise = "N/A", "N/A", "N/A"

            print(f"  Rank {rank:02d} | ID {prof_id:>5} | Score {score:.4f}")
            print(f"           Name: {name}")
            print(f"           Dept: {dept}")
            print(f"           Expertise: {expertise}")
            print()
    else:
        for rank, prof_id, score in recommendations:
            print(f"  Rank {rank:02d}: Professor ID {prof_id} | Similarity Score: {score:.4f}")


# ── Run Inference ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  TWO-TOWER RECOMMENDATION - CORRECTED INFERENCE")
    print("=" * 70 + "\n")

    # Get recommendations for scholar at row 0 (Scholar ID: 150)
    scholar_id, recommendations = get_top_k_recommendations(
        model=loaded_model,
        dataset=loaded_dataset,
        unique_prof_ids=unique_prof_ids,
        unique_prof_text=unique_prof_text,
        unique_prof_cat=unique_prof_cat,
        scholar_row_idx=0,
        top_k=10
    )

    display_recommendations(scholar_id, recommendations)
