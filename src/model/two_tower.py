import torch
import torch.nn as nn
import torch.nn.functional as F

class PHDTower(nn.Module):
    """
    Query Tower for PhD Applicants.
    Concatenates text embeddings (384), categorical embeddings (160), and numerical embeddings (32).
    """
    def __init__(self, text_dim=384, cat_dim=160, num_dim=32, hidden_1=512, hidden_2=256, output_dim=128, dropout=0.2):
        super().__init__()
        input_dim = text_dim + cat_dim + num_dim
        
        self.dense_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_2, output_dim)
        )
        
    def forward(self, text_emb, cat_emb, num_emb):
        all_embs = torch.cat([text_emb, cat_emb, num_emb], dim=-1)
        out = self.dense_layers(all_embs)
        return F.normalize(out, p=2, dim=-1)


class ProfTower(nn.Module):
    """
    Candidate Tower for Professors.
    Concatenates text embeddings (384) and categorical embeddings (160).
    """
    def __init__(self, text_dim=384, cat_dim=160, hidden_1=512, hidden_2=256, output_dim=128, dropout=0.2):
        super().__init__()
        input_dim = text_dim + cat_dim
        
        self.dense_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_2, output_dim)
        )
        
    def forward(self, text_emb, cat_emb):
        all_embs = torch.cat([text_emb, cat_emb], dim=-1)
        out = self.dense_layers(all_embs)
        return F.normalize(out, p=2, dim=-1)


class TwoTower(nn.Module):
    """
    Two-Tower Recommendation Model for matching PhD applicants with Professors.
    """
    def __init__(
        self, 
        phd_text_dim=384, phd_cat_dim=160, phd_num_dim=32,
        prof_text_dim=384, prof_cat_dim=160,
        output_dim=128, dropout=0.2
    ):
        super().__init__()
        self.query_tower = PHDTower(
            text_dim=phd_text_dim, 
            cat_dim=phd_cat_dim, 
            num_dim=phd_num_dim, 
            output_dim=output_dim, 
            dropout=dropout
        )
        self.candidate_tower = ProfTower(
            text_dim=prof_text_dim, 
            cat_dim=prof_cat_dim, 
            output_dim=output_dim, 
            dropout=dropout
        )
        
        self.apply(self._init_weights_xavier)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, batch):
        """
        Accepts a batch dictionary from DataLoader and returns (query_embeddings, candidate_embeddings).
        """
        query_embs = self.query_tower(
            batch["phd_text_emb"],
            batch["phd_cat_emb"],
            batch["phd_num_emb"]
        )
        candidate_embs = self.candidate_tower(
            batch["prof_text_emb"],
            batch["prof_cat_emb"]
        )
        return query_embs, candidate_embs

    @torch.no_grad()
    def build_candidate_index(self, prof_candidate_tensors: dict, device: torch.device = None):
        """
        Pre-computes and caches professor candidate embeddings for fast online recommendation lookup.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
            
        prof_text = prof_candidate_tensors["prof_text_emb"].to(device)
        prof_cat = prof_candidate_tensors["prof_cat_emb"].to(device)
        cand_embs = self.candidate_tower(prof_text, prof_cat)
        
        prof_ids = prof_candidate_tensors.get("prof_id", None)
        if isinstance(prof_ids, torch.Tensor):
            prof_ids = prof_ids.cpu().tolist()
            
        return {
            "prof_ids": prof_ids,
            "candidate_embeddings": cand_embs
        }

    @torch.no_grad()
    def predict_recommendations(
        self,
        phd_sample: dict,
        prof_candidate_tensors: dict = None,
        candidate_index: dict = None,
        top_k: int = 10,
        device: torch.device = None
    ):
        """
        Predicts Top-K recommended Professors for a PhD applicant.

        Args:
            phd_sample: dict with keys 'phd_text_emb', 'phd_cat_emb', 'phd_num_emb'
            prof_candidate_tensors: dict with keys 'prof_id', 'prof_text_emb', 'prof_cat_emb' (optional if candidate_index passed)
            candidate_index: precomputed candidate index dict from build_candidate_index() (optional)
            top_k: number of recommendations to return
            device: torch device

        Returns:
            dict with 'top_k_indices', 'top_k_prof_ids', and 'similarity_scores'
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        # Step 1: Query embedding
        phd_text = phd_sample["phd_text_emb"].to(device)
        phd_cat = phd_sample["phd_cat_emb"].to(device)
        phd_num = phd_sample["phd_num_emb"].to(device)

        if phd_text.dim() == 1:
            phd_text = phd_text.unsqueeze(0)
            phd_cat = phd_cat.unsqueeze(0)
            phd_num = phd_num.unsqueeze(0)

        query_emb = self.query_tower(phd_text, phd_cat, phd_num)

        # Step 2: Get candidate embeddings
        if candidate_index is not None:
            cand_embs = candidate_index["candidate_embeddings"].to(device)
            prof_ids = candidate_index.get("prof_ids", None)
        elif prof_candidate_tensors is not None:
            prof_text = prof_candidate_tensors["prof_text_emb"].to(device)
            prof_cat = prof_candidate_tensors["prof_cat_emb"].to(device)
            cand_embs = self.candidate_tower(prof_text, prof_cat)
            prof_ids = prof_candidate_tensors.get("prof_id", None)
            if isinstance(prof_ids, torch.Tensor):
                prof_ids = prof_ids.cpu().tolist()
        else:
            raise ValueError("Either prof_candidate_tensors or candidate_index must be provided.")

        # Step 3: Cosine similarity calculation
        similarities = torch.matmul(query_emb, cand_embs.T).squeeze(0)
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(top_k, len(similarities)))

        top_k_idx_list = top_k_indices.cpu().tolist()
        top_k_score_list = top_k_scores.cpu().tolist()

        result = {
            "top_k_indices": top_k_idx_list,
            "similarity_scores": top_k_score_list
        }

        if prof_ids is not None:
            result["top_k_prof_ids"] = [prof_ids[i] for i in top_k_idx_list]

        return result
