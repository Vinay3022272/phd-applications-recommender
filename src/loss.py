"""
Corrected Loss functions for Two-Tower training.

FIX: batch_contrastive_loss now accepts `labels` and builds a proper positive mask
instead of assuming diagonal alignment (query[i] matches candidate[i]).
"""

import numpy as np
import torch
import torch.nn.functional as F


class Loss:
    """
    Retrieval Loss functions & Evaluation Metrics for Two-Tower training.
    """

    @staticmethod
    def triplet_margin_loss(q_emb, pos_emb, neg_emb, margin=0.31783701615473536):
        return F.triplet_margin_loss(q_emb, pos_emb, neg_emb, margin=margin)

    @staticmethod
    def batch_triplet_loss(query_embeddings, candidate_embeddings, labels, margin=0.31783701615473536):
        pos_mask = (labels == 1.0)
        neg_mask = (labels == 0.0)

        if not pos_mask.any() or not neg_mask.any():
            neg_embs = torch.roll(candidate_embeddings, shifts=1, dims=0)
            return F.triplet_margin_loss(query_embeddings, candidate_embeddings, neg_embs, margin=margin)

        pos_queries = query_embeddings[pos_mask]
        pos_candidates = candidate_embeddings[pos_mask]
        neg_candidates = candidate_embeddings[neg_mask]

        num_pos = pos_queries.size(0)
        num_neg = neg_candidates.size(0)

        indices = torch.randint(0, num_neg, (num_pos,), device=query_embeddings.device)
        sampled_negs = neg_candidates[indices]

        return F.triplet_margin_loss(pos_queries, pos_candidates, sampled_negs, margin=margin)

    @staticmethod
    def batch_contrastive_loss(query_embeddings, candidate_embeddings, labels, temperature=0.07):
        """
        Corrected InfoNCE contrastive loss for paired (query, candidate, label) data.

        Key insight: In this dataset, query[i] IS paired with candidate[i] (diagonal 
        alignment is correct). BUT only rows where label==1 are positive pairs.
        Rows with label==0 are negative pairs and should be excluded from the loss.

        For each positive pair i:
          - Positive target: candidate[i]  (the diagonal entry)
          - Negatives: all other candidates[j] where j != i

        This is InfoNCE applied only to the positive subset of the batch.
        """
        q_norm = F.normalize(query_embeddings, p=2, dim=-1)
        c_norm = F.normalize(candidate_embeddings, p=2, dim=-1)

        pos_mask = (labels == 1.0)

        # Need at least 2 positive pairs to form a meaningful contrastive batch
        if pos_mask.sum() < 2:
            return torch.tensor(0.0, device=query_embeddings.device, requires_grad=True)

        # Full similarity matrix: (B, B)
        logits = torch.matmul(q_norm, c_norm.T) / temperature

        # Extract rows for positive pairs only
        pos_indices = torch.where(pos_mask)[0]
        pos_logits = logits[pos_indices]  # (num_pos, B)

        # Target: each positive query[i] should match candidate[i] (diagonal)
        # pos_indices tells us which row each positive came from,
        # so the target column index is the same as the row index
        targets = pos_indices  # (num_pos,)

        loss = F.cross_entropy(pos_logits, targets)
        return loss

    @staticmethod
    def l2_regularization_loss(model, weight_decay=1e-4):
        l2_reg = 0.0
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.sum(param ** 2)
        return weight_decay * 0.5 * l2_reg

    @staticmethod
    def calculate_retrieval_metrics(query_embeddings, candidate_embeddings):
        q_norm = F.normalize(query_embeddings, p=2, dim=-1)
        c_norm = F.normalize(candidate_embeddings, p=2, dim=-1)
        similarities = torch.matmul(q_norm, c_norm.T).cpu().numpy()
        num_queries = similarities.shape[0]
        positive_ranks = np.empty(num_queries, dtype=np.int32)
        target_indices = np.arange(num_queries)
        for i in range(num_queries):
            ranking = np.argsort(-similarities[i], kind="mergesort")
            positive_ranks[i] = int(np.where(ranking == target_indices[i])[0][0]) + 1
        return {
            "recall_at_10": float(np.mean(positive_ranks <= 10)),
            "recall_at_50": float(np.mean(positive_ranks <= 50)),
            "recall_at_100": float(np.mean(positive_ranks <= 100)),
            "mrr": float(np.mean(1.0 / positive_ranks))
        }
