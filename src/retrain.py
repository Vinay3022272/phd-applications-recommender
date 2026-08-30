"""
Corrected Training Script for Two-Tower PhD-Professor Recommendation Model.

KEY FIX: The contrastive loss now receives `labels` so it knows which pairs are
positive vs negative, instead of assuming diagonal alignment.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.abspath("."))

from model.two_tower import TwoTower
from loss import Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Paths ────────────────────────────────────────────────────────────────────
pt_path = r"C:\Users\ps302\OneDrive\Desktop\Recommend\src\data\final_dataset\labeled_training_dataset.pt"
ckpt_dir = r"C:\Users\ps302\OneDrive\Desktop\Recommend\src\checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "two_tower_checkpoint.pt")


# ── Load Dataset ─────────────────────────────────────────────────────────────
print("Loading dataset...")
loaded_dataset = torch.load(pt_path)
print("Loaded successfully!")
print("Number of training pairs:", len(loaded_dataset["label"]))
print("PhD Text Embeddings Matrix Shape:", loaded_dataset["phd_text_emb"].shape)
print("Prof Text Embeddings Matrix Shape:", loaded_dataset["prof_text_emb"].shape)


# ── Dataset & DataLoader ─────────────────────────────────────────────────────
class MatchingDataset(Dataset):
    """PyTorch Dataset for Two-Tower Applicant-Professor matching."""
    def __init__(self, data_dict):
        self.data = data_dict

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, idx):
        return {
            "scholar_id": self.data["scholar_id"][idx],
            "prof_id": self.data["prof_id"][idx],
            "label": self.data["label"][idx],
            "similarity_score": self.data["similarity_score"][idx],
            "phd_text_emb": self.data["phd_text_emb"][idx],
            "prof_text_emb": self.data["prof_text_emb"][idx],
            "phd_cat_emb": self.data["phd_cat_emb"][idx],
            "prof_cat_emb": self.data["prof_cat_emb"][idx],
            "phd_num_emb": self.data["phd_num_emb"][idx]
        }


full_dataset = MatchingDataset(loaded_dataset)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")


# ── Model ────────────────────────────────────────────────────────────────────
model = TwoTower(
    phd_text_dim=384,
    phd_cat_dim=160,
    phd_num_dim=32,
    prof_text_dim=384,
    prof_cat_dim=160,
    output_dim=128,
    dropout=0.2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# ── Training Functions ───────────────────────────────────────────────────────
def train_one_epoch(
    model, dataloader, optimizer, device,
    loss_mode="combined", margin=0.3178, weight_decay=1e-4, temperature=0.07
):
    model.train()
    total_loss, total_main, total_reg = 0.0, 0.0, 0.0
    for batch in dataloader:
        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        query_embs, candidate_embs = model(batch_data)
        labels = batch_data["label"]

        if loss_mode == "contrastive":
            # FIX: pass labels to contrastive loss
            main_loss = Loss.batch_contrastive_loss(
                query_embs, candidate_embs, labels=labels, temperature=temperature
            )
        elif loss_mode == "triplet":
            main_loss = Loss.batch_triplet_loss(
                query_embs, candidate_embs, labels=labels, margin=margin
            )
        elif loss_mode == "combined":
            # FIX: pass labels to contrastive loss
            contrastive = Loss.batch_contrastive_loss(
                query_embs, candidate_embs, labels=labels, temperature=temperature
            )
            triplet = Loss.batch_triplet_loss(
                query_embs, candidate_embs, labels=labels, margin=margin
            )
            main_loss = 0.5 * contrastive + 0.5 * triplet
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        reg_loss = Loss.l2_regularization_loss(model, weight_decay=weight_decay)
        loss = main_loss + reg_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_main += main_loss.item()
        total_reg += reg_loss.item()

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "main_loss": total_main / num_batches,
        "reg_loss": total_reg / num_batches
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    all_query_embs, all_candidate_embs = [], []
    for batch in dataloader:
        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        query_embs, candidate_embs = model(batch_data)
        all_query_embs.append(query_embs)
        all_candidate_embs.append(candidate_embs)
    return Loss.calculate_retrieval_metrics(
        torch.cat(all_query_embs, dim=0),
        torch.cat(all_candidate_embs, dim=0)
    )


def train_two_tower(
    model, train_loader, optimizer, val_loader=None, epochs=30,
    loss_mode="combined", margin=0.3178, weight_decay=1e-4, temperature=0.07,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    history = {"train_loss": [], "val_mrr": [], "val_recall_10": [], "val_recall_50": []}

    print(f"\nStarting Two-Tower Training [{loss_mode.upper()} Loss] on [{device}] for {epochs} Epochs...")
    print("=" * 80)
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_mode=loss_mode,
            margin=margin,
            weight_decay=weight_decay,
            temperature=temperature
        )
        epoch_loss = train_stats["loss"]
        history["train_loss"].append(epoch_loss)

        val_str = ""
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device)
            history["val_mrr"].append(val_metrics["mrr"])
            history["val_recall_10"].append(val_metrics["recall_at_10"])
            history["val_recall_50"].append(val_metrics["recall_at_50"])
            val_str = (
                f" | Val MRR: {val_metrics['mrr']:.4f}"
                f" | R@10: {val_metrics['recall_at_10']:.4f}"
                f" | R@50: {val_metrics['recall_at_50']:.4f}"
            )
            scheduler.step(epoch_loss)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_loss:.4f} "
            f"(Main: {train_stats['main_loss']:.4f}, Reg: {train_stats['reg_loss']:.4f})"
            f"{val_str}"
        )

    print("=" * 80)
    print("Training Completed!")
    return history


# ── Execute Training ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    history = train_two_tower(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        val_loader=val_loader,
        epochs=30,
        loss_mode="combined",
        margin=0.3178,
        weight_decay=1e-4,
        temperature=0.07,
        device=device
    )

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, ckpt_path)
    print(f"\nModel saved to: {ckpt_path}")
