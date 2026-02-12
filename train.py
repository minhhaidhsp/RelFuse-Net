import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import os
import sys

from config import Config
from model import RelFuseNet
from data_loader import RealMIMICDataset
from utils import construct_clinical_graph, FocalLoss, DisentanglementLoss


def train_pipeline():
    # --- 1. Setup ---
    print("=== RelFuse-Net Training Pipeline ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Loading data from: {Config.CSV_PATH}")

    # Check data existence
    if not os.path.exists(Config.CSV_PATH):
        print(f"ERROR: File {Config.CSV_PATH} not found.")
        print("Please follow instructions in README.md to generate the CSV from your MIMIC dataset.")
        sys.exit(1)

    # Tokenizer
    tokenizer_id = Config.LLM_ID if Config.USE_REAL_LLM else "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    try:
        full_dataset = RealMIMICDataset(Config.CSV_PATH, Config.IMG_DIR, tokenizer)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Split (80/10/10 or 80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # --- 2. Model Initialization ---
    model = RelFuseNet().to(Config.DEVICE)

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    criterion_task = FocalLoss()
    criterion_disentangle = DisentanglementLoss()

    # --- 3. Training Loop ---
    best_auc = 0.0

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            img = batch['image'].to(Config.DEVICE)
            txt = batch['text_ids'].to(Config.DEVICE)
            mask = batch['text_mask'].to(Config.DEVICE)
            tab = batch['tabular'].to(Config.DEVICE)
            lbl = batch['label'].to(Config.DEVICE)

            # Dynamic Inductive Graph Construction
            edge_index = construct_clinical_graph(tab, threshold=Config.SIMILARITY_THRESHOLD)

            optimizer.zero_grad()

            # Forward
            logits, z_shared, z_t, z_tab = model(img, txt, mask, tab, edge_index)

            # Losses
            loss_cls = criterion_task(logits, lbl)
            # Orthogonality Constraint
            loss_ortho = criterion_disentangle(z_shared, z_t) + criterion_disentangle(z_shared, z_tab)

            # Total Loss (Weighted sum)
            loss = loss_cls + 0.1 * loss_ortho

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_preds, val_lbls = [], []

        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(Config.DEVICE)
                txt = batch['text_ids'].to(Config.DEVICE)
                mask = batch['text_mask'].to(Config.DEVICE)
                tab = batch['tabular'].to(Config.DEVICE)

                edge_index = construct_clinical_graph(tab, threshold=Config.SIMILARITY_THRESHOLD)
                logits, _, _, _ = model(img, txt, mask, tab, edge_index)

                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_lbls.append(batch['label'].cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_lbls = np.vstack(val_lbls)

        try:
            auc = roc_auc_score(val_lbls, val_preds, average='macro')
            f1 = f1_score(val_lbls, (val_preds > 0.5).astype(int), average='macro')
        except ValueError:
            auc, f1 = 0.0, 0.0

        print(f"\n>>> Epoch {epoch + 1} Result: AUC = {auc:.4f} | F1 = {f1:.4f}")

        # Save checkpoint
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "relfusenet_best.pth")
            print("Saved Best Model.")


if __name__ == "__main__":
    train_pipeline()