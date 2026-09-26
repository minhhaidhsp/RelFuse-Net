"""
Uni-modal baselines from the manuscript's SOTA comparison table
(tab:sota_comparison in main.tex):
  - DenseNet-121 (Huang et al. 2017, bib39): image-only classifier.
  - ClinicalBERT (Alsentzer et al. 2019, bib40): text-only classifier
    (radiology report only).

Both reuse the EXACT SAME data (MimicCxrIvDataset / Config.CSV_TRAIN etc.),
the same 14-label class-weighted BCE (Eq. 13, losses.WeightedBCELoss), and
the same per-class + macro AUROC/AUPRC/F1 metrics + Config.NUM_RUNS-seed
mean+/-std reporting as train.py's RelFuseNet pipeline, so the numbers are
directly comparable to Table tab:sota_comparison's RelFuse-Net row without
re-deriving evaluation logic. Neither baseline uses the ICD/CPT graph,
GraphSAGE, or the vCLUB disentanglement loss -- this script's training loop
is deliberately much simpler than train.py's (plain DataLoader instead of
NeighborLoader, plain AdamW + class-weighted BCE, no graph, no vCLUB),
matching what these two baselines actually are in the paper.

Usage:
    python baselines.py --method densenet121
    python baselines.py --method clinicalbert

Both respect Config.EPOCHS / Config.NUM_RUNS / Config.BATCH_SIZE exactly
like train.py does, so lowering RUN_EPOCHS/RUN_NUM_RUNS in the Colab
notebook's "Run config" cell (which patches config.py) applies here too --
same compute budget, same seeds, for a fair comparison.
"""
import argparse
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from config import Config
from data_loader import MimicCxrIvDataset
from losses import WeightedBCELoss

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

CLINICALBERT_ID = "emilyalsentzer/Bio_ClinicalBERT"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DenseNet121Baseline(nn.Module):
    """Image-only baseline (bib39). Same DenseNet-121 backbone as
    model.py::VisionEncoder, but classifying directly instead of projecting
    into the shared PROJ_DIM fusion space -- there is nothing else to fuse
    with here, by design (this IS the uni-modal ablation)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121
        self.backbone = densenet121(weights="DEFAULT")
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, batch):
        feats = self.backbone(batch["image"].to(Config.DEVICE))
        return self.head(feats)


class ClinicalBERTBaseline(nn.Module):
    """Text-only baseline (bib40): the radiology report only, mean-pooled
    exactly like model.py::TextEncoder (Eq. 2/3), so the pooling strategy is
    not a confound when comparing against RelFuse-Net's own text branch."""

    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(CLINICALBERT_ID)
        hidden = self.bert.config.hidden_size
        self.head = nn.Linear(hidden, Config.NUM_CLASSES)

    def forward(self, batch):
        ids = batch["text_ids"].to(Config.DEVICE)
        mask = batch["text_mask"].to(Config.DEVICE)
        out = self.bert(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


METHODS = {
    "densenet121": DenseNet121Baseline,
    "clinicalbert": ClinicalBERTBaseline,
}

# MimicCxrIvDataset always tokenizes report_text regardless of whether the
# model uses it; densenet121 just gets a cheap, unused tokenization pass.
TOKENIZER_ID = {
    "densenet121": "distilbert-base-uncased",
    "clinicalbert": CLINICALBERT_ID,
}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))

    per_class = {}
    for c, name in enumerate(CHEXPERT_LABELS):
        try:
            auc = roc_auc_score(labels[:, c], probs[:, c])
            auprc = average_precision_score(labels[:, c], probs[:, c])
            f1 = f1_score(labels[:, c], (probs[:, c] > 0.5).astype(int))
        except ValueError:
            auc, auprc, f1 = float("nan"), float("nan"), float("nan")
        per_class[name] = {"auc": auc, "auprc": auprc, "f1": f1}

    macro = {
        "auc": np.nanmean([v["auc"] for v in per_class.values()]),
        "auprc": np.nanmean([v["auprc"] for v in per_class.values()]),
        "f1": np.nanmean([v["f1"] for v in per_class.values()]),
    }
    return {"per_class": per_class, "macro": macro}


def _resume_path(method, seed):
    return f"baseline_resume_{method}_seed{seed}.pt"


def run_one_seed(method: str, seed: int):
    set_seed(seed)
    print(f"\n=== Baseline={method}, seed={seed} ===")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID[method])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = MimicCxrIvDataset(Config.CSV_TRAIN, Config.IMG_DIR, tokenizer)
    val_ds = MimicCxrIvDataset(Config.CSV_VAL, Config.IMG_DIR, tokenizer)
    test_ds = MimicCxrIvDataset(Config.CSV_TEST, Config.IMG_DIR, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = METHODS[method]().to(Config.DEVICE)

    class_weight = WeightedBCELoss.compute_class_weight(
        torch.stack([train_ds[i]["label"] for i in range(len(train_ds))])
    )
    bce = WeightedBCELoss(class_weight).to(Config.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS, eta_min=Config.LR_MIN)

    ckpt_path = f"baseline_best_{method}_seed{seed}.pth"
    resume_path = _resume_path(method, seed)
    start_epoch = 0
    best_val_auc = 0.0
    if os.path.exists(resume_path):
        print(f"[Resume] Found {resume_path} -- resuming instead of restarting from epoch 0.")
        ckpt = torch.load(resume_path, map_location=Config.DEVICE)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt["best_val_auc"]
        random.setstate(ckpt["rng_state"]["python"])
        np.random.set_state(ckpt["rng_state"]["numpy"])
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        print(f"[Resume] Continuing from epoch {start_epoch + 1}/{Config.EPOCHS} "
              f"(best_val_auc so far={best_val_auc:.4f}).")

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            lbl = batch["label"].to(Config.DEVICE)
            opt.zero_grad()
            logits = model(batch)
            loss = bce(logits, lbl)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader)
        val_auc = val_metrics["macro"]["auc"]
        print(f"[{method}] Epoch {epoch + 1}/{Config.EPOCHS} | "
              f"train_loss={total_loss / max(n_batches, 1):.4f} | val_macro_AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), ckpt_path)

        tmp = resume_path + ".part"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_auc": best_val_auc,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }, tmp)
        os.replace(tmp, resume_path)

    if os.path.exists(resume_path):
        os.remove(resume_path)  # finished cleanly; nothing left to resume

    model.load_state_dict(torch.load(ckpt_path))
    test_metrics = evaluate(model, test_loader)
    print(f"[{method}, seed {seed}] TEST macro AUC={test_metrics['macro']['auc']:.4f} "
          f"F1={test_metrics['macro']['f1']:.4f} AUPRC={test_metrics['macro']['auprc']:.4f}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    args = parser.parse_args()

    all_runs = [run_one_seed(args.method, seed) for seed in range(Config.SEED, Config.SEED + Config.NUM_RUNS)]
    macro_aucs = [r["macro"]["auc"] for r in all_runs]
    macro_f1s = [r["macro"]["f1"] for r in all_runs]
    print(f"\n=== {args.method}: Final (mean +/- std over {len(all_runs)} runs) ===")
    print(f"Macro AUC: {np.mean(macro_aucs):.4f} +/- {np.std(macro_aucs):.4f}")
    print(f"Macro F1:  {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")

    out_path = f"baseline_results_{args.method}.json"
    with open(out_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"Full per-class results for every run saved to {out_path}")


if __name__ == "__main__":
    main()
