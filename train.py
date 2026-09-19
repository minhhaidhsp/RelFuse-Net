"""
RelFuse-Net training pipeline (Algorithm 2), corrected to:
  - use a REAL, precomputed ICD/CPT-based training graph (graph_utils.py),
  - attach validation/test admissions INDUCTIVELY (graph_utils.attach_inductive_nodes),
    so no gradient or message ever flows from held-out admissions into training,
  - train with GraphSAGE minibatch neighbor sampling (PyG NeighborLoader), matching
    Algorithm 2's "sample local neighbor set N(u) for each u in V_B" exactly,
  - use class-weighted BCE (Eq. 9) + the real vCLUB mutual-information upper bound
    (Eq. 7) instead of a cosine-orthogonality proxy,
  - run an explicit MLTM self-supervised pretraining stage (Eq. 3-5) before joint
    training, matching the paper's "after convergence" description of h_tab,
  - report PER-CLASS AUROC / AUPRC / F1 in addition to the macro average, and
  - repeat the whole pipeline over Config.NUM_RUNS seeds for mean +/- std (Table 7/8).

Run `preprocessing/build_mimic_dataset.py` first to produce the three CSVs and
the training graph this script expects.
"""
import os
import json
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from config import Config
from model import RelFuseNet
from data_loader import MimicCxrIvDataset
from graph_utils import attach_inductive_nodes
from losses import WeightedBCELoss, VCLUBLoss, mltm_reconstruction_loss

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_indices(dataset: MimicCxrIvDataset, indices):
    """Fetch and batch raw (unencoded) samples for a specific list of global node ids."""
    items = [dataset[i] for i in indices]
    return {
        "image": torch.stack([it["image"] for it in items]),
        "text_ids": torch.stack([it["text_ids"] for it in items]),
        "text_mask": torch.stack([it["text_mask"] for it in items]),
        "tabular": torch.stack([it["tabular"] for it in items]),
        "tabular_observed_mask": torch.stack([it["tabular_observed_mask"] for it in items]),
        "label": torch.stack([it["label"] for it in items]),
    }


def pretrain_mltm(model: RelFuseNet, train_ds: MimicCxrIvDataset):
    """Stage 1 (Eq. 3-5): self-supervised reconstruction pretraining of MLTM alone."""
    print("[MLTM] Stage 1: self-supervised pretraining ...")
    loader = DataLoader(
        list(range(len(train_ds))), batch_size=Config.BATCH_SIZE, shuffle=True
    )
    opt = optim.Adam(model.mltm_enc.parameters(), lr=Config.MLTM_PRETRAIN_LR)

    for epoch in range(Config.MLTM_PRETRAIN_EPOCHS):
        total = 0.0
        for idx_batch in loader:
            batch = collate_indices(train_ds, idx_batch.tolist())
            tab = batch["tabular"].to(Config.DEVICE)
            obs = batch["tabular_observed_mask"].to(Config.DEVICE)

            opt.zero_grad()
            _, x_hat, art_mask = model.mltm_enc(tab, obs, training=True)
            loss = mltm_reconstruction_loss(tab, x_hat, obs, art_mask)
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [MLTM pretrain] epoch {epoch + 1}/{Config.MLTM_PRETRAIN_EPOCHS} loss={total / len(loader):.4f}")


def build_full_graph(edge_index_train: torch.Tensor, n_train: int, extra_edges: torch.Tensor = None, n_extra: int = 0):
    n_total = n_train + n_extra
    if extra_edges is not None and extra_edges.numel() > 0:
        edge_index = torch.cat([edge_index_train, extra_edges], dim=1)
    else:
        edge_index = edge_index_train
    x_placeholder = torch.arange(n_total).unsqueeze(1)  # dummy; real features computed on the fly
    return Data(x=x_placeholder, edge_index=edge_index, num_nodes=n_total)


def run_epoch_train(model, train_ds, edge_index_train, bce, vclubs, vclub_opts, opt, n_train):
    model.train()
    graph = build_full_graph(edge_index_train, n_train)
    loader = NeighborLoader(
        graph,
        num_neighbors=[10] * Config.GRAPH_LAYERS,
        batch_size=Config.BATCH_SIZE,
        input_nodes=torch.arange(n_train),
        shuffle=True,
    )

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        global_ids = batch.n_id.tolist()
        local_edge_index = batch.edge_index.to(Config.DEVICE)
        n_seed = batch.batch_size  # first n_seed rows of n_id are the target nodes

        raw = collate_indices(train_ds, global_ids)
        img = raw["image"].to(Config.DEVICE)
        txt = raw["text_ids"].to(Config.DEVICE)
        mask = raw["text_mask"].to(Config.DEVICE)
        tab = raw["tabular"].to(Config.DEVICE)
        tab_obs = raw["tabular_observed_mask"].to(Config.DEVICE)
        lbl = raw["label"][:n_seed].to(Config.DEVICE)

        opt.zero_grad()
        out = model(img, txt, mask, tab, tab_obs, local_edge_index, training=True)

        logits_seed = out["logits"][:n_seed]
        loss_cls = bce(logits_seed, lbl)

        z_s, z_img, z_txt, z_tab = out["z_shared"], out["z_img"], out["z_text"], out["z_tab"]
        loss_disentangle = (
            vclubs["img"](z_s, z_img) + vclubs["text"](z_s, z_txt) + vclubs["tab"](z_s, z_tab)
        )
        loss = loss_cls + Config.DISENTANGLE_BETA * loss_disentangle
        loss.backward()
        opt.step()

        # Separately fit each vCLUB variational network (its own optimizer, detached inputs).
        for key, z_spec in (("img", z_img), ("text", z_txt), ("tab", z_tab)):
            vclub_opts[key].zero_grad()
            ll = vclubs[key].learning_loss(z_s.detach(), z_spec.detach())
            ll.backward()
            vclub_opts[key].step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, train_ds, eval_ds, edge_index_train, n_train):
    """Inductive evaluation: eval_ds nodes are attached ONLY to training nodes."""
    model.eval()
    extra_edges = attach_inductive_nodes(
        train_ds.icd_histories, train_ds.cpt_codes,
        eval_ds.icd_histories, eval_ds.cpt_codes,
        Config.CPT_OVERLAP_THRESHOLD,
    )
    n_eval = len(eval_ds)
    graph = build_full_graph(edge_index_train, n_train, extra_edges, n_eval)

    loader = NeighborLoader(
        graph,
        num_neighbors=[10] * Config.GRAPH_LAYERS,
        batch_size=Config.BATCH_SIZE,
        input_nodes=torch.arange(n_train, n_train + n_eval),
        shuffle=False,
    )

    all_logits, all_labels = [], []
    for batch in loader:
        global_ids = batch.n_id.tolist()
        local_edge_index = batch.edge_index.to(Config.DEVICE)
        n_seed = batch.batch_size

        # ids < n_train come from train_ds; ids >= n_train come from eval_ds (offset).
        train_ids = [g for g in global_ids if g < n_train]
        eval_ids_local = [g - n_train for g in global_ids if g >= n_train]

        raw_train = collate_indices(train_ds, train_ids) if train_ids else None
        raw_eval = collate_indices(eval_ds, eval_ids_local) if eval_ids_local else None

        def _cat(key):
            parts = []
            if raw_train is not None:
                parts.append(raw_train[key])
            if raw_eval is not None:
                parts.append(raw_eval[key])
            return torch.cat(parts, dim=0)

        # NeighborLoader keeps n_id order consistent with the concatenation order it
        # sampled in; for a from-scratch implementation, re-sort by original position
        # in global_ids to guarantee correctness rather than assuming ordering here.
        order = {gid: i for i, gid in enumerate(train_ids + [n_train + e for e in eval_ids_local])}
        perm = [order[g] for g in global_ids]

        img = _cat("image")[perm].to(Config.DEVICE)
        txt = _cat("text_ids")[perm].to(Config.DEVICE)
        mask = _cat("text_mask")[perm].to(Config.DEVICE)
        tab = _cat("tabular")[perm].to(Config.DEVICE)
        tab_obs = _cat("tabular_observed_mask")[perm].to(Config.DEVICE)
        lbl = _cat("label")[perm][:n_seed]

        out = model(img, txt, mask, tab, tab_obs, local_edge_index, training=False)
        all_logits.append(out["logits"][:n_seed].cpu())
        all_labels.append(lbl)

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


def run_one_seed(seed: int):
    set_seed(seed)
    print(f"\n=== Run with seed={seed} ===")

    tokenizer_id = Config.LLM_ID if Config.USE_REAL_LLM else "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = MimicCxrIvDataset(Config.CSV_TRAIN, Config.IMG_DIR, tokenizer)
    val_ds = MimicCxrIvDataset(Config.CSV_VAL, Config.IMG_DIR, tokenizer)
    test_ds = MimicCxrIvDataset(Config.CSV_TEST, Config.IMG_DIR, tokenizer)

    if not os.path.exists(Config.GRAPH_EDGES_TRAIN):
        raise FileNotFoundError(
            f"{Config.GRAPH_EDGES_TRAIN} not found; run preprocessing/build_mimic_dataset.py first."
        )
    edge_index_train = torch.load(Config.GRAPH_EDGES_TRAIN)
    n_train = len(train_ds)

    model = RelFuseNet().to(Config.DEVICE)

    pretrain_mltm(model, train_ds)

    pos_weight = WeightedBCELoss.compute_pos_weight(
        torch.stack([train_ds[i]["label"] for i in range(len(train_ds))])
    )
    bce = WeightedBCELoss(pos_weight).to(Config.DEVICE)

    vclubs = {k: VCLUBLoss(Config.PROJ_DIM, Config.VCLUB_HIDDEN).to(Config.DEVICE) for k in ("img", "text", "tab")}
    vclub_opts = {k: optim.Adam(v.parameters(), lr=1e-3) for k, v in vclubs.items()}

    main_params = [p for n, p in model.named_parameters()]
    opt = optim.AdamW(main_params, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Config.EPOCHS, eta_min=Config.LR_MIN)

    best_val_auc = 0.0
    for epoch in range(Config.EPOCHS):
        train_loss = run_epoch_train(model, train_ds, edge_index_train, bce, vclubs, vclub_opts, opt, n_train)
        scheduler.step()

        val_metrics = evaluate(model, train_ds, val_ds, edge_index_train, n_train)
        val_auc = val_metrics["macro"]["auc"]
        print(f"Epoch {epoch + 1}/{Config.EPOCHS} | train_loss={train_loss:.4f} | val_macro_AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"relfusenet_best_seed{seed}.pth")

    model.load_state_dict(torch.load(f"relfusenet_best_seed{seed}.pth"))
    test_metrics = evaluate(model, train_ds, test_ds, edge_index_train, n_train)
    print(f"[Seed {seed}] TEST macro AUC={test_metrics['macro']['auc']:.4f} "
          f"F1={test_metrics['macro']['f1']:.4f} AUPRC={test_metrics['macro']['auprc']:.4f}")
    return test_metrics


def main():
    all_runs = []
    for seed in range(Config.SEED, Config.SEED + Config.NUM_RUNS):
        all_runs.append(run_one_seed(seed))

    macro_aucs = [r["macro"]["auc"] for r in all_runs]
    macro_f1s = [r["macro"]["f1"] for r in all_runs]
    print("\n=== Final (mean +/- std over {} runs) ===".format(Config.NUM_RUNS))
    print(f"Macro AUC: {np.mean(macro_aucs):.4f} +/- {np.std(macro_aucs):.4f}")
    print(f"Macro F1:  {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")

    with open("relfusenet_results.json", "w") as f:
        json.dump(all_runs, f, indent=2)
    print("Full per-class results for every run saved to relfusenet_results.json")


if __name__ == "__main__":
    main()
