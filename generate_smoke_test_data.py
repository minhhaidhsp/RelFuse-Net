"""
SMOKE TEST ONLY. Generates small, entirely synthetic train/val/test CSVs (in the
exact schema data_loader.py expects) plus a matching training graph, so you can
run `train.py` end-to-end and catch bugs BEFORE spending time on PhysioNet
credentialing and the real preprocessing/build_mimic_dataset.py pipeline.

Results produced from this synthetic data are NOT valid experimental findings
and must never be reported as such -- they exist only to validate that the code
runs without crashing, with correct tensor shapes end to end.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

from config import Config

N_TRAIN, N_VAL, N_TEST = 300, 60, 60
N_ICD_VOCAB, N_CPT_VOCAB = 40, 20


def _make_admission(i, rng):
    is_sick = rng.random() > 0.5
    labels = np.zeros(Config.NUM_CLASSES)
    labels[0] = 1.0 if is_sick else 0.0
    if is_sick:
        labels[1:] = (rng.random(Config.NUM_CLASSES - 1) > 0.8).astype(float)

    base_color = rng.integers(150, 255) if is_sick else rng.integers(0, 100)
    img_arr = np.full((Config.IMG_SIZE, Config.IMG_SIZE, 3), base_color, dtype=np.uint8)
    img_arr = np.clip(img_arr + rng.integers(-20, 20, img_arr.shape), 0, 255).astype(np.uint8)
    img_name = f"admission_{i}.jpg"
    os.makedirs(Config.IMG_DIR, exist_ok=True)
    Image.fromarray(img_arr).save(os.path.join(Config.IMG_DIR, img_name))

    report = ("Patient shows significant consolidation, opacity, and infiltration."
               if is_sick else "Lungs are clear. No acute cardiopulmonary abnormalities.")

    tab_dim = Config.TABULAR_DIM
    observed = (rng.random(tab_dim) > 0.3).astype(np.float32)  # ~30% naturally missing
    values = (rng.standard_normal(tab_dim) + (2.0 if is_sick else -2.0)) * observed

    n_icd = rng.integers(0, 4)
    icd_history = [f"ICD{rng.integers(0, N_ICD_VOCAB)}" for _ in range(n_icd)]
    n_cpt = rng.integers(0, 8)
    cpt_codes = [f"CPT{rng.integers(0, N_CPT_VOCAB)}" for _ in range(n_cpt)]

    return {
        "subject_id": 100000 + i,
        "hadm_id": 200000 + i,
        "image_path": img_name,
        "report_text": report,
        "tabular_values": ",".join(f"{x:.4f}" for x in values),
        "tabular_observed_mask": ",".join(str(int(x)) for x in observed),
        "icd_history": ";".join(icd_history),
        "cpt_codes": ";".join(cpt_codes),
        "labels": ",".join(str(int(x)) for x in labels),
    }


def main():
    rng = np.random.default_rng(Config.SEED)
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

    counter = 0
    splits = {}
    for name, n in (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)):
        rows = []
        for _ in range(n):
            rows.append(_make_admission(counter, rng))
            counter += 1
        splits[name] = pd.DataFrame(rows)

    splits["train"].to_csv(Config.CSV_TRAIN, index=False)
    splits["val"].to_csv(Config.CSV_VAL, index=False)
    splits["test"].to_csv(Config.CSV_TEST, index=False)
    print(f"Wrote synthetic smoke-test CSVs: {N_TRAIN} train / {N_VAL} val / {N_TEST} test.")

    # Build the matching training graph so train.py has something to load.
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from graph_utils import build_patient_graph
    import torch

    def _split_codes(s):
        return [c for c in s.split(";") if c]

    train_icd = [_split_codes(s) for s in splits["train"]["icd_history"]]
    train_cpt = [_split_codes(s) for s in splits["train"]["cpt_codes"]]
    edge_index = build_patient_graph(train_icd, train_cpt, Config.CPT_OVERLAP_THRESHOLD)
    torch.save(edge_index, Config.GRAPH_EDGES_TRAIN)
    print(f"Wrote synthetic training graph with {edge_index.shape[1] // 2} undirected edges "
          f"to {Config.GRAPH_EDGES_TRAIN}")
    print("\nYou can now smoke-test with, e.g.: set Config.USE_REAL_LLM=False and Config.EPOCHS=2, "
          "then run `python train.py`.")


if __name__ == "__main__":
    main()
