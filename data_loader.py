"""
Dataset for the REAL, preprocessed MIMIC-CXR-JPG + MIMIC-IV cohort produced by
preprocessing/build_mimic_dataset.py. Expected CSV columns per row (= one
admission, i.e. one graph node -- Section 4.1.1's unit of analysis):

  subject_id          : MIMIC-IV patient id
  hadm_id             : MIMIC-IV admission id
  image_path          : path (relative to Config.IMG_DIR) to the SINGLE
                         representative view already selected upstream by
                         priority PA > AP > LATERAL > LL (Section 3.2.1)
  report_text         : radiology report text for that admission's CXR study
  tabular_values      : comma-separated D floats, Z-score normalized;
                         naturally-missing entries are pre-filled with 0.0
                         and MUST be flagged via tabular_observed_mask, never
                         inferred from the value itself
  tabular_observed_mask: comma-separated D 0/1 flags (1 = genuinely recorded)
  icd_history         : ';'-separated ICD codes from admissions that
                         discharged strictly before this admission's admittime
  cpt_codes           : ';'-separated CPT/HCPCS codes from THIS admission,
                         recorded strictly before the CXR acquisition time t0
  labels              : comma-separated 14 0/1 CheXpert labels
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from config import Config


def _parse_float_list(s: str, dim: int) -> np.ndarray:
    if pd.isna(s) or str(s).strip() == "":
        return np.zeros(dim, dtype=np.float32)
    vals = np.fromstring(str(s), sep=",", dtype=np.float32)
    if len(vals) < dim:
        vals = np.concatenate([vals, np.zeros(dim - len(vals), dtype=np.float32)])
    return vals[:dim]


def _parse_code_list(s: str):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [c for c in str(s).split(";") if c]


class MimicCxrIvDataset(Dataset):
    """Loads one split (train/val/test) of the linked MIMIC-CXR + MIMIC-IV cohort."""

    def __init__(self, csv_path: str, img_dir: str, tokenizer):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run preprocessing/build_mimic_dataset.py first "
                f"against your local MIMIC-CXR-JPG + MIMIC-IV downloads."
            )
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        print(f"[Data] Loaded {len(self.data)} admissions from {csv_path}")

        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Pre-parsed once for fast graph construction without re-touching the DataFrame.
        self.icd_histories = [_parse_code_list(v) for v in self.data["icd_history"]]
        self.cpt_codes = [_parse_code_list(v) for v in self.data["cpt_codes"]]
        self.subject_ids = self.data["subject_id"].tolist()
        self.hadm_ids = self.data["hadm_id"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, str(row["image_path"]))
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Warning: could not load {img_path} ({e}); using a blank tensor.")
            image = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)

        encoding = self.tokenizer(
            str(row["report_text"]),
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LEN,
            return_tensors="pt",
        )

        tabular = torch.tensor(
            _parse_float_list(row["tabular_values"], Config.TABULAR_DIM), dtype=torch.float32
        )
        observed_mask = torch.tensor(
            _parse_float_list(row["tabular_observed_mask"], Config.TABULAR_DIM), dtype=torch.float32
        )

        labels = torch.tensor(
            _parse_float_list(row["labels"], Config.NUM_CLASSES), dtype=torch.float32
        )

        return {
            "index": idx,  # local index -> used to look up graph node id
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "image": image,
            "text_ids": encoding["input_ids"].squeeze(0),
            "text_mask": encoding["attention_mask"].squeeze(0),
            "tabular": tabular,
            "tabular_observed_mask": observed_mask,
            "label": labels,
        }
