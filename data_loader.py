import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from config import Config


class RealMIMICDataset(Dataset):
    """
    Dataset loader for REAL MIMIC-CXR and MIMIC-III data.

    Prerequisites:
    1. CSV file must contain columns: 'image_path', 'report_text', 'clinical_features', 'labels'.
    2. Images must be extracted to Config.IMG_DIR.
    """

    def __init__(self, csv_path, img_dir, tokenizer, split='train'):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.split = split

        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Master CSV not found at {csv_path}. "
                                    f"Please generate the CSV from MIMIC data.")

        self.data = pd.read_csv(csv_path)
        print(f"[Data] Loaded {len(self.data)} samples from {csv_path}")

        # Standard ImageNet Normalization
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # --- 1. Load Image ---
        img_name = str(row['image_path'])
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Using blank tensor.")
            image = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)

        # --- 2. Load Text Report ---
        text = str(row['report_text'])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_LEN,
            return_tensors='pt'
        )

        # --- 3. Load Clinical Features (Tabular) ---
        # Expecting comma-separated string in CSV: "0.5,1.2,-0.1..."
        try:
            feats = np.fromstring(str(row['clinical_features']), sep=',')
            tabular = torch.tensor(feats, dtype=torch.float32)

            # Handle dimension mismatch (Pad or Trim)
            if len(tabular) < Config.TABULAR_DIM:
                padding = torch.zeros(Config.TABULAR_DIM - len(tabular))
                tabular = torch.cat([tabular, padding])
            else:
                tabular = tabular[:Config.TABULAR_DIM]
        except:
            tabular = torch.zeros(Config.TABULAR_DIM)

        # --- 4. Load Labels ---
        # Expecting comma-separated string: "1,0,0,1..."
        try:
            lbls = np.fromstring(str(row['labels']), sep=',')
            labels = torch.tensor(lbls, dtype=torch.float32)
        except:
            labels = torch.zeros(Config.NUM_CLASSES)

        return {
            'image': image,
            'text_ids': encoding['input_ids'].squeeze(0),
            'text_mask': encoding['attention_mask'].squeeze(0),
            'tabular': tabular,
            'label': labels
        }