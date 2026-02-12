import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import Config


def generate_smart_synthetic_data(num_samples=500):
    """
    Generates synthetic data with embedded patterns to ensure model convergence
    during demonstration (Proof-of-Concept).
    """
    print(f"[DataGen] Generating {num_samples} samples in {Config.DATA_DIR}...")

    os.makedirs(Config.IMG_DIR, exist_ok=True)
    data = []

    for i in tqdm(range(num_samples)):
        # 1. Simulate Label (Binary classification logic for demo convergence)
        # If 'sick', we inject specific patterns into Image, Text, and Tabular
        is_sick = np.random.rand() > 0.5

        # Labels: 14 classes. We force index 0 (e.g., Pneumonia) to match 'is_sick'
        labels = np.zeros(Config.NUM_CLASSES)
        labels[0] = 1.0 if is_sick else 0.0
        # Add random noise to other labels
        if is_sick: labels[1:] = (np.random.rand(Config.NUM_CLASSES - 1) > 0.8).astype(float)

        # 2. Image Pattern (High pixel intensity for sick, low for healthy)
        base_color = np.random.randint(150, 255) if is_sick else np.random.randint(0, 100)
        img_arr = np.full((Config.IMG_SIZE, Config.IMG_SIZE, 3), base_color, dtype=np.uint8)
        # Add noise
        noise = np.random.randint(-20, 20, img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)

        img_name = f"patient_{i}.jpg"
        Image.fromarray(img_arr).save(os.path.join(Config.IMG_DIR, img_name))

        # 3. Text Pattern (Keywords)
        if is_sick:
            report = "Patient shows significant consolidation, opacity, and infiltration."
        else:
            report = "Lungs are clear. No acute cardiopulmonary abnormalities. Normal."

        # 4. Tabular Pattern (Shifted distribution)
        # Sick: Mean +2.0, Healthy: Mean -2.0
        offset = 2.0 if is_sick else -2.0
        features = np.random.randn(Config.TABULAR_DIM) + offset

        data.append({
            'image_path': img_name,
            'report_text': report,
            'clinical_features': ','.join([f"{x:.4f}" for x in features]),
            'labels': ','.join([str(int(x)) for x in labels])
        })

    df = pd.DataFrame(data)
    df.to_csv(Config.CSV_PATH, index=False)
    print(f"[DataGen] Master CSV saved at {Config.CSV_PATH}")


if __name__ == "__main__":
    generate_smart_synthetic_data()