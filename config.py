import torch


class Config:
    """
    Configuration hyperparameters matching the RelFuse-Net paper specifications.
    """
    # --- System Settings ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    NUM_WORKERS = 4

    # --- Path Settings (CRITICAL: User must provide these) ---
    # Reviewers must place the MIMIC data in these directories
    DATA_DIR = "./data"
    CSV_PATH = "./data/mimic_master.csv"  # The master CSV containing paths and labels
    IMG_DIR = "./data/images"  # Directory containing JPG images

    # --- Data Dimensions ---
    IMG_SIZE = 224  # Input resolution for DenseNet-121
    MAX_LEN = 512  # Max sequence length for Medical-Llama3 reports
    TABULAR_DIM = 50  # Number of clinical features (Lab tests, Vitals)
    NUM_CLASSES = 14  # CheXpert label definitions

    # --- Model Architecture ---
    HIDDEN_DIM = 512  # Projection dimension (d)
    GRAPH_HIDDEN = 512  # GraphSAGE hidden units
    GRAPH_LAYERS = 2  # Number of GraphSAGE hops (K=2)

    # --- MLTM Settings ---
    MASK_RATIO = 0.75  # Masking ratio rho (Section 3.2.2)

    # --- Graph Construction ---
    SIMILARITY_THRESHOLD = 0.5  # Threshold tau for edge creation (Section 3.3.1)

    # --- Optimization ---
    BATCH_SIZE = 32  # Effective batch size
    EPOCHS = 20  # Standard training duration
    LR = 2e-4  # Initial learning rate (AdamW)
    WEIGHT_DECAY = 1e-2  # Regularization

    # --- LLM Settings (Medical-Llama3) ---
    # Set to True to load the 8B model (Requires A100 40GB+ GPU)
    # Set to False to use a lightweight proxy (DistilBERT) for debugging/low-resource review
    USE_REAL_LLM = True
    LLM_ID = "meta-llama/Meta-Llama-3-8B"
    LORA_R = 16  # LoRA Rank r
    LORA_ALPHA = 32  # LoRA Alpha
    LORA_DROPOUT = 0.1