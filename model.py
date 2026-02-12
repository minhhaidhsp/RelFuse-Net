import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from torch_geometric.nn import SAGEConv
from config import Config


# --- 1. Vision Encoder (DenseNet-121) ---
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121
        # Load DenseNet-121 backbone (Pretrained on ImageNet/CheXpert recommended)
        self.backbone = densenet121(weights='DEFAULT')
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(num_ftrs, Config.HIDDEN_DIM)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)


# --- 2. Text Encoder (Medical-Llama3 + LoRA) ---
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        if Config.USE_REAL_LLM:
            print(f"[Model] Initializing {Config.LLM_ID} with LoRA...")
            try:
                # Load Base Model in 4-bit quantization for efficiency
                base_model = AutoModel.from_pretrained(Config.LLM_ID, load_in_4bit=True)

                # Apply LoRA (Section 4.2.1)
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=Config.LORA_R,
                    lora_alpha=Config.LORA_ALPHA,
                    lora_dropout=Config.LORA_DROPOUT
                )
                self.llm = get_peft_model(base_model, peft_config)
                self.embed_dim = 4096  # Llama-3 hidden size
            except Exception as e:
                print(f"Error loading Llama-3: {e}. Fallback to CPU/Small model recommended.")
                raise e
        else:
            print("[Model] Using DistilBERT as proxy (Low Resource Mode).")
            self.llm = AutoModel.from_pretrained("distilbert-base-uncased")
            self.embed_dim = 768

        self.fc = nn.Linear(self.embed_dim, Config.HIDDEN_DIM)

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)

        if Config.USE_REAL_LLM:
            # Llama-3: Use the last token representation
            emb = outputs.last_hidden_state[:, -1, :]
        else:
            # DistilBERT: Use CLS token
            emb = outputs.last_hidden_state[:, 0, :]

        return self.fc(emb)


# --- 3. MLTM Encoder (Masked Lab-Test Modeling) ---
class MLTMEncoder(nn.Module):
    """
    Implements Masked Lab-Test Modeling (Section 3.2.2).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.TABULAR_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, Config.HIDDEN_DIM)
        )

    def forward(self, x, training=True):
        # Apply random masking only during training
        if training:
            mask_prob = 1 - Config.MASK_RATIO  # Keep probability (1 - 0.75 = 0.25)
            # Create binary mask
            mask = torch.bernoulli(torch.full_like(x, mask_prob)).to(x.device)
            # Apply mask and scale up to maintain magnitude
            x = x * mask / mask_prob
        return self.net(x)


# --- 4. RelFuse-Net Integrator ---
class RelFuseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_enc = VisionEncoder()
        self.text_enc = TextEncoder()
        self.mltm_enc = MLTMEncoder()

        # Inductive GraphSAGE (Section 3.3.1)
        # Input: Concatenation of [Vision, Text, Tabular] -> 3 * d
        input_graph_dim = Config.HIDDEN_DIM * 3
        self.graph_conv1 = SAGEConv(input_graph_dim, Config.GRAPH_HIDDEN)
        self.graph_conv2 = SAGEConv(Config.GRAPH_HIDDEN, Config.HIDDEN_DIM)

        # Disentanglement Projectors (Section 3.3.2)
        # Shared Projector (P_s)
        self.proj_shared = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        # Modality-Specific Projectors (P_t, P_tab)
        self.proj_spec_txt = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        self.proj_spec_tab = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)

        # Final Classifier
        # Input: Concat[Shared, Specific_Text, Specific_Tab, Graph_Context]
        fusion_dim = Config.HIDDEN_DIM * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, Config.NUM_CLASSES)
        )

    def forward(self, img, txt_ids, txt_mask, tab, edge_index):
        # 1. Unimodal Encoding
        h_v = self.vision_enc(img)
        h_t = self.text_enc(txt_ids, txt_mask)
        h_tab = self.mltm_enc(tab, training=self.training)

        # 2. Graph Construction & Reasoning
        # Concatenate features to form node embeddings
        node_feats = torch.cat([h_v, h_t, h_tab], dim=1)

        # GraphSAGE Hop 1
        h_g = F.relu(self.graph_conv1(node_feats, edge_index))
        h_g = F.dropout(h_g, p=0.3, training=self.training)
        # GraphSAGE Hop 2
        h_g = self.graph_conv2(h_g, edge_index)
        # L2 Normalization (Algorithm 2)
        h_g = F.normalize(h_g, p=2, dim=1)

        # 3. Disentanglement
        z_shared = self.proj_shared(h_g)  # Shared context from Graph
        z_t = self.proj_spec_txt(h_t)  # Specific Text features
        z_tab = self.proj_spec_tab(h_tab)  # Specific Tabular features

        # 4. Late Fusion
        combined = torch.cat([z_shared, z_t, z_tab, h_g], dim=1)
        logits = self.classifier(combined)

        return logits, z_shared, z_t, z_tab