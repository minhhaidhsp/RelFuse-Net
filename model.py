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
        self.backbone = densenet121(weights='DEFAULT')
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(num_ftrs, Config.PROJ_DIM)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)


# --- 2. Text Encoder (Medical-Llama3 + LoRA), MEAN-POOLED (Eq. 2/3) ---
class TextEncoder(nn.Module):
    """
    Encodes the radiology report with Medical-Llama3-8B (frozen backbone + LoRA)
    and MEAN-POOLS the hidden-state sequence (Eq. 2/3), rather than taking only
    the final token's hidden state. Mean pooling is the deliberate design choice
    described in Section 3.2.1 and must be used for any result reported as the
    paper's findings; last-token pooling is not equivalent and should only be
    used, if ever, as an explicit ablation.
    """

    def __init__(self):
        super().__init__()
        if Config.USE_REAL_LLM:
            print(f"[Model] Initializing {Config.LLM_ID} with LoRA...")
            base_model = AutoModel.from_pretrained(Config.LLM_ID, load_in_4bit=True)
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=Config.LORA_R,
                lora_alpha=Config.LORA_ALPHA,
                lora_dropout=Config.LORA_DROPOUT,
            )
            self.llm = get_peft_model(base_model, peft_config)
            self.embed_dim = 4096  # Llama-3-8B hidden size (Section 3.2.1)
        else:
            print("[Model] USE_REAL_LLM=False: using DistilBERT as a smoke-test proxy. "
                  "Do NOT report results produced this way as the paper's findings.")
            self.llm = AutoModel.from_pretrained("distilbert-base-uncased")
            self.embed_dim = 768

        self.fc = nn.Linear(self.embed_dim, Config.PROJ_DIM)

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # [B, L, d]

        # Mean pooling over real tokens only (Eq. 2), excluding padding via the
        # attention mask -- averaging padded zeros in would bias short reports.
        mask = attention_mask.unsqueeze(-1).to(H.dtype)  # [B, L, 1]
        summed = (H * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        h_text = summed / count  # [B, d]

        return self.fc(h_text)


# --- 3. MLTM Encoder (Masked Lab-Test Modeling, Section 3.2.2) ---
class MLTMEncoder(nn.Module):
    """
    Self-supervised tabular encoder. `forward` returns BOTH the projected feature
    used downstream by the graph/fusion stages, and the full reconstruction x_hat
    used only by the Stage-1 pretraining loss (losses.mltm_reconstruction_loss).

    `observed_mask` (m_i, 1 = genuinely recorded) must come from the real data;
    `artificial_mask` (drawn only from positions where observed_mask == 1) is
    generated here during training to implement Eq. 3's self-supervised task.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(Config.TABULAR_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, Config.MLTM_HIDDEN),
        )
        self.decoder = nn.Linear(Config.MLTM_HIDDEN, Config.TABULAR_DIM)
        self.proj = nn.Linear(Config.MLTM_HIDDEN, Config.PROJ_DIM)

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor, training: bool = True):
        if training:
            keep_prob = 1 - Config.MASK_RATIO
            # Only mask positions that are actually observed; naturally-missing
            # entries are already zero-filled upstream and stay excluded.
            artificial_mask = torch.bernoulli(
                torch.full_like(x, keep_prob)
            ) * observed_mask
        else:
            artificial_mask = observed_mask  # no extra hiding at eval time

        x_in = x * artificial_mask
        h = self.encoder(x_in)
        x_hat = self.decoder(h)
        h_proj = self.proj(h)
        return h_proj, x_hat, artificial_mask


# --- 4. RelFuse-Net Integrator ---
class RelFuseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_enc = VisionEncoder()
        self.text_enc = TextEncoder()
        self.mltm_enc = MLTMEncoder()

        # Inductive GraphSAGE (Section 3.3.1): input = Concat[image, text, tabular]
        input_graph_dim = Config.PROJ_DIM * 3
        self.graph_conv1 = SAGEConv(input_graph_dim, Config.GRAPH_HIDDEN)
        self.graph_conv2 = SAGEConv(Config.GRAPH_HIDDEN, Config.PROJ_DIM)

        # Disentanglement projectors (Section 3.3.2 / Algorithm 2): ALL FOUR take
        # the SAME graph-contextualized representation h_v^graph as input.
        self.proj_shared = nn.Linear(Config.PROJ_DIM, Config.PROJ_DIM)
        self.proj_spec_img = nn.Linear(Config.PROJ_DIM, Config.PROJ_DIM)
        self.proj_spec_txt = nn.Linear(Config.PROJ_DIM, Config.PROJ_DIM)
        self.proj_spec_tab = nn.Linear(Config.PROJ_DIM, Config.PROJ_DIM)

        # Final classifier: Concat(z_s, z_img, z_text, z_tab) -- Eq. 8, no
        # separate raw graph-context term (that would defeat the point of
        # having disentangled it into z_s).
        fusion_dim = Config.PROJ_DIM * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, Config.NUM_CLASSES),
        )

    def forward(self, img, txt_ids, txt_mask, tab, tab_observed_mask, edge_index, training=None):
        is_training = self.training if training is None else training

        # 1. Unimodal encoding
        h_v = self.vision_enc(img)
        h_t = self.text_enc(txt_ids, txt_mask)
        h_tab, x_hat, artificial_mask = self.mltm_enc(tab, tab_observed_mask, training=is_training)

        # 2. Initial node feature z_i = Concat(image, text, tabular) -- Eq. "z_i" (Algorithm 1)
        node_feats = torch.cat([h_v, h_t, h_tab], dim=1)

        # 3. GraphSAGE message passing over the (precomputed, ICD/CPT-based) graph
        h_g = F.relu(self.graph_conv1(node_feats, edge_index))
        h_g = F.dropout(h_g, p=0.3, training=is_training)
        h_g = self.graph_conv2(h_g, edge_index)
        h_g = F.normalize(h_g, p=2, dim=1)  # L2 normalization (Algorithm 2)

        # 4. Disentanglement (Eq. 7/8): all four projectors read h_g
        z_s = self.proj_shared(h_g)
        z_img = self.proj_spec_img(h_g)
        z_text = self.proj_spec_txt(h_g)
        z_tab = self.proj_spec_tab(h_g)

        z_final = torch.cat([z_s, z_img, z_text, z_tab], dim=1)
        logits = self.classifier(z_final)

        return {
            "logits": logits,
            "z_shared": z_s,
            "z_img": z_img,
            "z_text": z_text,
            "z_tab": z_tab,
            "tab_x": tab,
            "tab_x_hat": x_hat,
            "tab_observed_mask": tab_observed_mask,
            "tab_artificial_mask": artificial_mask,
        }
