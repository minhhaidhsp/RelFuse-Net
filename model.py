import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch_geometric.nn import SAGEConv
from config import Config


# --- 1. Vision Encoder (DenseNet-121) ---
class VisionEncoder(nn.Module):
    """DenseNet-121 backbone -> Config.PROJ_DIM projection.

    Memory note: this is called on EVERY node RelFuseNet.forward() is given,
    not just the Config.BATCH_SIZE=32 seed admissions -- GraphSAGE needs the
    sampled neighbors' image features too. With GRAPH_LAYERS=2 hops of 10
    neighbors each (train.py's NeighborLoader), one mini-batch's sampled
    subgraph can be an order of magnitude larger than 32 nodes, and running
    DenseNet121 -- whose dense blocks keep concatenating an ever-growing
    feature map across depth (`torch.cat` inside torchvision's
    `bn_function`) -- over all of it in a single forward call is what
    exhausted GPU memory (crash traceback pointed exactly at that `torch.cat`).

    Both mitigations below are pure memory/compute tradeoffs: neither changes
    BATCH_SIZE, GRAPH_LAYERS, num_neighbors, or the forward/backward math --
    same result (up to negligible floating-point reassociation), more compute
    time, bounded peak memory:
      - gradient checkpointing (training only): don't retain every dense
        block's intermediate concatenated feature map for backward: discard
        after the forward pass and recompute from the chunk's input during
        backward instead.
      - chunking: never run more than CHUNK_SIZE images through the backbone
        in one call, regardless of how large the sampled subgraph is.
    """

    CHUNK_SIZE = 64

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121
        self.backbone = densenet121(weights='DEFAULT')
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(num_ftrs, Config.PROJ_DIM)

    def _encode_chunk(self, x_chunk):
        if self.training:
            if x_chunk.is_floating_point() and not x_chunk.requires_grad:
                # checkpoint needs an input that requires grad to attach the
                # backward graph even though only the backbone's parameters
                # (not the raw pixels) actually need gradients here.
                x_chunk = x_chunk.clone().requires_grad_(True)
            return checkpoint(self.backbone, x_chunk, use_reentrant=False)
        with torch.no_grad():
            return self.backbone(x_chunk)

    def forward(self, x):
        if x.shape[0] <= self.CHUNK_SIZE:
            features = self._encode_chunk(x)
        else:
            features = torch.cat(
                [self._encode_chunk(chunk) for chunk in x.split(self.CHUNK_SIZE, dim=0)],
                dim=0,
            )
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

    This encoder is only ever called under Scenario B (retrospective, report-
    assisted). Under Scenario A (prospective, report-free) RelFuseNet.forward
    never calls it at all -- see the scenario branch there -- so no report
    content can leak into the model regardless of what this class does.
    """

    def __init__(self):
        super().__init__()
        if Config.USE_REAL_LLM:
            print(f"[Model] Initializing {Config.LLM_ID} with LoRA...")
            # `load_in_4bit=True` as a direct kwarg to from_pretrained() was removed
            # in newer transformers releases -- it now falls straight through to the
            # underlying model's __init__ instead of being consumed for quantization
            # setup, causing "unexpected keyword argument 'load_in_4bit'". The
            # supported way (and what actually still works) is an explicit
            # BitsAndBytesConfig passed as quantization_config, with device_map so
            # accelerate places the quantized weights directly on the GPU.
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base_model = AutoModel.from_pretrained(
                Config.LLM_ID,
                quantization_config=quant_config,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            # Standard QLoRA prep (casts norms to fp32, enables input grads on the
            # frozen 4-bit base) -- required for stable gradients through a
            # quantized backbone with only the LoRA adapters trainable.
            base_model = prepare_model_for_kbit_training(base_model)
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

    # Memory: RelFuseNet.forward calls this encoder on the ENTIRE GraphSAGE-
    # sampled subgraph, not just the batch's seed nodes -- the graph layers
    # need text features for neighbor nodes too (same rationale as
    # VisionEncoder above). An 8B-parameter LLM forward (hidden_size=4096, up
    # to MAX_LEN=512 tokens, many decoder layers) is far more memory-hungry
    # per sample than DenseNet-121, so a large sampled subgraph can still OOM
    # here even though prepare_model_for_kbit_training() already enables the
    # base model's own internal per-layer gradient checkpointing -- that only
    # bounds CROSS-LAYER activation memory, not the batch-size dimension
    # (e.g. the O(seq_len^2) attention score matrix scales with how many
    # sequences are processed at once). Fix: bound how many sequences go
    # through the LLM in one forward call, the same way VisionEncoder bounds
    # images -- split into chunks of at most CHUNK_SIZE and concatenate the
    # pooled outputs.
    #
    # Deliberately NOT wrapped in an extra torch.utils.checkpoint.checkpoint()
    # call like VisionEncoder's chunks are: input_ids/attention_mask are
    # integer tensors that can never have requires_grad=True, and
    # torch.utils.checkpoint silently drops autograd tracking through a call
    # whose inputs are all non-differentiable -- wrapping it that way would
    # silently break gradients into the LoRA adapters (a correctness bug, far
    # worse than the OOM it would "fix"). The base model's own internal,
    # correctly-wired gradient checkpointing already applies per layer, so no
    # outer checkpoint wrapper is needed -- or safe -- here.
    #
    # CHUNK_SIZE=8 was tried on real GPU L4 (23GB) hardware first and still
    # OOM'd -- "22.00 GiB memory in use" out of 22.03 GiB, missing only 224
    # MiB for one MLP matmul inside a single Llama decoder layer. Lowered to
    # 4: the 8B backbone's constant resident footprint (quantized weights +
    # per-layer checkpointed segment inputs) leaves very little headroom on a
    # 23GB card once VisionEncoder + the graph layers are also holding memory
    # in the same forward/backward pass, so even one chunk of 8 sequences
    # through the LLM was too much. If 4 still isn't enough, try 2 or 1 next
    # -- there is no smaller unit below a single sequence.
    CHUNK_SIZE = 4

    def _encode_chunk(self, ids_chunk, mask_chunk):
        if self.training:
            outputs = self.llm(input_ids=ids_chunk, attention_mask=mask_chunk)
        else:
            with torch.no_grad():
                outputs = self.llm(input_ids=ids_chunk, attention_mask=mask_chunk)
        H = outputs.last_hidden_state  # [chunk, L, d]

        # Mean pooling over real tokens only (Eq. 2), excluding padding via the
        # attention mask -- averaging padded zeros in would bias short reports.
        mask = mask_chunk.unsqueeze(-1).to(H.dtype)  # [chunk, L, 1]
        summed = (H * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count  # [chunk, d]

    def forward(self, input_ids, attention_mask):
        if input_ids.shape[0] <= self.CHUNK_SIZE:
            h_text = self._encode_chunk(input_ids, attention_mask)
        else:
            h_text = torch.cat(
                [
                    self._encode_chunk(ids_chunk, mask_chunk)
                    for ids_chunk, mask_chunk in zip(
                        input_ids.split(self.CHUNK_SIZE, dim=0),
                        attention_mask.split(self.CHUNK_SIZE, dim=0),
                    )
                ],
                dim=0,
            )
        return self.fc(h_text)


# --- 3. MLTM Encoder (Masked Lab-Test Modeling, Section 3.2.2) ---
class MLTMEncoder(nn.Module):
    """
    Self-supervised tabular encoder. `forward` returns the projected feature used
    downstream by the graph/fusion stages, the full reconstruction x_hat used only
    by the Stage-1 pretraining loss (losses.mltm_reconstruction_loss), and the
    artificial-masking indicator a_i actually used, for the caller to pass into
    that loss.

    Three distinct binary indicators (do not conflate them -- this is exactly
    the distinction Referee 2, point #8 asked for):
      o (o_i): observation indicator, 1 = genuinely recorded in the raw EHR.
               Comes from the real data via `observed_mask`; never generated here.
      a (a_i): artificial-masking indicator, 1 = held out at this training step
               and used as a reconstruction target. Generated HERE, sampled only
               from positions where o_i == 1 (so a_i <= o_i elementwise always).
               a_i == 0 everywhere at inference (nothing is held out).
      v (v_i): visibility mask, v_i = o_i * (1 - a_i). This is what the encoder
               is actually allowed to see as input (Eq. 3): x_tilde_i = x_i * v_i.
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
        o = observed_mask
        if training:
            # a_i ~ Bernoulli(rho), restricted to genuinely-observed positions.
            a = torch.bernoulli(torch.full_like(x, Config.MASK_RATIO)) * o
        else:
            a = torch.zeros_like(x)  # inference: nothing held out

        v = o * (1 - a)          # visibility mask (Eq. 3)
        x_in = x * v             # x_tilde_i
        h = self.encoder(x_in)
        x_hat = self.decoder(h)
        h_proj = self.proj(h)
        return h_proj, x_hat, a


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

    def forward(self, img, txt_ids, txt_mask, tab, tab_observed_mask, edge_index,
                training=None, scenario=None):
        is_training = self.training if training is None else training
        # Table 2 / Ablation Study: "A" = prospective, report-free; "B" = full model.
        scenario = Config.SCENARIO if scenario is None else scenario
        if scenario not in ("A", "B"):
            raise ValueError(f"scenario must be 'A' or 'B', got {scenario!r}")

        # 1. Unimodal encoding
        h_v = self.vision_enc(img)
        if scenario == "A":
            # Prospective / report-free: the text encoder is never called, so no
            # report content -- real or otherwise -- can reach the model. This is
            # what makes Scenario A the leakage-safe predictor in the t0 table.
            h_t = torch.zeros(h_v.shape[0], Config.PROJ_DIM, device=h_v.device, dtype=h_v.dtype)
        else:
            h_t = self.text_enc(txt_ids, txt_mask)
        h_tab, x_hat, a_tab = self.mltm_enc(tab, tab_observed_mask, training=is_training)

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
            "tab_artificial_mask": a_tab,  # a_i, for losses.mltm_reconstruction_loss
            "scenario": scenario,
        }
