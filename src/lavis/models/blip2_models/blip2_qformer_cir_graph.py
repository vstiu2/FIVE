"""BLIP2-based CIR model with graph-guided visual enhancement.

Architecture:
  Query side:
    1. Frozen ViT encodes reference image -> image_embeds [B, 1+N, D_vit]
    2. Q-Former encodes image_embeds -> F_reference [B, 32, H]
    3. Q-Former fusion (F_reference + caption) -> z_rm [B, 256]  (TME main path)
    4. Graph-guided selector over patch tokens -> Fl [B, D_vit]
    5. Global mean-pool of patch tokens       -> Fg [B, D_vit]
    6. FFT branch over local region           -> Ff [B, D_vit]
    7. Project Fg/Fl/Ff to embed_dim (256)
    8. TokenFusionBlock(z_rm, Fg, Fl, Ff)     -> final_query [B, 256]

  Target side (unchanged from TME):
    vision_proj(F_target) -> z_target [B, 32, 256]

  Score:
    sim = max_j <final_query, z_target[:,j]>   (TME max-sim)

  Training-only auxiliary head (TgtAwareAuxHead):
    q_tokens cross-attend tgt_tokens.detach() -> q_guided [B, 256]
    L_aux = InfoNCE(q_guided, z_target_mean)

  Total loss:
    L = L_main + lambda_aux * L_aux
"""
from __future__ import annotations

import logging
import sys
import os
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

# Ensure src/model is importable regardless of working directory
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from model.graph_utils import encode_graph_node_features, graph_to_node_texts
from model.reference_visual import GraphGuidedLocalTokenSelector
from model.reference_fft import FixedBoundaryFFTBranch
from model.token_fusion import TokenFusionBlock, TgtAwareAuxHead


@registry.register_model("blip2_cir_graph")
class Blip2QformerCirGraph(Blip2Base):
    """BLIP2 CIR model with graph-guided visual query enhancement."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
    }

    def __init__(
        self,
        vit_model: str = "eva_clip_g",
        img_size: int = 224,
        drop_path_rate: float = 0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        num_query_token: int = 32,
        cross_attention_freq: int = 2,
        embed_dim: int = 256,
        max_txt_len: int = 32,
        # graph / selector
        selector_hidden_dim: int = 512,
        selector_top_k: int = 32,
        selector_dropout: float = 0.1,
        # fusion
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.1,
        # aux head
        aux_head_num_heads: int = 4,
        lambda_aux: float = 0.3,
        # FFT
        fft_band_edges: str = "0,0.15,0.35,0.6,1.01",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_txt_len = max_txt_len
        self.lambda_aux = lambda_aux
        self.num_query_token = num_query_token

        # ── ViT (frozen) ──────────────────────────────────────────────────────
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("ViT frozen.")

        vit_dim = self.visual_encoder.embed_dim  # 1408

        # ── Q-Former ──────────────────────────────────────────────────────────
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                param.data.copy_(state_dict[name.replace("_query", "")])

        qformer_h = self.Qformer.config.hidden_size  # 768

        # ── Projection heads (TME style) ──────────────────────────────────────
        self.vision_proj = nn.Linear(qformer_h, embed_dim)   # target tokens -> 256
        self.text_proj = nn.Linear(qformer_h, embed_dim)     # z_rm -> 256

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # ── Visual enhancement modules (query side only) ──────────────────────
        # Project ViT tokens to a smaller selector space to save memory
        self.visual_to_selector = nn.Linear(vit_dim, selector_hidden_dim)

        self.selector = GraphGuidedLocalTokenSelector(
            vit_embed_dim=selector_hidden_dim,
            node_embed_dim=qformer_h,  # BERT/Q-Former hidden
            hidden_dim=selector_hidden_dim,
            top_k=selector_top_k,
            dropout=selector_dropout,
        )

        self.fft_branch = FixedBoundaryFFTBranch(
            embed_dim=selector_hidden_dim,
            band_edges=fft_band_edges,
        )

        # Project Fg / Fl / Ff from selector_hidden_dim -> embed_dim (256)
        self.global_proj = nn.Linear(selector_hidden_dim, embed_dim)
        self.local_proj = nn.Linear(selector_hidden_dim, embed_dim)
        self.freq_proj = nn.Linear(selector_hidden_dim, embed_dim)

        # ── Token-level cross-attention fusion ────────────────────────────────
        self.token_fusion = TokenFusionBlock(
            embed_dim=embed_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )

        # ── Auxiliary head (training only) ────────────────────────────────────
        self.aux_head = TgtAwareAuxHead(
            qformer_hidden=qformer_h,
            embed_dim=embed_dim,
            num_heads=aux_head_num_heads,
        )

        # Tracks whether graph branches are active (set by training loop)
        self.graph_enabled: bool = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def device(self):
        return next(self.parameters()).device

    def _tokenize(self, texts: Sequence[str]) -> Any:
        return self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

    def _encode_nodes(self, graphs: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """Encode graph node texts with the Q-Former BERT backbone."""
        return encode_graph_node_features(
            tokenizer=self.tokenizer,
            bert_model=self.Qformer.bert,
            graphs=graphs,
            device=self.device,
            max_txt_len=self.max_txt_len,
        )

    # ── Core encode methods ───────────────────────────────────────────────────

    def encode_image(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Q-Former image encoding.  image_embeds already layer-normed."""
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        out = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return out.last_hidden_state  # [B, 32, H]

    def encode_fusion(self, F_image: torch.Tensor, text_tokens: Any) -> torch.Tensor:
        """Q-Former fusion: image tokens + caption -> z_rm [B, 256].

        Returns the text-start position vector (index 32), normalized.
        """
        bs = text_tokens.input_ids.shape[0]
        assert F_image.shape[:2] == (bs, self.num_query_token)
        image_atts = torch.ones(F_image.shape[:-1], dtype=torch.long, device=F_image.device)
        attention_mask = torch.cat([image_atts, text_tokens.attention_mask], dim=1)
        out = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=F_image,
            attention_mask=attention_mask,
            return_dict=True,
        )
        z_rm = F.normalize(
            self.text_proj(out.last_hidden_state[:, self.num_query_token, :]), dim=-1
        )
        q_tokens = out.last_hidden_state[:, : self.num_query_token, :]
        return z_rm, q_tokens  # [B,256], [B,32,H]

    def _visual_branches(
        self, image_embeds_ln: torch.Tensor, graphs: Sequence[Any]
    ) -> Dict[str, torch.Tensor]:
        """Compute Fg / Fl / Ff from reference image embeds (ln applied)."""
        patch_tokens = image_embeds_ln[:, 1:, :]                      # [B, N, 1408]
        sel_tokens = self.visual_to_selector(patch_tokens)             # [B, N, 512]

        node_out = self._encode_nodes(graphs)
        grid_size = int(patch_tokens.shape[1] ** 0.5)
        local_out = self.selector(
            patch_tokens=sel_tokens,
            node_features=node_out["node_features"],
            node_mask=node_out["node_mask"],
            token_hw=(grid_size, grid_size),
        )

        # Mean-pool of raw ViT patch tokens, then project to selector space for Fg
        Fg_raw = patch_tokens.mean(dim=1)                              # [B, 1408]
        Fg_raw = self.visual_to_selector(Fg_raw)                       # [B, 512]

        Ff_raw = self.fft_branch(
            sel_tokens,
            local_out["token_scores"],
            local_out["token_hw"],
        )                                                              # [B, 512]

        return {
            "Fg": F.normalize(self.global_proj(Fg_raw), dim=-1),      # [B, 256]
            "Fl": F.normalize(self.local_proj(local_out["local_feature"]), dim=-1),
            "Ff": F.normalize(self.freq_proj(Ff_raw), dim=-1),
            "token_scores": local_out["token_scores"],
            "node_token_scores": local_out["node_token_scores"],
            "selected_idx": local_out["selected_idx"],
            "token_hw": local_out["token_hw"],
        }

    def _build_query(
        self,
        image_embeds_ln: torch.Tensor,
        captions: List[str],
        graphs: Sequence[Any],
    ) -> torch.Tensor:
        """Shared query construction used by both forward() and inference()."""
        F_reference = self.encode_image(image_embeds_ln)
        text_tokens = self._tokenize(captions)
        z_rm, _ = self.encode_fusion(F_reference, text_tokens)

        if self.graph_enabled:
            vis = self._visual_branches(image_embeds_ln, graphs)
            final_query = self.token_fusion(z_rm, vis["Fg"], vis["Fl"], vis["Ff"])
        else:
            final_query = z_rm

        return F.normalize(final_query, dim=-1)

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(
        self,
        image_embeds_ln: torch.Tensor,  # [B, 1+N, 1408]  ln_vision applied
        F_target_all: torch.Tensor,     # [Ntgt, 32, H]   precomputed gallery
        captions: List[str],
        graphs: Sequence[Any],
    ) -> torch.Tensor:                  # [B, Ntgt]
        final_query = self._build_query(image_embeds_ln, captions, graphs)

        z_target = F.normalize(self.vision_proj(F_target_all), dim=-1)  # [Ntgt, 32, 256]
        sim = torch.matmul(
            final_query.unsqueeze(1).unsqueeze(1),
            z_target.permute(0, 2, 1),
        ).squeeze(2)                    # [B, Ntgt, 32]
        sim, _ = sim.max(dim=-1)        # [B, Ntgt]
        return sim

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        samples: Dict[str, Any],
        warmup: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        samples keys:
            "image"      : ViT features of reference [B, 1+N, 1408]  (precomputed)
            "target"     : ViT features of target    [B, 1+N, 1408]  (precomputed)
            "text_input" : list of caption strings
            "graph"      : list of graph dicts / JSON strings
        warmup: if True, skip graph branches and aux head (only Q-Former fusion)
        """
        # Keep graph_enabled in sync with the warmup flag so inference() always
        # uses the same path that training used in the current epoch.
        self.graph_enabled = not warmup

        image_embeds = samples["image"].to(self.device)
        target_embeds = samples["target"].to(self.device)
        captions = samples["text_input"]
        graphs = samples["graph"]

        with self.maybe_autocast():
            image_embeds_ln = self.ln_vision(image_embeds)
            target_embeds_ln = self.ln_vision(target_embeds)

        F_reference = self.encode_image(image_embeds_ln)              # [B, 32, H]
        F_target = self.encode_image(target_embeds_ln)                # [B, 32, H]
        z_target = F.normalize(self.vision_proj(F_target), dim=-1)   # [B, 32, 256]

        text_tokens = self._tokenize(captions)
        z_rm, q_tokens = self.encode_fusion(F_reference, text_tokens)  # [B,256], [B,32,H]

        if self.graph_enabled:
            vis = self._visual_branches(image_embeds_ln, graphs)
            final_query = self.token_fusion(z_rm, vis["Fg"], vis["Fl"], vis["Ff"])
            final_query = F.normalize(final_query, dim=-1)
        else:
            final_query = F.normalize(z_rm, dim=-1)

        # ── Main loss (InfoNCE, bidirectional) ────────────────────────────────
        loss_dict: Dict[str, torch.Tensor] = {}
        self.temp.data = torch.clamp(self.temp.data, min=1e-2)
        sim = torch.matmul(
            final_query.unsqueeze(1).unsqueeze(1),
            z_target.permute(0, 2, 1),
        ).squeeze(2)                    # [B, B, 32]
        sim, _ = sim.max(dim=-1)        # [B, B]

        gt = torch.arange(sim.shape[0], device=sim.device)
        l_i2t = F.cross_entropy(sim / self.temp, gt)
        l_t2i = F.cross_entropy(sim.t() / self.temp, gt)
        loss_dict["l_main"] = (l_i2t + l_t2i) / 2.0

        # ── Aux loss (tgt-aware, skip during warmup) ──────────────────────────
        if self.graph_enabled:
            q_guided = self.aux_head(q_tokens, F_target.detach())       # [B, 256]
            z_tgt_mean = F.normalize(self.vision_proj(F_target).mean(dim=1), dim=-1)
            sim_aux = q_guided @ z_tgt_mean.t()                         # [B, B]
            l_aux_i2t = F.cross_entropy(sim_aux / self.temp, gt)
            l_aux_t2i = F.cross_entropy(sim_aux.t() / self.temp, gt)
            loss_dict["l_aux"] = (l_aux_i2t + l_aux_t2i) / 2.0
        else:
            loss_dict["l_aux"] = torch.zeros((), device=self.device)

        return loss_dict

    # ── LAVIS config loader ───────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            vit_model=cfg.get("vit_model", "eva_clip_g"),
            img_size=cfg.get("image_size", 224),
            drop_path_rate=cfg.get("drop_path_rate", 0),
            use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
            vit_precision=cfg.get("vit_precision", "fp16"),
            freeze_vit=cfg.get("freeze_vit", True),
            num_query_token=cfg.get("num_query_token", 32),
            cross_attention_freq=cfg.get("cross_attention_freq", 2),
            embed_dim=cfg.get("embed_dim", 256),
            max_txt_len=cfg.get("max_txt_len", 32),
            selector_hidden_dim=cfg.get("selector_hidden_dim", 512),
            selector_top_k=cfg.get("selector_top_k", 32),
            selector_dropout=cfg.get("selector_dropout", 0.1),
            fusion_num_heads=cfg.get("fusion_num_heads", 4),
            fusion_dropout=cfg.get("fusion_dropout", 0.1),
            aux_head_num_heads=cfg.get("aux_head_num_heads", 4),
            lambda_aux=cfg.get("lambda_aux", 0.3),
            fft_band_edges=cfg.get("fft_band_edges", "0,0.15,0.35,0.6,1.01"),
        )
        model.load_checkpoint_from_config(cfg)
        return model
