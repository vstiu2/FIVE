"""SSN-BLIP2: Action-Specific Transitions + Multi-Scale Feature Fusion.

Architecture overview
─────────────────────
Query side (reference image + text + graph):

  1. Frozen EVA-ViT-G  →  patch_tokens [B, N, 1408]

  2. Path-U QFormer(original_patches, image+text)  →  Fq_u [B, 256]
     Pretrained anchors: ref_embed (image-only) and text_embed (text_proj).

  3. ActionRouter (hard routing by graph.action → Remove / Add / Move)
       Each transition module returns:
         F_trans      [B, 256]  → enters final_query
         M_obj_ref    [B, N]    → spatial weight for the Ff branch
         F_obj_region [B, 768]  → object-region feature for the aux loss
       (replace is routed to Remove: graph.object is the OLD object, in the ref.)

  4. Multi-scale visual features (from ORIGINAL patch_tokens)
       Fg = global mean-pool             → [B, 256]
       Fl = GraphGuidedLocalSelector     → [B, 256]  (object appearance)
       Ff = FFTBranch(M_obj_ref)         → [B, 256]  (object texture/freq)
     Fq_fused = TokenFusionBlock(Fq_u, [Fg, Fl, Ff])

  5. final_query = α·combiner(text,ref) + β·Fq_fused + γ·F_trans
                 + s·text_embed + (1-s)·ref_embed     (γ = gamma_max·σ(raw_gamma))

Target side:
  QFormer(target_patches, image-only)  →  vision_proj  →  z_target [B, 32, 256]

Loss:
  L_main  = InfoNCE(final_query, z_target)     (max-sim over 32 target tokens)
  L_aux   = InfoNCE(TgtAwareAuxHead(q,tgt), z_target)
  L_trans = L_remove + L_add + L_move          (action-specific, see action_transitions)
  L       = L_main + lambda_aux·L_aux + lambda_group·L_group
  (action-specific transition aux losses removed; localization is provided by
  a frozen grounding-trained ObjectLocalizer that supplies M_obj directly.)
"""
from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── LAVIS / model imports (copied from TME_v1) ───────────────────────────────
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from model.graph_utils import encode_graph_node_features, graph_to_node_texts
from model.reference_visual import GraphGuidedLocalTokenSelector
from model.reference_fft import FixedBoundaryFFTBranch
from model.token_fusion import TokenFusionBlock, TgtAwareAuxHead
from model.action_transitions import (
    ActionRouter,
    action_to_category,
)
from model.object_localizer import ObjectLocalizer, aggregate_mask


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class Model(Blip2Base):
    """SSN-BLIP2 composed image retrieval model."""

    def __init__(
        self,
        # ViT
        vit_model: str = "eva_clip_g",
        img_size: int = 224,
        drop_path_rate: float = 0.0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        # QFormer
        num_query_token: int = 32,
        cross_attention_freq: int = 2,
        embed_dim: int = 256,
        max_txt_len: int = 32,
        # Decomposer / selector
        selector_hidden_dim: int = 512,
        selector_top_k: int = 32,
        selector_dropout: float = 0.1,
        # Fusion
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.1,
        # Aux head
        aux_head_num_heads: int = 4,
        lambda_aux: float = 0.3,
        # Group hard-negative auxiliary loss
        lambda_group: float = 0.1,
        group_scale: float = 20.0,
        # SPRC-style alignment regularizer (MSE between fusion query and text anchor)
        # 0.0 disables it → behavior identical to v11.
        lambda_align: float = 0.0,
        # ComqueryFormer-style multi-scale global contrastive loss on Fq_u
        # (0.0 disables it → v12 behavior)
        lambda_global: float = 0.0,
        # Action-specific transition branch (Remove / Add / Move)
        gamma_init: float = 0.1,         # initial gamma (γ·F_trans contribution)
        gamma_max: float = 0.5,          # upper bound of gamma (sigmoid-parameterized)
        # Frozen ObjectLocalizer (grounding-trained selector)
        selector_ckpt: str = os.environ.get("FIVE_SELECTOR_CKPT", "weight/selector_eva.pt"),
        selector_hidden_dim_loc: int = 512,
        selector_dropout_loc: float = 0.1,
        # ColBERT-style multi-vector query-target token matching loss.
        # 0.0 disables it → previous behavior.
        lambda_colbert: float = 0.0,
        # Label smoothing for InfoNCE losses (0.0 = no smoothing, v12 behavior)
        label_smoothing: float = 0.0,
        # alpha/beta init + clamp upper bound (defaults reproduce v12 behavior:
        # init=0.01, clamp_max=0.5). Raise for new experiments to give the
        # combiner / Fq_fused branches more capacity.
        alpha_init: float = 0.01,
        beta_init: float = 0.01,
        alpha_max: float = 0.5,
        beta_max: float = 0.5,
        # FFT
        fft_band_edges: str = "0,0.15,0.35,0.6,1.01",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_txt_len = max_txt_len
        self.lambda_aux = lambda_aux
        self.lambda_group = lambda_group
        self.lambda_align = lambda_align
        self.lambda_global = lambda_global
        self.gamma_max = gamma_max
        # Fixed (non-learnable) scale for the group-aware auxiliary head; lower
        # than self.temp's effective scale=100 so hard-neg gradients are bounded.
        self.register_buffer('group_scale', torch.tensor(float(group_scale)))
        self.lambda_colbert = lambda_colbert
        self.label_smoothing = label_smoothing
        self.alpha_max = alpha_max
        self.beta_max = beta_max
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

        vit_dim = self.visual_encoder.embed_dim   # 1408

        # ── Q-Former (shared weights, two sets of query_tokens) ──────────────
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                param.data.copy_(state_dict[name.replace("_query", "")])

        qformer_h = self.Qformer.config.hidden_size   # 768

        # ── Projection heads ─────────────────────────────────────────────────
        # vision_proj: keep pretrained 768→256 weights (loaded from BLIP2 ckpt).
        # text_proj: keep pretrained 768→256 weights (loaded from BLIP2 ckpt).
        # Both project into the same pretrained 256-dim ITC space.
        self.vision_proj = nn.Linear(qformer_h, embed_dim)
        self.text_proj = nn.Linear(qformer_h, embed_dim)
        proj_dim = embed_dim                                      # 256 — same as pretrained

        self.temp = nn.Parameter(0.01 * torch.ones([]))   # init scale=100, matching SSN logit_scale

        # ── SSN-style linear combiner ────────────────────────────────────────
        # output = alpha * combiner_out + beta * Fq_fused + s * text_embed + (1-s) * ref_embed
        combiner_input_dim = proj_dim * 2    # concat(text_embed, ref_embed) = 512
        # dynamic_scalar gets text_embed + ref_embed + Fq_fused (visual context helps decide blend ratio)
        scalar_input_dim = proj_dim * 3      # 768
        self.combiner_layer = nn.Linear(combiner_input_dim, proj_dim * 4)
        self.combiner_output = nn.Linear(proj_dim * 4, proj_dim)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(scalar_input_dim, proj_dim * 4),
            nn.ReLU(),
            nn.Linear(proj_dim * 4, 1),
            nn.Sigmoid(),
        )
        # Learnable scales for combiner and visual fusion contributions.
        # Defaults (init=0.01, clamp_max=0.5) reproduce v12 behavior. New
        # experiments can raise init / clamp_max to let the multi-scale
        # visual fusion branch (Fq_fused) contribute more.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

        # ── gamma: weight of the transition branch in final_query ─────────────
        # γ = gamma_max * sigmoid(raw_gamma)  (smooth & bounded, never zero-grad).
        # raw_gamma initialised so γ ≈ gamma_init at start.
        g0 = min(max(gamma_init, 1e-4), gamma_max - 1e-4)
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(g0 / (gamma_max - g0))))

        # ── Action-specific transition router (Remove / Add / Move) ───────────
        self.action_router = ActionRouter(
            vit_dim=vit_dim,
            node_dim=qformer_h,
            hidden_dim=selector_hidden_dim,
            proj_dim=proj_dim,
            dropout=selector_dropout,
        )

        # ── Frozen grounding-trained ObjectLocalizer ──────────────────────────
        # Provides accurate M_obj for the transition modules and the Ff branch.
        self.object_localizer = ObjectLocalizer(
            vit_dim=vit_dim, node_dim=qformer_h,
            hidden_dim=selector_hidden_dim_loc, dropout=selector_dropout_loc,
        )
        if selector_ckpt and os.path.isfile(selector_ckpt):
            ck = torch.load(selector_ckpt, map_location="cpu")
            sd = ck.get("model", ck)
            missing, unexpected = self.object_localizer.load_state_dict(sd, strict=False)
            logging.info(f"loaded ObjectLocalizer from {selector_ckpt} "
                         f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            logging.warning(f"selector_ckpt not found: {selector_ckpt!r}. "
                            "ObjectLocalizer will run with random weights — "
                            "localization will be unreliable.")
        for p in self.object_localizer.parameters():
            p.requires_grad = False
        self.object_localizer = self.object_localizer.eval()
        self.object_localizer.train = disabled_train

        # ── Visual enhancement modules (use ORIGINAL patch_tokens) ────────────
        self.visual_to_selector = nn.Linear(vit_dim, selector_hidden_dim)

        # GraphGuidedLocalTokenSelector expects selector_hidden_dim inputs
        self.selector = GraphGuidedLocalTokenSelector(
            vit_embed_dim=selector_hidden_dim,
            node_embed_dim=qformer_h,
            hidden_dim=selector_hidden_dim,
            top_k=selector_top_k,
            dropout=selector_dropout,
        )
        self.fft_branch = FixedBoundaryFFTBranch(
            embed_dim=selector_hidden_dim,
            band_edges=fft_band_edges,
        )
        self.global_proj = nn.Linear(selector_hidden_dim, proj_dim)
        self.local_proj  = nn.Linear(selector_hidden_dim, proj_dim)
        self.freq_proj   = nn.Linear(selector_hidden_dim, proj_dim)

        # ── TokenFusionBlock ─────────────────────────────────────────────────
        self.token_fusion = TokenFusionBlock(
            embed_dim=proj_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )

        # (dynamic_scalar removed — Fq_fused is used directly as final_query.
        #  Full gradients flow from l_pos into Fg/Fl/Ff immediately.)

        # ── Auxiliary head (degradation path supervision) ─────────────────────
        self.aux_head = TgtAwareAuxHead(
            qformer_hidden=qformer_h,
            embed_dim=proj_dim,
            num_heads=aux_head_num_heads,
        )

        self._proj_dim = proj_dim

        # Aux-branch ramp: training loop sets this to a value in [0, 1] during
        # the first few graph epochs to phase in beta/gamma smoothly (and avoid
        # injecting random-MLP noise at the moment graph branches first turn on).
        # 1.0 = no ramp (default).
        self.register_buffer("aux_ramp", torch.tensor(1.0), persistent=False)

        # Set by training loop to toggle graph branches
        self.graph_enabled: bool = False

        # ── Ablation switches (default all off → bit-identical to baseline) ──
        # Toggled via set_ablation(...) from train / test CLI flags.
        self.ablate_no_trans: bool = False     # zero out γ·F_trans
        self.ablate_no_fl: bool = False        # zero out Fl in vis dict
        self.ablate_no_ff: bool = False        # zero out Ff in vis dict
        self.ablate_force_route: str = ""      # "" | "Remove" | "Add" | "Move"

    def set_ablation(
        self,
        no_trans: bool = False,
        no_fl: bool = False,
        no_ff: bool = False,
        force_route: str = "",
    ) -> None:
        """Configure ablation switches. All flags default to False/empty so
        a model loaded from a baseline checkpoint and never reconfigured is
        bit-identical to the un-ablated version."""
        self.ablate_no_trans = bool(no_trans)
        self.ablate_no_fl = bool(no_fl)
        self.ablate_no_ff = bool(no_ff)
        if force_route not in ("", "Remove", "Add", "Move"):
            raise ValueError(
                f"force_route must be '', 'Remove', 'Add' or 'Move'; got {force_route!r}"
            )
        self.ablate_force_route = force_route

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def gamma(self):
        """Sigmoid-parameterized transition weight γ ∈ (0, gamma_max)."""
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    @staticmethod
    def _parse_graph(g: Any) -> Dict[str, Any]:
        import json
        if isinstance(g, str):
            try:
                g = json.loads(g) if g.strip() else {}
            except json.JSONDecodeError:
                g = {}
        return g if isinstance(g, dict) else {}

    def _tokenize(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

    def _encode_nodes(self, graphs: Sequence[Any]) -> Dict[str, torch.Tensor]:
        return encode_graph_node_features(
            tokenizer=self.tokenizer,
            bert_model=self.Qformer.bert,
            graphs=graphs,
            device=self.device,
            max_txt_len=self.max_txt_len,
        )

    def _encode_field(self, graphs: Sequence[Any], field: str) -> torch.Tensor:
        """Encode a single top-level graph field (e.g. action/location) as text.

        Returns the BERT CLS embedding, [B, H]. Missing/unknown → 'unknown'.
        """
        texts = []
        for g in graphs:
            g = self._parse_graph(g)
            val = g.get(field) or "unknown"
            texts.append(f"graph {field}: {val}")
        tokens = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_txt_len, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.Qformer.bert(
                tokens.input_ids, attention_mask=tokens.attention_mask, return_dict=True,
            )
        return out.last_hidden_state[:, 0, :]   # CLS, [B, H]

    def _encode_object_for_localizer(self, graphs: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """Encode ONLY the object as a single node, matching the selector's
        training text format ("graph node: {label}"). Falls back to the first
        graph node label when the 'object' field is missing/unknown."""
        mini = []
        for g in graphs:
            g = self._parse_graph(g)
            obj = g.get("object")
            if not obj or str(obj).strip().lower() == "unknown":
                nodes = g.get("nodes") or []
                obj = (nodes[0].get("label") if nodes and isinstance(nodes[0], dict) else None)
            mini.append({"nodes": [{"label": obj or "unknown"}]})
        return encode_graph_node_features(
            tokenizer=self.tokenizer, bert_model=self.Qformer.bert,
            graphs=mini, device=self.device, max_txt_len=self.max_txt_len,
        )

    def _encode_obj_nodes(
        self, graphs: Sequence[Any]
    ) -> Dict[str, torch.Tensor]:
        """Encode only object/position nodes (exclude action pseudo-node)."""
        import json
        all_texts = []
        obj_counts = []
        for g in graphs:
            if isinstance(g, str):
                try:
                    g = json.loads(g) if g.strip() else {}
                except json.JSONDecodeError:
                    g = {}
            texts: List[str] = []
            seen_labels: set = set()
            action_label = None
            if isinstance(g, dict):
                action_label = g.get("action")
                if isinstance(action_label, str):
                    action_label = action_label.strip().lower()
                else:
                    action_label = None

                for node in g.get("nodes", []):
                    if not isinstance(node, dict):
                        continue
                    label = node.get("label") or node.get("text") or node.get("name") or "unknown"
                    label_norm = str(label).strip().lower()
                    # Drop the action node here; it's encoded separately via
                    # _encode_field("action"), so keeping it would dilute obj features.
                    if action_label is not None and label_norm == action_label:
                        continue
                    if label_norm in seen_labels:
                        continue
                    seen_labels.add(label_norm)
                    texts.append(f"graph node: {label}")
                for field in ("object", "location", "result_location"):
                    val = g.get(field)
                    if not val or val == "unknown":
                        continue
                    val_norm = str(val).strip().lower()
                    if val_norm in seen_labels:
                        continue
                    seen_labels.add(val_norm)
                    texts.append(f"graph {field}: {val}")
            if not texts:
                texts = ["graph node: unknown"]
            obj_counts.append(len(texts))
            all_texts.extend(texts)

        max_m = max(obj_counts)
        B = len(graphs)
        # Pad by repeating last node text
        padded: List[str] = []
        mask_list = []
        offsets = [0] + list(__import__('itertools').accumulate(obj_counts))
        for i, cnt in enumerate(obj_counts):
            block = all_texts[offsets[i]: offsets[i] + cnt]
            mask_list.append([True] * cnt + [False] * (max_m - cnt))
            block = block + [block[-1]] * (max_m - cnt)
            padded.extend(block)

        tokens = self.tokenizer(
            padded,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.Qformer.bert(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
                return_dict=True,
            )
        feats = out.last_hidden_state[:, 0, :]        # [B*M, H]
        feats = F.normalize(feats, dim=-1)
        feats = feats.view(B, max_m, -1)              # [B, M, H]
        node_mask = torch.tensor(mask_list, dtype=torch.bool, device=self.device)
        return {"node_features": feats, "node_mask": node_mask}

    # ── QFormer encoding paths ────────────────────────────────────────────────

    def _qformer_image_only(
        self,
        image_embeds: torch.Tensor,   # [B, N, H]  (already ln_vision applied)
        query_tokens: torch.Tensor,   # [1, Q, H]
    ) -> torch.Tensor:                # [B, Q, H]
        bs = image_embeds.shape[0]
        # Ensure consistent dtype: cast image_embeds to QFormer param dtype
        qformer_dtype = next(self.Qformer.parameters()).dtype
        image_embeds = image_embeds.to(qformer_dtype)
        q = query_tokens.expand(bs, -1, -1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        out = self.Qformer.bert(
            query_embeds=q,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return out.last_hidden_state   # [B, Q, H]

    def _qformer_fusion(
        self,
        image_embeds_ln: torch.Tensor,   # [B, 1+N, 1408]  ln_vision-normed ViT patches
        text_tokens: Any,
    ):
        """Single-pass image+text fusion: query tokens cross-attend to image patches
        AND self-attend to text tokens simultaneously.

        Returns:
          z_q   : [B, 256]  from query tokens via vision_proj
          q_tokens: [B, 32, H]
          text_cls: [B, H]  CLS from text stream (for text_proj anchor)
        """
        B = image_embeds_ln.shape[0]
        qformer_dtype = next(self.Qformer.parameters()).dtype
        image_embeds = image_embeds_ln.to(qformer_dtype)
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        q = self.query_tokens.expand(B, -1, -1)
        q_atts = torch.ones(q.shape[:2], dtype=torch.long, device=q.device)
        attn_mask = torch.cat([q_atts, text_tokens.attention_mask], dim=1)
        out = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=q,
            attention_mask=attn_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        all_tokens = out.last_hidden_state
        q_tokens = all_tokens[:, : self.num_query_token, :]                    # [B, 32, H]
        text_stream = all_tokens[:, self.num_query_token:, :]                   # [B, Lt, H]
        text_cls = text_stream[:, 0, :] if text_stream.shape[1] > 0 else q_tokens[:, 0, :]
        z_q = F.normalize(self.vision_proj(q_tokens).mean(dim=1), dim=-1)
        return z_q, q_tokens, text_cls

    # ── Visual branches (Fg, Fl, Ff) ─────────────────────────────────────────

    def _visual_branches(
        self,
        patch_tokens_ln: torch.Tensor,   # [B, N, 1408]  (original, ln_vision applied)
        obj_mask: torch.Tensor,          # [B, N]  accurate obj mask from ObjectLocalizer
        obj_node_out: Dict[str, torch.Tensor],  # kept for API compat (unused now)
        graphs: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        sel_tokens = self.visual_to_selector(patch_tokens_ln)  # [B, N, 512]
        grid_size = int(patch_tokens_ln.shape[1] ** 0.5)
        w = obj_mask.to(sel_tokens.dtype).unsqueeze(-1)        # [B, N, 1]

        # Fg: global mean-pool (whole scene)
        Fg_raw = sel_tokens.mean(dim=1)                        # [B, 512]

        # Fl: object-region aggregate, weighted by the accurate M_obj
        Fl_raw = (sel_tokens * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)  # [B, 512]

        # Ff: object-region frequency, M_obj as spatial weight
        Ff_raw = self.fft_branch(sel_tokens, obj_mask, (grid_size, grid_size))

        Fl_out = F.normalize(self.local_proj(Fl_raw), dim=-1)
        Ff_out = F.normalize(self.freq_proj(Ff_raw), dim=-1)
        if self.ablate_no_fl:
            Fl_out = torch.zeros_like(Fl_out)
        if self.ablate_no_ff:
            Ff_out = torch.zeros_like(Ff_out)
        return {
            "Fg": F.normalize(self.global_proj(Fg_raw), dim=-1),
            "Fl": Fl_out,
            "Ff": Ff_out,
        }

    # ── Query building (shared by forward & inference) ────────────────────────

    def _build_query(
        self,
        image_embeds_ln: torch.Tensor,   # [B, 1+N, 1408]  ln_vision applied
        captions: List[str],
        graphs: Sequence[Any],
        target_patches: Any = None,      # [B, N, 1408] ln_vision'd target patches (train only)
    ) -> Dict[str, torch.Tensor]:
        patch_tokens = image_embeds_ln[:, 1:, :]               # [B, N, 1408]

        # ── Path-U: upgradation (original image + text, single-pass fusion) ──
        text_tokens = self._tokenize(captions)
        Fq_u, q_tokens, text_cls = self._qformer_fusion(image_embeds_ln, text_tokens)

        # ── Pretrained anchors (SSN-style) ───────────────────────────────────
        # ref_embed: TRUE image-only Q-Former pass (no text leakage); text_embed:
        # text CLS → text_proj. The SSN blend interpolates between the two.
        F_img_only = self._qformer_image_only(image_embeds_ln, self.query_tokens)
        ref_embed = F.normalize(self.vision_proj(F_img_only).mean(dim=1), dim=-1)
        text_embed = F.normalize(self.text_proj(text_cls), dim=-1)

        raw_combined = torch.cat([text_embed, ref_embed], dim=-1)         # [B, 512]
        combined = F.relu(self.combiner_layer(raw_combined))              # [B, 1024]
        alpha = torch.clamp(self.alpha, min=0.0, max=self.alpha_max)

        out: Dict[str, torch.Tensor] = {
            "q_tokens": q_tokens, "Fq_u": Fq_u, "text_embed": text_embed,
        }

        if self.graph_enabled:
            # ── Encode graph nodes / fields ──────────────────────────────────
            obj_node_out = self._encode_obj_nodes(graphs)
            nm = obj_node_out["node_mask"].float().unsqueeze(-1)
            obj_feat = ((obj_node_out["node_features"] * nm).sum(1)
                        / nm.sum(1).clamp_min(1e-6))                       # [B, H]
            action_feat = self._encode_field(graphs, "action")            # [B, H]
            loc_feat = self._encode_field(graphs, "result_location")      # [B, H]  destination

            # ── Hard routing by action category ──────────────────────────────
            parsed = [self._parse_graph(g) for g in graphs]
            action_labels = [action_to_category(p.get("action")) for p in parsed]
            # Ablation A2: force every sample to a single category. Keeps the
            # transition branch active but removes the routing signal.
            if self.ablate_force_route:
                action_labels = [self.ablate_force_route] * len(action_labels)

            # ── Object localization via the frozen grounding-trained selector ──
            # Query with the OBJECT node ONLY, in the "graph node: {label}" format
            # the selector was trained on (full _encode_obj_nodes appends
            # unsupervised location pseudo-nodes that pollute the max-agg mask).
            loc_nodes = self._encode_object_for_localizer(graphs)
            with torch.no_grad():
                M_obj_ref = aggregate_mask(self.object_localizer(
                    patch_tokens.float(),
                    loc_nodes["node_features"].float(),
                    loc_nodes["node_mask"],
                ))                                                            # [B, N]
            # Plan-A: the object of an "Add" edit is NOT in the reference, so the
            # selector can only false-positive there → use a uniform (global)
            # mask for Add samples. Remove/Move keep the localized mask.
            is_add = torch.tensor([c == "Add" for c in action_labels],
                                  device=M_obj_ref.device)
            if is_add.any():
                M_obj_ref = M_obj_ref.clone()
                M_obj_ref[is_add] = 1.0

            router_out = self.action_router(
                action_labels, patch_tokens, M_obj_ref,
                obj_feat, loc_feat, action_feat,
            )
            F_trans = router_out["F_trans"]                                # [B, 256]

            # ── Multi-scale visual features (Fl/Ff use the selector's obj mask) ─
            vis = self._visual_branches(patch_tokens, M_obj_ref, obj_node_out, graphs)
            Fq_fused = self.token_fusion(Fq_u, vis["Fg"], vis["Fl"], vis["Ff"])

            # ── SSN-style linear addition + transition branch ────────────────
            scalar_input = torch.cat([text_embed, ref_embed, Fq_fused], dim=-1)  # [B, 768]
            dynamic_s = self.dynamic_scalar(scalar_input)                     # [B, 1]
            beta = torch.clamp(self.beta, min=0.0, max=self.beta_max) * self.aux_ramp
            gamma = self.gamma * self.aux_ramp                                 # γ ∈ (0, gamma_max)
            # Ablation A1: drop the F_trans branch entirely.
            if self.ablate_no_trans:
                gamma = gamma * 0.0
            # L2-normalize F_trans so it lives on the unit sphere like the
            # text/ref anchors; otherwise the outer normalize(final_query)
            # eats the gradient that should flow into raw_gamma (the trans_mlp
            # can otherwise change F_trans's magnitude freely, leaving gamma
            # fixed at its init value — observed in the force-beta run).
            F_trans_n = F.normalize(F_trans, dim=-1).to(Fq_fused.dtype)
            final_query = (
                alpha * self.combiner_output(combined)
                + beta * Fq_fused
                + gamma * F_trans_n
                + dynamic_s * text_embed
                + (1 - dynamic_s) * ref_embed
            )
        else:
            # Warmup: SSN-style combination without graph branches
            scalar_input = torch.cat(
                [text_embed, ref_embed, torch.zeros_like(ref_embed)], dim=-1)
            dynamic_s = self.dynamic_scalar(scalar_input)                     # [B, 1]
            final_query = (
                alpha * self.combiner_output(combined)
                + dynamic_s * text_embed
                + (1 - dynamic_s) * ref_embed
            )

        out["final_query"] = F.normalize(final_query, dim=-1)
        return out

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(
        self,
        image_embeds_ln: torch.Tensor,   # [B, 1+N, 1408]
        F_target_all: torch.Tensor,      # [Ntgt, 32, H]  precomputed gallery
        captions: List[str],
        graphs: Sequence[Any],
        graph_enabled: bool = True,
    ) -> torch.Tensor:                   # [B, Ntgt]
        prev_graph_enabled = self.graph_enabled
        self.graph_enabled = graph_enabled
        try:
            out = self._build_query(image_embeds_ln, captions, graphs)
            final_query = out["final_query"]   # [B, 256]
        finally:
            self.graph_enabled = prev_graph_enabled

        # Project gallery tokens → [Ntgt, 32, 256], max-sim over 32 tokens per gallery image
        z_tgt_tokens = F.normalize(self.vision_proj(F_target_all), dim=-1)  # [Ntgt, 32, 256]
        # final_query [B, 256] @ z_tgt_tokens [Ntgt, 32, 256] → [B, Ntgt, 32] → max → [B, Ntgt]
        sim = torch.einsum('bd,ntd->bnt', final_query, z_tgt_tokens).max(dim=-1).values
        return sim

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        samples: Dict[str, Any],
        warmup: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        samples keys:
            "image"      : precomputed ViT features of reference [B, 1+N, 1408]
            "target"     : precomputed ViT features of target    [B, 1+N, 1408]
            "text_input" : list of caption strings
            "graph"      : list of graph dicts
            "hard"       : (optional) [B, K, 1+N, 1408] same-group hard negatives
            "hard_mask"  : (optional) [B, K] bool, True for valid hard entries
        warmup: if True, skip graph branches (QFormer fusion only)
        """
        self.graph_enabled = not warmup

        image_embeds = samples["image"].to(self.device)
        target_embeds = samples["target"].to(self.device)
        captions = samples["text_input"]
        graphs = samples["graph"]
        hard_embeds = samples.get("hard")
        hard_mask = samples.get("hard_mask")

        with self.maybe_autocast():
            image_embeds_ln = self.ln_vision(image_embeds)
            target_embeds_ln = self.ln_vision(target_embeds)
            if hard_embeds is not None:
                hard_embeds = hard_embeds.to(self.device)
                B, K = hard_embeds.shape[:2]
                hard_flat = hard_embeds.view(B * K, *hard_embeds.shape[2:])
                hard_embeds_ln = self.ln_vision(hard_flat)
            else:
                hard_embeds_ln = None

        # ── Target encoding ───────────────────────────────────────────────────
        F_target = self._qformer_image_only(target_embeds_ln, self.query_tokens)  # [B, 32, H]
        # Project each of the 32 tokens → [B, 32, 256], normalize per-token
        z_tgt_tokens = F.normalize(self.vision_proj(F_target), dim=-1)   # [B, 32, 256]

        # ── Query encoding (pass target patches so transitions can score tgt) ──
        target_patches = target_embeds_ln[:, 1:, :]
        query_out = self._build_query(image_embeds_ln, captions, graphs, target_patches)
        final_query = query_out["final_query"]   # [B, 256]
        q_tokens = query_out["q_tokens"]         # [B, 32, H]
        Fq_u = query_out["Fq_u"]                 # [B, 256]  fusion (image+text) query
        text_embed = query_out["text_embed"]     # [B, 256]  pure-text anchor

        # ── Main InfoNCE loss: symmetric, max-sim over 32 target tokens ────────
        # final_query [B, 256] @ z_tgt_tokens [B, 32, 256] → [B, B, 32] → max → [B, B]
        self.temp.data = torch.clamp(self.temp.data, min=1e-3)
        scale = 1.0 / self.temp

        sim_tokens = torch.einsum('bd,ntd->bnt', final_query, z_tgt_tokens)  # [B, B, 32]
        z_tgt_mean = sim_tokens.max(dim=-1).values                            # [B, B]

        # ── Main InfoNCE: pure in-batch (matches v9, no hard-neg interference) ──
        pos_sim = scale * z_tgt_mean
        gt = torch.arange(pos_sim.shape[0], device=pos_sim.device)
        ls = self.label_smoothing
        l_pos = (F.cross_entropy(pos_sim, gt, label_smoothing=ls)
                 + F.cross_entropy(pos_sim.T, gt, label_smoothing=ls)) / 2

        # ── Group-aware hard negatives (independent auxiliary loss) ────────────
        # Same-group candidates are highly similar to ref/tgt. At scale=100 they
        # would overpower the in-batch InfoNCE if added to its logits. Instead
        # we compute a SEPARATE small-weight CE over [positive_target, K hard]
        # with a lower fixed scale so the gradient doesn't dominate.
        if hard_embeds_ln is not None:
            B = final_query.shape[0]
            K = hard_embeds.shape[1]
            with torch.no_grad():
                F_hard = self._qformer_image_only(hard_embeds_ln, self.query_tokens)  # [B*K, 32, H]
                z_hard = F.normalize(self.vision_proj(F_hard), dim=-1)                  # [B*K, 32, 256]
            z_hard = z_hard.view(B, K, z_hard.shape[1], z_hard.shape[2]).detach()
            sim_hard = torch.einsum('bd,bktd->bkt', final_query, z_hard).max(dim=-1).values  # [B, K]
            sim_hard = sim_hard.masked_fill(~hard_mask, -1e4)
            # Positive: each anchor's own target similarity (diagonal of z_tgt_mean)
            pos_sim_diag = z_tgt_mean.diag().unsqueeze(1)                       # [B, 1]
            group_logits = torch.cat([pos_sim_diag, sim_hard], dim=1) * self.group_scale
            group_gt = torch.zeros(B, dtype=torch.long, device=final_query.device)
            l_group = F.cross_entropy(group_logits, group_gt)
        else:
            l_group = torch.zeros((), device=self.device)

        # ── Auxiliary losses (graph phase only) ───────────────────────────────
        if self.graph_enabled:
            # Aux head: q_tokens cross-attended with target tokens → single vector
            q_guided = self.aux_head(q_tokens, F_target.detach())           # [B, 256]
            sim_aux_tok = torch.einsum('bd,ntd->bnt', q_guided, z_tgt_tokens)
            sim_aux = scale * sim_aux_tok.max(dim=-1).values                # [B, B]
            l_aux = (F.cross_entropy(sim_aux, gt) + F.cross_entropy(sim_aux.T, gt)) / 2

            total = (l_pos + self.lambda_aux * l_aux
                     + self.lambda_group * l_group)
        else:
            l_aux = torch.zeros((), device=self.device)
            total = l_pos + self.lambda_group * l_group

        # ── ComqueryFormer-style multi-scale global contrastive (optional) ────
        # Direct InfoNCE on Fq_u against target gives the BLIP2 fusion head an
        # extra supervision signal that is NOT diluted by combiner/token_fusion.
        # lambda_global=0.0 disables it (v12 behavior).
        if self.lambda_global > 0.0:
            sim_global_tok = torch.einsum('bd,ntd->bnt', Fq_u, z_tgt_tokens)  # [B, B, 32]
            sim_global = scale * sim_global_tok.max(dim=-1).values            # [B, B]
            gt_global = torch.arange(sim_global.shape[0], device=sim_global.device)
            l_global = (F.cross_entropy(sim_global, gt_global, label_smoothing=ls)
                        + F.cross_entropy(sim_global.T, gt_global, label_smoothing=ls)) / 2
            total = total + self.lambda_global * l_global
        else:
            l_global = torch.zeros((), device=self.device)

        # ── ColBERT-style multi-vector matching (KEY: fixes query/target asymmetry) ──
        # Target side keeps 32 tokens (max-sim). But final_query is a SINGLE
        # vector → query loses all multi-token granularity. We add a parallel
        # supervision where the 32 query tokens (token-wise projected to 256)
        # match the 32 target tokens via ColBERT sum-of-max similarity.
        # This gives the query side capacity that matches the target side.
        if self.lambda_colbert > 0.0:
            # Token-wise project q_tokens [B, 32, H] → [B, 32, 256]
            q_tok_proj = F.normalize(self.vision_proj(q_tokens), dim=-1)  # [B, 32, 256]
            # ColBERT sum-of-max: for each query token, max similarity to any
            # target token; sum over query tokens → similarity scalar per pair.
            # einsum: q[b, i, d] · t[n, j, d] → s[b, n, i, j]
            sim_qt = torch.einsum('bid,njd->bnij', q_tok_proj, z_tgt_tokens)
            # max over target tokens (dim=3), then mean over query tokens
            colbert_sim = sim_qt.max(dim=-1).values.mean(dim=-1)            # [B, B]
            colbert_logits = scale * colbert_sim
            gt_cb = torch.arange(colbert_logits.shape[0], device=colbert_logits.device)
            l_colbert = (F.cross_entropy(colbert_logits, gt_cb, label_smoothing=ls)
                         + F.cross_entropy(colbert_logits.T, gt_cb, label_smoothing=ls)) / 2
            total = total + self.lambda_colbert * l_colbert
        else:
            l_colbert = torch.zeros((), device=self.device)

        # ── SPRC-style alignment loss (optional regularizer) ──────────────────
        # MSE pulls the image+text fusion query toward the pure-text anchor in
        # the shared 256-d ITC space. This regularizes the query to ignore
        # image content irrelevant to the caption — exactly the property CIRR
        # group recall stresses. lambda_align=0.0 disables it (v11 behavior).
        if self.lambda_align > 0.0:
            l_align = F.mse_loss(Fq_u, text_embed.detach())
            total = total + self.lambda_align * l_align
        else:
            l_align = torch.zeros((), device=self.device)

        return {
            "l_main":   l_pos,
            "l_aux":    l_aux,
            "l_group":  l_group,
            "l_align":  l_align,
            "l_global": l_global,
            "l_colbert": l_colbert,
            "loss":     total,
        }
