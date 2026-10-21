"""Test-only: evaluate the model with ``final_query = normalize(s·text + (1-s)·ref)``.

Why this script exists
----------------------
We want to see how the model performs when ONLY the SSN-style blend branch
contributes to ``final_query`` — i.e. drop the combiner, drop β·Fq_fused,
drop γ·F_trans, and keep just the learned linear interpolation between
the pure-text anchor and the image-only anchor.

What it does
------------
* Loads the existing checkpoint as-is (no retraining).
* Monkey-patches ``Model._build_query`` so it returns the same dict but with
  ``final_query`` replaced by ``normalize(s·text_embed + (1-s)·ref_embed)``,
  where ``s = dynamic_scalar([text_embed, ref_embed, Fq_fused])`` — exactly
  the same scalar the original model computes.
* Delegates the rest (dataset, gallery encoding, metrics) to
  ``src/test_cirr_blip2.py`` verbatim.

Note: ``dynamic_scalar`` reads Fq_fused as a feature, not as an additive
term, so we still need to build Fq_fused in the graph-enabled path. If you
want a Fq_fused-free variant, pass ``--ablate_no_fl --ablate_no_ff``; the
attention block degenerates to a single global condition token.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python src/ablation_retrain/test_only_blend.py \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ssn_blip2_selector_v2/best_arithmetic.pth \\
        --dataset CustomCIRR --split test \\
        --feature_dir features/CustomCIRR --gpu 0

Results land beside the checkpoint as
``test_results_<split>_only_blend.json`` / ``test_<split>_only_blend.log``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import models.ssn_blip2 as ssn_blip2_module  # noqa: E402
import test_cirr_blip2 as tcb_test           # noqa: E402
from models.ssn_blip2 import Model as BaselineModel  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Patched _build_query: identical signature/dict to the baseline, but
# final_query is replaced by normalize(s·text_embed + (1-s)·ref_embed).
# ─────────────────────────────────────────────────────────────────────────────

def _patched_build_query(
    self,
    image_embeds_ln: torch.Tensor,
    captions: List[str],
    graphs: Sequence[Any],
    target_patches: Any = None,
) -> Dict[str, torch.Tensor]:
    patch_tokens = image_embeds_ln[:, 1:, :]
    text_tokens = self._tokenize(captions)
    Fq_u, q_tokens, text_cls = self._qformer_fusion(image_embeds_ln, text_tokens)

    # Pretrained anchors (same definitions as the baseline _build_query).
    F_img_only = self._qformer_image_only(image_embeds_ln, self.query_tokens)
    ref_embed = F.normalize(self.vision_proj(F_img_only).mean(dim=1), dim=-1)
    text_embed = F.normalize(self.text_proj(text_cls), dim=-1)

    out: Dict[str, torch.Tensor] = {
        "q_tokens": q_tokens, "Fq_u": Fq_u, "text_embed": text_embed,
    }

    if self.graph_enabled:
        # We still need Fq_fused to feed dynamic_scalar, because that scalar
        # was learned conditioned on the visual context vector. The visual
        # branches (Fg/Fl/Ff) honor ablation flags.
        from model.action_transitions import action_to_category
        from model.object_localizer import aggregate_mask

        obj_node_out = self._encode_obj_nodes(graphs)
        loc_nodes = self._encode_object_for_localizer(graphs)

        parsed = [self._parse_graph(g) for g in graphs]
        action_labels = [action_to_category(p.get("action")) for p in parsed]
        if self.ablate_force_route:
            action_labels = [self.ablate_force_route] * len(action_labels)

        with torch.no_grad():
            M_obj_ref = aggregate_mask(self.object_localizer(
                patch_tokens.float(),
                loc_nodes["node_features"].float(),
                loc_nodes["node_mask"],
            ))
        is_add = torch.tensor([c == "Add" for c in action_labels],
                              device=M_obj_ref.device)
        if is_add.any():
            M_obj_ref = M_obj_ref.clone()
            M_obj_ref[is_add] = 1.0

        vis = self._visual_branches(patch_tokens, M_obj_ref, obj_node_out, graphs)
        Fq_fused = self.token_fusion(Fq_u, vis["Fg"], vis["Fl"], vis["Ff"])
        scalar_input = torch.cat([text_embed, ref_embed, Fq_fused], dim=-1)
        dynamic_s = self.dynamic_scalar(scalar_input)
    else:
        scalar_input = torch.cat(
            [text_embed, ref_embed, torch.zeros_like(ref_embed)], dim=-1)
        dynamic_s = self.dynamic_scalar(scalar_input)

    final_query = dynamic_s * text_embed + (1.0 - dynamic_s) * ref_embed
    out["final_query"] = F.normalize(final_query, dim=-1)
    return out


def _patch_model_class() -> None:
    ssn_blip2_module.Model._build_query = _patched_build_query  # type: ignore[assignment]
    BaselineModel._build_query = _patched_build_query           # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Run baseline main(), then rename outputs to add an _only_blend suffix so
# existing test results are not overwritten.
# ─────────────────────────────────────────────────────────────────────────────

def _run_main_with_suffix() -> None:
    from pathlib import Path
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--split', type=str, default='val')
    known, _ = p.parse_known_args()
    log_dir = Path(known.model_path).parent
    split = known.split

    log_src = log_dir / f'test_{split}.log'
    json_src = log_dir / f'test_results_{split}.json'
    log_dst = log_dir / f'test_{split}_only_blend.log'
    json_dst = log_dir / f'test_results_{split}_only_blend.json'

    for f in (log_dst, json_dst):
        if f.exists():
            f.unlink()

    try:
        tcb_test.main()
    finally:
        if log_src.exists():
            log_src.replace(log_dst)
        if json_src.exists():
            json_src.replace(json_dst)
        print(f"\n→ Outputs renamed to:\n   {log_dst}\n   {json_dst}\n")


if __name__ == '__main__':
    _patch_model_class()
    _run_main_with_suffix()
