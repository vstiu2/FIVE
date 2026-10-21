"""Test-only: evaluate the model with ``final_query = normalize(Fq_fused)``.

Why this script exists
----------------------
We want to see how the model performs when ONLY the ``β·Fq_fused`` branch
contributes to ``final_query``. Because the original ``_build_query`` ends
with ``F.normalize(final_query, dim=-1)``, the scalar β is washed out, so
"only β·Fq_fused" is mathematically identical to "final_query = normalize(Fq_fused)".

What it does
------------
* Loads the existing checkpoint as-is (no retraining).
* Monkey-patches ``Model._build_query`` so it returns the same dict but with
  ``final_query`` replaced by ``normalize(Fq_fused)``.
* Delegates the rest (dataset, gallery encoding, metrics) to
  ``src/test_cirr_blip2.py`` verbatim.

Combine freely with the ablation switches (``--ablate_no_fl`` etc.) — those
still affect what Fq_fused sees through the visual branches.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python src/ablation_retrain/test_only_fq_fused.py \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ssn_blip2_selector_v2/best_arithmetic.pth \\
        --dataset CustomCIRR --split test \\
        --feature_dir features/CustomCIRR --gpu 0

    # Pair with retrained ablation ckpts:
    CUDA_VISIBLE_DEVICES=0 python src/ablation_retrain/test_only_fq_fused.py \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ablation_retrain/no_fl_ff/best_arithmetic.pth \\
        --dataset CustomCIRR --split test \\
        --feature_dir features/CustomCIRR --gpu 0 \\
        --ablate_no_fl --ablate_no_ff

Results land beside the checkpoint as ``test_results_<split>_only_fq_fused.json``
and ``test_<split>_only_fq_fused.log``. (To avoid overwriting the regular
test outputs, ``--split`` is internally suffixed before main() runs.)
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
# Patched _build_query: identical to the baseline, but final_query is
# replaced by normalize(Fq_fused). Falls back to normalize(Fq_u) when graph
# branches are disabled (Fq_fused isn't built in that path).
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

    # Keep these around so the rest of the pipeline (e.g. inference()) gets
    # the same dict shape as the baseline.
    F_img_only = self._qformer_image_only(image_embeds_ln, self.query_tokens)
    ref_embed = F.normalize(self.vision_proj(F_img_only).mean(dim=1), dim=-1)
    text_embed = F.normalize(self.text_proj(text_cls), dim=-1)

    out: Dict[str, torch.Tensor] = {
        "q_tokens": q_tokens, "Fq_u": Fq_u, "text_embed": text_embed,
    }

    if self.graph_enabled:
        # Re-run the graph branches so vis["Fg/Fl/Ff"] respect the ablation
        # flags. ObjectLocalizer / ActionRouter are needed to produce M_obj
        # for Fl/Ff (Fl/Ff may then be zeroed out by their own flags).
        obj_node_out = self._encode_obj_nodes(graphs)
        nm = obj_node_out["node_mask"].float().unsqueeze(-1)
        obj_feat = ((obj_node_out["node_features"] * nm).sum(1)
                    / nm.sum(1).clamp_min(1e-6))
        action_feat = self._encode_field(graphs, "action")
        loc_feat = self._encode_field(graphs, "result_location")

        from model.action_transitions import action_to_category
        from model.object_localizer import aggregate_mask

        parsed = [self._parse_graph(g) for g in graphs]
        action_labels = [action_to_category(p.get("action")) for p in parsed]
        if self.ablate_force_route:
            action_labels = [self.ablate_force_route] * len(action_labels)

        loc_nodes = self._encode_object_for_localizer(graphs)
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

        # We don't actually use F_trans here, but the router has side-effect-
        # free outputs so calling it is fine. Skip to save time.
        vis = self._visual_branches(patch_tokens, M_obj_ref, obj_node_out, graphs)
        Fq_fused = self.token_fusion(Fq_u, vis["Fg"], vis["Fl"], vis["Ff"])
        final_query = Fq_fused
    else:
        # No graph branches → Fq_fused isn't defined. Use Fq_u as the closest
        # analog (this is the BLIP-2 fusion query before any visual injection).
        final_query = Fq_u

    out["final_query"] = F.normalize(final_query, dim=-1)
    return out


def _patch_model_class() -> None:
    ssn_blip2_module.Model._build_query = _patched_build_query  # type: ignore[assignment]
    BaselineModel._build_query = _patched_build_query           # type: ignore[assignment]


def _suffix_split_in_argv() -> None:
    """Trick the baseline ``main()`` into writing logs/json to a separate
    filename so we don't overwrite the regular test results. We do this by
    monkey-patching ``argparse.ArgumentParser.parse_args`` to append a
    suffix to ``args.split`` AFTER parsing — but only for log/result paths.

    Simpler: just wrap ``tcb_test.main`` and swap ``args.split`` post-parse.
    But ``main`` reads split before saving — we want the dataset to still
    load with the real split. So we patch the result filename instead.

    The cleanest implementation: leave args.split alone; instead intercept
    Path operations by overriding ``main`` itself.
    """
    # Actually, simplest: rename outputs via a small wrapper around main().
    pass  # implemented via _run_main below.


def _run_main_with_suffix() -> None:
    """Run baseline main() then rename the two output files to add the
    ``_only_fq_fused`` suffix, so regular test results are not overwritten.
    """
    from pathlib import Path
    # Parse a copy of argv to figure out where outputs will land.
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--split', type=str, default='val')
    known, _ = p.parse_known_args()
    log_dir = Path(known.model_path).parent
    split = known.split

    log_src = log_dir / f'test_{split}.log'
    json_src = log_dir / f'test_results_{split}.json'
    log_dst = log_dir / f'test_{split}_only_fq_fused.log'
    json_dst = log_dir / f'test_results_{split}_only_fq_fused.json'

    # Pre-delete dst so the run starts clean.
    for f in (log_dst, json_dst):
        if f.exists():
            f.unlink()

    try:
        tcb_test.main()
    finally:
        # Rename whichever the baseline produced into the suffixed name.
        if log_src.exists():
            log_src.replace(log_dst)
        if json_src.exists():
            json_src.replace(json_dst)
        print(f"\n→ Outputs renamed to:\n   {log_dst}\n   {json_dst}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _patch_model_class()
    _run_main_with_suffix()
