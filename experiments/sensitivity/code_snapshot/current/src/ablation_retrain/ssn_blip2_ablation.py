"""SSN-BLIP2 with an extra ablation switch for the global mean-pool branch (Fg).

The baseline model in ``src/models/ssn_blip2.py`` already supports
``ablate_no_fl`` / ``ablate_no_ff`` / ``ablate_no_trans``. To run the
Fl / Fg / Ff retrain ablation we need a matching ``ablate_no_fg`` switch.

We subclass the baseline ``Model`` so behavior is bit-identical when all
ablation flags are off (no risk of regressing the baseline).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.ssn_blip2 import Model as BaselineModel  # noqa: E402


class Model(BaselineModel):
    """Adds ``ablate_no_fg`` (zero out the global mean-pool branch Fg)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default OFF → identical to baseline when nothing is configured.
        self.ablate_no_fg: bool = False

    # Override to accept the additional ``no_fg`` kwarg.
    def set_ablation(
        self,
        no_trans: bool = False,
        no_fl: bool = False,
        no_fg: bool = False,
        no_ff: bool = False,
        force_route: str = "",
    ) -> None:
        super().set_ablation(
            no_trans=no_trans,
            no_fl=no_fl,
            no_ff=no_ff,
            force_route=force_route,
        )
        self.ablate_no_fg = bool(no_fg)

    # The only reason to override ``_visual_branches`` is to apply the Fg
    # zeroing. Everything else (Fl/Ff branches, normalization, ablation
    # flags inherited from the parent) is reused verbatim.
    def _visual_branches(
        self,
        patch_tokens_ln: torch.Tensor,
        obj_mask: torch.Tensor,
        obj_node_out: Dict[str, torch.Tensor],
        graphs: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        vis = super()._visual_branches(patch_tokens_ln, obj_mask, obj_node_out, graphs)
        if self.ablate_no_fg:
            vis["Fg"] = torch.zeros_like(vis["Fg"])
        return vis
