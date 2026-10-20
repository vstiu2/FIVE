"""SSN-BLIP2-v3: action-conditioned token mask + push-pull KL.

What this package adds on top of SSN_v2 (``src/models/ssn_blip2.py``)
-----------------------------------------------------------------------

1. ``ActionMaskHead`` produces a per-patch soft mask ``M_act ∈ [0, 1]^N``
   conditioned on the action embedding and the frozen ObjectLocalizer prior
   ``M_obj``. The mask is applied at the **Q-Former input** (multiplying
   patch tokens), so a single mask propagates into Fq_u / ref_embed / Fl /
   Ff / Fg simultaneously — unlike the v2 ``F_trans`` term which was a tiny
   additive contribution at the very end of ``final_query``.

2. ``Model`` (subclass of v2) runs ``_build_query`` twice per training step:
   once with ``M_act`` (positive query) and once with ``1 − M_act`` (negative
   query). A push-pull KL objective inspired by SSN forces the action mask
   to be discriminative: positive query ⇒ on-diagonal, negative query ⇒
   off-diagonal from positive's distribution.

3. The existing F_trans branch is kept for inference compatibility but its
   gradient is dwarfed by the masked-query path; ablation flag
   ``--ablate_no_action_mask`` collapses ``M_act`` back to all-ones, which
   makes the model bit-identical to v2 in inference.

Nothing under ``src/models/`` or ``src/model/`` is modified.
"""
