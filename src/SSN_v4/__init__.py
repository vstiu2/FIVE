"""SSN-BLIP2-v4: action-only mask + semantic injection.

Differences from v3
-------------------
1. **Main path stays raw.** Q-Former (image+text fusion → Fq_u, image-only
   → ref_embed, text_proj → text_embed) sees the ORIGINAL patches. M_act
   no longer pollutes Fq_u/ref_embed/text_embed → ablation no longer
   triggers OOD inputs at test time.

2. **M_act applied only to action-side branches.** ActionRouter (F_trans),
   Fl, Ff, Fg are now computed on ``patches_final``:

       patches_final = patches × M_act + α_inject · M_loc · obj_proto

   The first term **gates** what to preserve from the reference; the
   second **injects** what to introduce from the caption (e.g. the "flag"
   in "Add a flag on the road"). Different action types correspond to
   different sparsity patterns of (M_act, M_loc, α_inject).

3. **SemanticInjector** is a new module. ``M_loc`` reuses the frozen
   ObjectLocalizer querying the ``result_location`` node; ``obj_proto`` is
   produced by an MLP from the BERT embedding of the ``object`` field,
   projecting into ViT space.

Ablation flags
--------------
   --ablate_no_action_mask        force M_act = 1
   --ablate_no_semantic_inject    force α_inject = 0

Nothing under src/models/, src/model/, src/SSN_v3/ is modified.
"""
