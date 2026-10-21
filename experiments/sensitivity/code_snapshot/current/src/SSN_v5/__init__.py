"""SSN-BLIP2-v5: v4 + box constraint on mean(M_act).

What's different from v4
------------------------
The only architectural change is the mask-mean regularizer in the push-pull
objective. v4 used a quadratic pull toward 0.5:

    l_mask_target = (mean(M_act) - 0.5)²        # v4

In run1 this proved too weak — mean(M_act) collapsed from 0.59 → 0.27 after
epoch 16, dragging best_arithmetic down. v5 replaces it with a one-sided
box constraint:

    l_mask_target = ReLU(lo - mean)² + ReLU(mean - hi)²   # v5

with lo=0.4, hi=0.7. Inside the box there is no penalty (mask is free to
learn); outside, a quadratic penalty pulls it back. The weight is also
raised from 0.1 → 0.5 so the constraint is strong enough to hold the mean
inside the box during late-stage training.

Everything else (ActionMaskHead, SemanticInjector, counterfactual,
blend_dropout, ablation flags) is inherited from v4 unchanged.

v4 source files are NOT modified.
"""
