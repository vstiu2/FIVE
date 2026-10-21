"""Ablation-retrain package for SSN-BLIP2.

Contains a thin subclass of the baseline ``Model`` that adds an
``ablate_no_fg`` switch (alongside the existing ``ablate_no_fl`` /
``ablate_no_ff`` / ``ablate_no_trans`` switches), plus a training entry
point that wires the new flag through the CLI.

Nothing here modifies the original ``src/models/ssn_blip2.py`` or
``src/train_cirr_blip2.py``.
"""
