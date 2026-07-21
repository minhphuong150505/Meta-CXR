"""Findings-first inference over externally fine-tuned MedGemma checkpoints.

This package runs inference only. It constructs no optimizer, computes no
gradients, and never calls ``model.train()``. Stage-1 META-CXR/MHCAC training
is a separate, still-supported concern and lives under ``pretraining/``.
"""
