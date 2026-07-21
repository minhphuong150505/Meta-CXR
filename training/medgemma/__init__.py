"""MedGemma Stage-2 components.

Modules here must not import LAVIS, the META-CXR vision encoders, the Q-Former
or MHCAC. The Q-Former *interface* (a projected ``[B, N, 768]`` tensor handed to
``SoftTokenProjector``) is all the hybrid ablation needs; producing that tensor
is Stage-1's job and lives in ``training/stage1/``.
"""
