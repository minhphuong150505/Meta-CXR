"""Training-loop infrastructure: run state, RNG capture and checkpoint I/O.

Deliberately knows nothing about MIMIC-CXR, report records, MedGemma, Figure 9
or GCS. It depends on torch and the standard library only, which is what makes
resume correctness testable on CPU with a toy model.
"""
