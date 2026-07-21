"""Stage-1 (BLIP-2 / Q-Former / MHCAC) access for Stage-2 code.

This package is deliberately empty at import time. Every symbol that pulls in
LAVIS, the vision encoders or MHCAC lives in ``lavis_loader``, so that
``import training.stage1`` costs nothing and a ``medgemma_direct`` run never
touches the Stage-1 stack. Import ``training.stage1.lavis_loader`` explicitly,
and only from a code path that has already established it needs Stage-1.
"""
