"""Vector Stack deterministic topological feature construction.

This package provides deterministic, multi-block topological liftings
from persistence diagrams (H0,H1) into a concatenated feature vector.

Blocks implemented (v1):
 - Persistence Landscapes (multi-resolution)
 - Persistence Images (multi-scale grids, intensity normalized)
 - Betti Curves
 - Sliced Wasserstein Projections (deterministic angles)
 - Static Multi-Scale Kernel Dictionary Responses (k-means++ trained on train diagrams only)

Silhouettes intentionally deferred until core evaluation completes.

All functions operate strictly on raw persistence diagrams (birth, death)
and never compute forbidden statistical proxy aggregates.

Normalization artifacts and dictionary centers must be fit exclusively
on training data respecting temporal split integrity (max train ts < min test ts).
"""

from .vector_stack import (
    VectorStackConfig,
    build_vector_stack,
    prepare_kernel_dictionaries,
    compute_block_features,
    stack_and_normalize,
)

__all__ = [
    'VectorStackConfig',
    'build_vector_stack',
    'prepare_kernel_dictionaries',
    'compute_block_features',
    'stack_and_normalize',
]
