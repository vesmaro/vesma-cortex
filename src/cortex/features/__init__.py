"""Features package — the frozen pair-feature contract lives in pair.py.

The B2 graph block (ADR 0002) is GATED: :data:`FEATURE_NAMES` stays the
untouched default; graph evidence enters only via
:func:`cortex.features.graph.attach_graph_features`.
"""

from cortex.features.graph import (
    GRAPH_FEATURES,
    GraphEvidence,
    GraphSidecar,
    GraphSidecarError,
    attach_graph_features,
)
from cortex.features.pair import (
    FEATURE_NAMES,
    FIELD_COSINE_FEATURES,
    FeatureVector,
    PairRecord,
    features,
)

__all__ = [
    "FEATURE_NAMES",
    "FIELD_COSINE_FEATURES",
    "GRAPH_FEATURES",
    "FeatureVector",
    "PairRecord",
    "GraphEvidence",
    "GraphSidecar",
    "GraphSidecarError",
    "attach_graph_features",
    "features",
]
