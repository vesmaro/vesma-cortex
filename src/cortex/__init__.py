"""cortex — development library for the mnema-cortex decision model.

Dual-epoch delivery (ADR 0001, owner addendum P2):

- THIS repo is the train epoch: corpus generation, pair features, D/N
  candidate training, internal CV selection, ONNX export, single-shot
  evaluation (preregistration v2, frozen).
- The runtime epoch is a single self-contained ONNX artifact
  (``mnema-cortex-v1``) bundled into the engine by the NanoProvider
  pattern — the engine never imports this package.

Contracts: docs/specs/inference-v1.md (artifact), docs/specs/data-contract.md
(data). Slice A3a ships contracts and skeletons only; algorithm bodies land
in A3b+ (every stub raises NotImplementedError by design — silent partial
implementations would violate the honest-skeleton rule).
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Re-exported so callers can pin the artifact identity from one place
#: (single-source rule: defined in cortex.artifacts, A3a).
from cortex.artifacts import ARTIFACT_NAME

__all__ = ["ARTIFACT_NAME", "__version__"]
