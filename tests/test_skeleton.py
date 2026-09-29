"""Skeleton contract tests (A3a).

Pins the slice's own deliverables:

1. every module imports and the contracted signatures exist;
2. ARTIFACT_NAME is defined in exactly ONE place (ADR 0001 П1);
3. FEATURE_NAMES is frozen — order and composition are contract;
4. grid caps (≤ 8 per ADR 0001 V1);
5. zero network imports across src/cortex — the repo-side twin of the
   engine's tests/test_mcp_core_isolation.py AST tripwire;
6. CLI subcommands exist and fail loud (exit 2), never fake success;
7. algorithm stubs raise NotImplementedError (honest skeleton — no
   silent partial implementations).
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "cortex"

# ── 1. Modules import and signatures exist ────────────────────────────────────


def test_package_imports() -> None:
    import cortex
    import cortex.artifacts
    import cortex.candidates
    import cortex.candidates.d_boost
    import cortex.candidates.n_head
    import cortex.cli.main
    import cortex.data.fingerprints
    import cortex.data.holdout
    import cortex.eval.runner
    import cortex.features.pair
    import cortex.pretrain.corruption
    import cortex.select.cv

    assert cortex.ARTIFACT_NAME == cortex.artifacts.ARTIFACT_NAME == "mnema-cortex-v1"


def test_contract_signatures() -> None:
    from cortex.cli.main import main
    from cortex.data.holdout import assert_labels_isolated, assert_no_pair_overlap, split_holdout
    from cortex.eval.runner import run_baseline, run_single_shot
    from cortex.features.pair import features
    from cortex.pretrain.corruption import generate_pretrain_pairs
    from cortex.select.cv import run_cv, select_candidate

    for fn in (
        features,
        split_holdout,
        assert_no_pair_overlap,
        assert_labels_isolated,
        run_cv,
        select_candidate,
        run_single_shot,
        run_baseline,
        generate_pretrain_pairs,
        main,
    ):
        assert callable(fn), f"contract function missing: {fn}"

    sig = inspect.signature(features)
    assert list(sig.parameters) == ["record_a", "record_b", "similarity", "vec_a", "vec_b", "field_cosines"]
    assert sig.parameters["field_cosines"].kind is inspect.Parameter.KEYWORD_ONLY


def test_candidate_surface() -> None:
    from cortex.candidates.d_boost import DBoostModel
    from cortex.candidates.n_head import NHeadModel

    for cls in (DBoostModel, NHeadModel):
        for method in ("train", "predict_proba", "export_onnx"):
            assert hasattr(cls, method), f"{cls.__name__}.{method} missing"


def test_stubs_raise_not_implemented() -> None:
    from cortex.artifacts import build_metadata_props, sha256_file
    from cortex.data.holdout import split_holdout
    from cortex.features.pair import features

    with pytest.raises(NotImplementedError):
        features(None, None, 0.5)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        sha256_file(Path("."))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        build_metadata_props(embedder_pin="", corpus_fingerprint="", trained_at="", candidate="x", feature_names=())
    with pytest.raises(NotImplementedError):
        split_holdout([])


# ── 2. ARTIFACT_NAME single source ────────────────────────────────────────────


def _artifact_name_assignments() -> list[str]:
    """AST scan (engine test_mcp_core_isolation pattern): files that ASSIGN
    the ARTIFACT_NAME constant (plain Assign or annotated AnnAssign — the
    repo idiom is ``NAME: Final[str] = ...``). Re-exports via import are
    allowed."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "ARTIFACT_NAME":
                    offenders.append(rel)
    return offenders


def test_artifact_name_defined_once() -> None:
    assignments = _artifact_name_assignments()
    assert assignments == ["artifacts/__init__.py"], (
        f"ARTIFACT_NAME must live in exactly one place (cortex/artifacts/__init__.py): {assignments}"
    )


# ── 3. Frozen feature contract ────────────────────────────────────────────────


def test_feature_names_frozen() -> None:
    from cortex.features.pair import FEATURE_NAMES, FIELD_COSINE_FEATURES

    assert FEATURE_NAMES == (
        "cos_target",
        "char3_jaccard",
        "char4_jaccard",
        "char5_jaccard",
        "char3_containment",
        "char4_containment",
        "char5_containment",
        "tag_jaccard",
        "title_len_delta",
        "body_len_delta",
        "tag_count_delta",
        "type_match",
        "lang_match",
    )
    assert FIELD_COSINE_FEATURES == ("cos_title", "cos_body", "cos_tags")
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES), "feature names must be unique"
    assert len(set(FIELD_COSINE_FEATURES) & set(FEATURE_NAMES)) == 0


def test_frozen_protocol_constants() -> None:
    from cortex.data.fingerprints import CANONICAL_JSON_KWARGS
    from cortex.data.holdout import HOLDOUT_FRACTION
    from cortex.eval.runner import BASELINE_THRESHOLD, DUPLICATE_THRESHOLD_PROBABILITY
    from cortex.select.cv import CV_FOLDS, CV_SEEDS

    assert CANONICAL_JSON_KWARGS == {"sort_keys": True, "ensure_ascii": False, "separators": (",", ":")}
    assert HOLDOUT_FRACTION == 0.3
    assert BASELINE_THRESHOLD == 0.92
    assert DUPLICATE_THRESHOLD_PROBABILITY == 0.5
    assert CV_FOLDS == 5
    assert CV_SEEDS == tuple(range(1, 21))


def test_grid_caps() -> None:
    from cortex.candidates.d_boost import GRID_D
    from cortex.candidates.n_head import GRID_N

    assert len(GRID_D) <= 8, "ADR-0001 V1: grids ≤ 8 configurations"
    assert len(GRID_N) <= 8, "ADR-0001 V1: grids ≤ 8 configurations"


# ── 5. Import isolation: zero network (engine AST tripwire twin) ──────────────

#: Modules whose IMPORT would open a network surface. Zero tolerance —
#: training and evaluation are local by charter §6.
_FORBIDDEN_ROOTS = {"socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib"}


def _network_import_files() -> list[str]:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any((a.name.split(".")[0] in _FORBIDDEN_ROOTS) for a in node.names):
                    offenders.append(rel)
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in _FORBIDDEN_ROOTS:
                    offenders.append(rel)
    return offenders


def test_zero_network_imports() -> None:
    offenders = _network_import_files()
    assert not offenders, (
        "network imports are forbidden anywhere in src/cortex (charter §6, "
        f"inference-v1.md §9): {offenders}"
    )


# ── 6. CLI surface: stubs fail loud ───────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["export-corpus", "--store-uri", "file:x?mode=ro", "--out", "data/x"],
        ["pretrain", "--corpus", "c"],
        ["train", "--train-manifest", "m"],
        ["select", "--train-manifest", "m"],
        ["export-artifact", "--model", "m", "--out", "o"],
        ["eval", "--artifact", "a", "--holdout", "h", "--run-log", "r"],
    ],
)
def test_cli_subcommands_are_stubbed(argv: list[str]) -> None:
    from cortex.cli.main import NOT_IMPLEMENTED_EXIT, main

    assert main(argv) == NOT_IMPLEMENTED_EXIT


def test_cli_stub_message_on_stderr() -> None:
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from cortex.cli.main import main; "
        "raise SystemExit(main(['train', '--train-manifest', 'm']))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 2
    assert "not implemented in A3a" in result.stderr
