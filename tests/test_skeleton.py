"""Skeleton contract tests (A3a).

Pins the slice's own deliverables:

1. every module imports and the contracted signatures exist;
2. ARTIFACT_NAME is defined in exactly ONE place (ADR 0001 П1);
3. FEATURE_NAMES is frozen — order and composition are contract;
4. grid caps (≤ 8 per ADR 0001 V1);
5. zero network imports across src/cortex — the repo-side twin of the
   engine's tests/test_mcp_core_isolation.py AST tripwire;
6. CLI surface: all six subcommands implemented (A2 landed export-corpus);
   every command fails loud on bad input (exit 2, stderr message) — never
   fake success. End-to-end command contracts live in tests/test_cli.py.
"""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "cortex"

# ── 1. Modules import and signatures exist ────────────────────────────────────


def _disputed_pair():
    from cortex.data.holdout import SplitPair

    return SplitPair("a--b", "P1", "disputed", "x", "y", "z")


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

    assert cortex.ARTIFACT_NAME == cortex.artifacts.ARTIFACT_NAME == "vesma-cortex-v1"


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


def test_algorithm_modules_implemented() -> None:
    """A3b replaced the honest stubs: the contracted entry points now
    execute (or validate loudly) instead of raising NotImplementedError.
    export-corpus landed in A2 (see tests/test_store_export.py)."""
    from cortex.artifacts import build_metadata_props
    from cortex.data.holdout import split_holdout
    from cortex.features.pair import PairRecord, features

    # contract violations surface as ValueError (typed), never as a stub
    with pytest.raises(ValueError):
        features(PairRecord("a", "b"), PairRecord("a", "b"), similarity=1.5)
    with pytest.raises(ValueError):
        split_holdout([_disputed_pair()])
    with pytest.raises(ValueError):
        build_metadata_props(embedder_pin="", corpus_fingerprint="", trained_at="", candidate="x", feature_names=())


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


# ── 6. CLI surface: all six commands implemented (A2 landed export-corpus) ───


def test_cli_subcommands_surface() -> None:
    from cortex.cli.main import build_parser

    parser = build_parser()
    subcommands = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subcommands = set(action.choices)
    assert subcommands == {
        "export-corpus",
        "pretrain",
        "train",
        "select",
        "export-artifact",
        "eval",
    }


def test_cli_export_corpus_missing_store_fails_loud(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """export-corpus is implemented (A2): a missing store is a loud usage
    error (exit 2 + stderr), never fake success and never a stub claim."""
    from cortex.cli.main import main

    code = main(["export-corpus", "--store-uri", "file:x?mode=ro", "--out", str(tmp_path / "x")])
    assert code == 2
    stderr = capsys.readouterr().err
    assert stderr.strip()
    assert "stub" not in stderr.lower()


@pytest.mark.parametrize(
    "argv",
    [
        ["pretrain", "--corpus", "does-not-exist.jsonl", "--out", "x"],
        ["train", "--train-manifest", "does-not-exist.jsonl", "--out", "x"],
        ["select", "--train-manifest", "does-not-exist.jsonl"],
        ["eval", "--artifact", "does-not-exist.onnx", "--holdout", "h", "--run-log", "r"],
    ],
)
def test_cli_implemented_commands_fail_loud(argv: list[str], capsys: pytest.CaptureFixture) -> None:
    """The A3b commands are implemented: a missing input is a loud usage
    error (exit 2 + stderr), NOT the A3a stub path and never success."""
    from cortex.cli.main import main

    assert main(argv) == 2
    stderr = capsys.readouterr().err
    assert stderr.strip(), "failures must be loud on stderr"
    assert "A3a" not in stderr, "implemented commands must not claim stub status"
