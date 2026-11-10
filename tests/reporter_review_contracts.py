"""Bounded, implementation-neutral invariants for M2.7 review evidence."""

import ast
import json
import math
from pathlib import Path
import re

import joblib
import numpy as np


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def exception_chain(error):
    seen, chain = set(), []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return chain


def chain_has(error, original):
    return any(item is original for item in exception_chain(error))


def refusal_mentions(error, words):
    message = " ".join(str(e).lower() for e in exception_chain(error))
    assert any(word in message for word in words), f"Unrelated refusal: {message}"


def strict_json(path):
    def reject(token):
        raise ValueError(f"Nonstandard JSON token: {token}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def unavailable(value):
    """Accept explicit missing representations, never invented finite measures."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "nan",
            "nat",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
            "missing",
            "unavailable",
            "not available",
            "n/a",
            "undefined",
        }
    if isinstance(value, dict):
        leaves = list(text_leaves(value))
        explicit = any(
            any(
                w in text.lower()
                for w in [
                    "missing",
                    "unavailable",
                    "nonfinite",
                    "non-finite",
                    "nan",
                    "inf",
                    "nat",
                    "undefined",
                ]
            )
            for _, text in leaves
        )
        numeric = [v for v in value.values() if isinstance(v, (int, float, bool))]
        return explicit and not numeric and value.get("value") is None
    return False


def text_leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from text_leaves(child, (*path, str(key)))
    elif isinstance(value, list):
        for child in value:
            yield from text_leaves(child, path)
    elif isinstance(value, str) and value.strip():
        yield path, value


def preserve_tree(actual, expected):
    if isinstance(expected, dict):
        assert set(expected) <= set(actual), "Representable fields were dropped"
        for key in expected:
            preserve_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple, np.ndarray)):
        assert len(actual) == len(expected), "Population cardinality changed"
        for got, wanted in zip(actual, expected):
            preserve_tree(got, wanted)
    elif (
        expected is None
        or isinstance(expected, (float, np.floating))
        and not math.isfinite(expected)
    ):
        assert unavailable(
            actual
        ), f"Missing value became an invented measure: {actual!r}"
    elif isinstance(expected, np.generic):
        preserve_tree(actual, expected.item())
    else:
        assert actual == expected


def text_value(text, label):
    # Whitespace and blank lines are presentation, not scientific information.
    matches = re.findall(
        r"^\s*-?\s*" + re.escape(label) + r"\s*:\s*(.*?)\s*$", text, re.MULTILINE
    )
    assert len(matches) == 1, f"Missing or ambiguous scientific field: {label}"
    return matches[0]


def model_is_unavailable(summary):
    for key, value in summary.items():
        if "model" in key.lower():
            assert unavailable(value), f"Unprovided model provenance: {key}={value!r}"


def metric_diagnostic(report, records, logger_name, metric, original):
    """Require both the affected metric and a nonempty original cause."""
    metric_tokens = ["ae", "autoencoder"] if metric == "ae" else ["pca"]
    cause = str(original).lower()

    def relevant(path, text):
        context = " ".join(path).lower() + " " + text.lower()
        identified = any(
            re.search(r"\b" + word + r"\b|" + word + r"_", context)
            for word in metric_tokens
        )
        # Keys may identify a metric, but cannot supply a failure reason.
        explained = (
            cause in text.lower() or type(original).__name__.lower() in text.lower()
        )
        return identified and explained

    structured = any(relevant(path, text) for path, text in text_leaves(report))
    logged = any(
        record.name == logger_name
        and record.levelno >= 30
        and relevant((), record.getMessage())
        for record in records
    )
    return structured or logged


def incomplete_marker(root, aliases):
    """Recognize a bounded explicit marker linking every affected public alias."""
    for path in root.rglob("*.json"):
        try:
            marker = strict_json(path)
        except (ValueError, OSError):
            continue
        leaves = [text.lower().replace("\\", "/") for _, text in text_leaves(marker)]
        if not any(
            text in ["incomplete", "failed", "unavailable", "aborted"]
            for text in leaves
        ):
            continue
        if all(any(alias.name.lower() in text for text in leaves) for alias in aliases):
            return True
    return False


def embedding_group(root, generations):
    """Read public archives and aliases; verify population, values and timestamp."""
    run_dir = root / "runs/phy3500_umap_tsne"
    legacy = root / "phy3500_embeddings.joblib"
    aliases = [
        run_dir / "phy3500_embeddings_latest.joblib",
        legacy,
        run_dir / "phy3500_umap_tsne_run_latest.json",
        run_dir / "phy3500_umap_tsne_run_latest.txt",
    ]
    if incomplete_marker(root, aliases):
        return "explicitly incomplete"

    def read_group(paths):
        jb, legacy_file, js, txt = paths
        assert all(p.is_file() for p in paths), "Published group has missing members"
        try:
            payload = joblib.load(jb)
            report = strict_json(js)
        except Exception as error:
            raise AssertionError(
                "Published group contains unreadable artifacts"
            ) from error
        text = txt.read_text(encoding="utf-8")
        labels = payload["y"].tolist()
        chosen = [g for g in generations if labels == list(g["y"])]
        assert len(chosen) == 1, "Payload population is unidentifiable"
        generation = chosen[0]
        np.testing.assert_array_equal(payload["Z_umap"], generation["Z_umap"])
        other = joblib.load(legacy_file)
        np.testing.assert_array_equal(other["y"], generation["y"])
        np.testing.assert_array_equal(other["Z_umap"], generation["Z_umap"])
        counts = {
            label: list(generation["y"]).count(label) for label in set(generation["y"])
        }
        assert report["class_counts"] == counts, "JSON population differs from joblib"
        expected_x_mean = math.fsum(
            float(row[0]) for row in generation["Z_umap"]
        ) / len(generation["Z_umap"])
        assert math.isclose(
            report["embedding_stats"]["umap"]["x_mean"], expected_x_mean, rel_tol=1e-14
        ), "JSON coordinates differ from joblib"
        assert (
            ast.literal_eval(text_value(text, "y")) == counts
        ), "TXT population differs from joblib"
        assert (
            report["run_timestamp_utc"]
            == payload["timestamp_utc"]
            == other["timestamp_utc"]
        )
        assert report["run_timestamp_utc"] == text_value(text, "Timestamp (UTC)")
        return labels

    result = read_group(aliases)
    # New orphaned timestamped files are public artifacts too, not a valid rollback.
    stamps = set()
    for path in run_dir.glob("*"):
        match = re.search(r"(\d{8}T\d{6}Z)", path.name)
        if match:
            stamps.add(match[1])
    for stamp in stamps:
        jb = run_dir / f"phy3500_embeddings_{stamp}.joblib"
        read_group(
            [
                jb,
                jb,
                run_dir / f"phy3500_umap_tsne_run_{stamp}.json",
                run_dir / f"phy3500_umap_tsne_run_{stamp}.txt",
            ]
        )
    return result
