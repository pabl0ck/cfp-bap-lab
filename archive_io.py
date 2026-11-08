"""Lossless benchmark records, provenance, and validation (no solver calls)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "matrix_completion.py", "ellipsoids.py", "run_experiments.py",
    "archive_io.py", "experiment_config.json",
)
MATRIX_DEEP = r"ecCRM, Deep (T=P_Y P_X P_Y), $\alpha_k=0.5$"
MATRIX_METHODS = ["Cimmino", "MAP", "cCRM", MATRIX_DEEP]
ELLIPSOID_METHODS = [
    "cCRM",
    r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-1/2}$",
    r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-1}$",
    r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-2}$",
    r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=0.9^{k+1}$",
    r"ecCRM, Deep ($T=P_Y P_X P_Y$), $\alpha_k=0.9^{k+1}$",
    "MAP", "Cimmino",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def environment():
    import numpy as np
    import matplotlib
    import pandas as pd
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        np.show_config()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        ).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__},
        "thread_limits": {key: os.environ.get(key) for key in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "numpy_build_configuration": configuration.getvalue(),
        "git_parent_at_measurement": revision,
        "git_worktree_dirty_at_measurement": dirty,
        "note": "Source hashes identify the measured files even before the archive commit exists.",
    }


def validate_instance(record, manifest):
    index = record["instance_id"]
    if not isinstance(index, int) or not 0 <= index < len(manifest["config"]["seeds"]):
        raise ValueError("Unexpected instance ID")
    if record["seed"] != manifest["config"]["seeds"][index]:
        raise ValueError(f"Seed mismatch for instance {index}")
    expected = manifest["methods"]
    if [row["Method"] for row in record["results"]] != expected:
        raise ValueError(f"Missing, duplicated, or reordered methods for instance {index}")
    for row in record["results"]:
        history = row["Violations"]
        iterations = row.get("Iterations", row.get("Iters"))
        if not history or iterations != len(history):
            raise ValueError(f"History/iteration mismatch for instance {index}")
        if iterations > manifest["config"]["max_iter"]:
            raise ValueError(f"Iteration limit exceeded for instance {index}")
        values = [row["Time"], row.get("Viol", row.get("FinalViol")), *history]
        if any(not math.isfinite(value) or value < 0 for value in values) or row["Time"] <= 0:
            raise ValueError(f"Invalid numerical data for instance {index}")
        if manifest["benchmark"] == "matrix_completion" and row["Instance"] != index:
            raise ValueError("Result row does not match its instance")
        if manifest["benchmark"] == "ellipsoids" and row["FinalViol"] != history[-1]:
            raise ValueError("Reported residual does not match the final history sample")


def load_dataset(directory, verify_sources=True):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or not manifest.get("complete"):
        raise ValueError(f"Dataset is incomplete or unsupported: {directory}")
    if verify_sources and manifest["source_sha256"] != source_hashes():
        raise ValueError(f"Source files differ from the measured version: {directory}")
    expected_ids = list(range(len(manifest["config"]["seeds"])))
    records = []
    paths = sorted((directory / "instances").glob("instance_*.json"))
    if len(paths) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} instance records in {directory}")
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest["instance_sha256"].get(path.name) != digest:
            raise ValueError(f"Raw record checksum mismatch: {path}")
        record = read_json(path)
        validate_instance(record, manifest)
        records.append(record)
    if [record["instance_id"] for record in records] != expected_ids:
        raise ValueError("Instance IDs are missing or duplicated")
    return manifest, records
