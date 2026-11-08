"""Run the unchanged numerical kernels and checkpoint every measured instance."""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import numpy as np

from archive_io import (
    ROOT, ELLIPSOID_METHODS, MATRIX_METHODS, environment, read_json,
    source_hashes, validate_instance, write_json,
)
import matrix_completion as mc
import ellipsoids as ell


def measure_matrix(config, instance_id, seed):
    px = mc.PSDConeProjector(config["n"])
    py, z0, observed_fraction = mc.generate_matrix_completion_problem(
        n=config["n"], rank=config["rank"],
        observation_probability=config["observation_probability"],
        rng=np.random.default_rng(seed),
    )
    results = []
    for cfg in mc.make_configs(px, py):
        kwargs = dict(z0=z0, P_X=px, P_Y=py,
                      max_iter=config["max_iter"], tol=config["tolerance"])
        if cfg["m_type"] == "ap":
            trajectory, history, elapsed = mc.ap_iterates(**kwargs)
        else:
            start = time.perf_counter()
            if cfg["m_type"] == "cimmino":
                trajectory, history = mc.cimmino_iterates(**kwargs)
            else:
                trajectory, history = mc.eccrm_iterates(
                    **kwargs, T=cfg["T"], alpha=config["alpha"],
                )
            elapsed = time.perf_counter() - start
        final = trajectory[-1]
        # Preserve the original external check, outside the measured region.
        violation = max(np.linalg.norm(final - px(final)),
                        np.linalg.norm(final - py(final)))
        count = len(history)
        results.append({
            "Instance": instance_id, "Method": cfg["name"],
            "Time": elapsed, "Iterations": count, "Viol": float(violation),
            "Converged": bool(violation <= config["tolerance"]),
            "PSD_Projs": count * cfg["psd_per_iter"],
            "Affine_Projs": count * cfg["affine_per_iter"],
            "Violations": [float(value) for value in history],
        })
    return {
        "instance_id": instance_id, "seed": seed,
        "observed_fraction": float(observed_fraction),
        "results": results,
    }


def measure_ellipsoids(config, instance_id, seed):
    results = ell.run_single_instance(
        np.random.default_rng(seed), max_iter=config["max_iter"],
        tol=config["tolerance"], n=config["n"],
    )
    for row in results:
        row["Time"] = float(row["Time"])
        row["FinalViol"] = float(row["FinalViol"])
        row["Violations"] = [float(value) for value in row["Violations"]]
    return {"instance_id": instance_id, "seed": seed, "results": results}


def run_dataset(benchmark, config, data_root, profile, resume=False):
    directory = Path(data_root) / benchmark
    manifest_path = directory / "manifest.json"
    current_sources = source_hashes()
    current_environment = environment()
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; use --resume or a new --data-dir")
        manifest = read_json(manifest_path)
        if manifest["config"] != config or manifest["source_sha256"] != current_sources:
            raise ValueError("Cannot mix instances from different configurations or source files")
        for field in ("python", "platform", "versions", "thread_limits", "numpy_build_configuration"):
            if manifest["environment"][field] != current_environment[field]:
                raise ValueError(f"Cannot mix timed measurements from different environments ({field})")
    else:
        manifest = {
            "schema_version": 1, "benchmark": benchmark, "profile": profile,
            "config": config, "complete": False,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "methods": MATRIX_METHODS if benchmark == "matrix_completion" else ELLIPSOID_METHODS,
            "source_sha256": current_sources, "environment": current_environment,
            "measurement_origin": "Fresh run; original unsaved timing samples cannot be reconstructed.",
            "instance_sha256": {},
        }
        write_json(manifest_path, manifest)
    measure = measure_matrix if benchmark == "matrix_completion" else measure_ellipsoids
    for instance_id, seed in enumerate(config["seeds"]):
        path = directory / "instances" / f"instance_{instance_id:03d}.json"
        if path.exists():
            record = read_json(path)
            validate_instance(record, manifest)
            if manifest["instance_sha256"].get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Existing record checksum mismatch: {path}")
            print(f"{benchmark}: keeping instance {instance_id + 1}/{len(config['seeds'])} (seed {seed})", flush=True)
            continue
        start = time.perf_counter()
        record = measure(config, instance_id, seed)
        record["measured_utc"] = datetime.now(timezone.utc).isoformat()
        validate_instance(record, manifest)
        write_json(path, record)
        manifest["instance_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(manifest_path, manifest)
        print(f"{benchmark}: saved instance {instance_id + 1}/{len(config['seeds'])} "
              f"(seed {seed}, {time.perf_counter() - start:.1f}s)", flush=True)
    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["all", "matrix_completion", "ellipsoids"], default="all")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--instances", type=int, default=None, help="Use the first N listed seeds (default: all 50)")
    parser.add_argument("--smoke", action="store_true", help="Two small validation instances, never paper data")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.instances is not None and not 2 <= args.instances <= 50:
        parser.error("--instances must be between 2 and 50")
    data_root = args.data_dir or (ROOT / "validation_data" if args.smoke else ROOT / "data")
    defaults = read_json(ROOT / "experiment_config.json")
    benchmarks = ["matrix_completion", "ellipsoids"] if args.benchmark == "all" else [args.benchmark]
    for benchmark in benchmarks:
        config = dict(defaults[benchmark])
        count = args.instances or (2 if args.smoke else 50)
        config["seeds"] = config["seeds"][:count]
        profile = "paper" if count == 50 and not args.smoke else "validation"
        if args.smoke:
            config["n"] = 24 if benchmark == "matrix_completion" else 32
            config["max_iter"] = 30
        run_dataset(benchmark, config, data_root, profile, args.resume)
    print(f"Saved measurements to {data_root}. Run reproduce.py to rebuild summaries and figures.", flush=True)


if __name__ == "__main__":
    main()
