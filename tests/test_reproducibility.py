"""Integrity checks for archived instance pairing and numerical exports."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

from archive_io import MATRIX_METHODS, read_json, validate_instance
from matrix_completion import print_paired_statistics
from reproduce import summarize, write_csv


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "benchmark": "matrix_completion", "methods": MATRIX_METHODS,
            "config": {"seeds": [42, 43], "max_iter": 10,
                       "tolerance": 0.01, "table_standard_deviation_ddof": 1},
        }
        self.record = {
            "instance_id": 0, "seed": 42,
            "results": [{"Instance": 0, "Method": method, "Time": 0.2,
                         "Iterations": 2, "Viol": 0.005,
                         "PSD_Projs": 4, "Affine_Projs": 4,
                         "Violations": [0.1, 0.005]}
                        for method in MATRIX_METHODS],
        }

    def test_paired_records_require_identical_instance_and_seed(self):
        validate_instance(self.record, self.manifest)
        wrong = deepcopy(self.record)
        wrong["seed"] = 43
        with self.assertRaisesRegex(ValueError, "Seed mismatch"):
            validate_instance(wrong, self.manifest)
        wrong = deepcopy(self.record)
        wrong["results"][0]["Instance"] = 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_instance(wrong, self.manifest)

    def test_missing_method_and_truncated_history_are_rejected(self):
        wrong = deepcopy(self.record)
        wrong["results"].pop()
        with self.assertRaisesRegex(ValueError, "methods"):
            validate_instance(wrong, self.manifest)
        wrong = deepcopy(self.record)
        wrong["results"][0]["Violations"].pop()
        with self.assertRaisesRegex(ValueError, "History/iteration"):
            validate_instance(wrong, self.manifest)

    def test_joint_bootstrap_preserves_pair_dependence(self):
        # Every paired candidate is exactly half its baseline: every bootstrap
        # sample must retain the same 50% reduction for both metrics.
        with contextlib.redirect_stdout(io.StringIO()):
            result = print_paired_statistics(
                "known proportional pairs", [10, 30, 100], [5, 15, 50],
                [1, 3, 10], [0.5, 1.5, 5], n_bootstrap=500, seed=71,
            )
        self.assertEqual(result["iteration_reduction_ci95_pct"], (50.0, 50.0))
        self.assertEqual(result["runtime_reduction_ci95_pct"], (50.0, 50.0))
        self.assertEqual(result["extra_runtime_reduction_ci95_pp"], (0.0, 0.0))

    def test_aggregation_uses_raw_measurements_and_declared_sd(self):
        other = deepcopy(self.record)
        other.update(instance_id=1, seed=43)
        for row in other["results"]:
            row.update(Instance=1, Time=0.4)
        rows = summarize(self.manifest, [self.record, other])
        self.assertAlmostEqual(rows[0]["time_seconds_mean"], 0.3)
        self.assertAlmostEqual(rows[0]["time_seconds_sd"], 0.2 / 2**0.5)
        self.assertEqual(rows[0]["reported_convergence_fraction"], 1.0)

    def test_csv_keeps_float_round_trip(self):
        import csv
        value = 0.12345678901234568
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            write_csv(path, [{"time": value}])
            with path.open(newline="") as stream:
                restored = float(next(csv.DictReader(stream))["time"])
        self.assertEqual(value, restored)


if __name__ == "__main__":
    unittest.main()
