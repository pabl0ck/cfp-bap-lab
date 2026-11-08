"""Rebuild Tables 1--2, paired statistics, and figures using archived data only."""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import contextlib
import csv
import hashlib
import io
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from archive_io import (
    ROOT, MATRIX_DEEP, ELLIPSOID_METHODS, load_dataset, read_json, write_json,
)
import matrix_completion as mc
import ellipsoids as ell


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latex_table(path, rows, benchmark):
    def mean_sd(row, field, digits):
        text = f"{row[field + '_mean']:.{digits}f} \\pm {row[field + '_sd']:.{digits}f}"
        if field in ("time_seconds", "iterations"):
            successful = [r for r in rows if r["reported_convergence_fraction"] == 1.0]
            if row in successful and row[field + "_mean"] == min(r[field + "_mean"] for r in successful):
                text = r"\mathbf{" + text + "}"
        return "$" + text + "$"

    matrix = benchmark == "matrix_completion"
    names = {"cCRM": "cCRM", "MAP": "MAP", "Cimmino": "Cimmino", MATRIX_DEEP: "ecCRM Deep"}
    for method in ELLIPSOID_METHODS:
        if method not in names:
            suffix = method.split(", $", 1)[1].rstrip("$")
            names[method] = ("Deep ecCRM" if "Deep" in method else "ecCRM") + " ($" + suffix + "$)"
    header = (r"Method & Time (s) & Iterations & Final violation & PSD projections & Affine projections \\") if matrix else (
        r"Method & Iterations & Time (s) & Final violation \\")
    lines = [r"% Generated from archived per-instance records by reproduce.py.",
             r"\begin{tabular}{lccccc}" if matrix else r"\begin{tabular}{lccc}",
             r"\toprule", header, r"\midrule"]
    for row in rows:
        violation = f"${row['reported_residual_mean']:.2e}$"
        # Scientific notation is emitted as proper LaTeX.
        mantissa, exponent = f"{row['reported_residual_mean']:.2e}".split("e")
        violation = f"${mantissa} \\times 10^{{{int(exponent)}}}$"
        if matrix:
            cells = [names[row["Method"]], mean_sd(row, "time_seconds", 2),
                     mean_sd(row, "iterations", 0), violation,
                     mean_sd(row, "psd_projections", 0), mean_sd(row, "affine_projections", 0)]
        else:
            cells = [names[row["Method"]], mean_sd(row, "iterations", 1),
                     mean_sd(row, "time_seconds", 3), violation]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(manifest, records):
    rows = []
    ddof = manifest["config"]["table_standard_deviation_ddof"]
    for method in manifest["methods"]:
        samples = [next(row for row in rec["results"] if row["Method"] == method) for rec in records]
        row = {"Method": method, "n_instances": len(samples), "sd_ddof": ddof}
        fields = {"time_seconds": "Time"}
        if manifest["benchmark"] == "matrix_completion":
            fields.update(iterations="Iterations", reported_residual="Viol",
                          psd_projections="PSD_Projs", affine_projections="Affine_Projs")
        else:
            fields.update(iterations="Iters", reported_residual="FinalViol")
        for field, key in fields.items():
            values = np.asarray([sample[key] for sample in samples], dtype=float)
            row[field + "_mean"] = float(values.mean())
            row[field + "_sd"] = float(values.std(ddof=ddof))
        row["reported_convergence_fraction"] = float(np.mean([
            sample[fields["reported_residual"]] <= manifest["config"]["tolerance"] for sample in samples
        ]))
        rows.append(row)
    return rows


def export_one(benchmark, manifest, records, output):
    matrix = benchmark == "matrix_completion"
    number = 1 if matrix else 2
    results = []
    raw_rows = []
    observed = []
    histories = {method: [] for method in manifest["methods"]}
    stats = {method: {key: [] for key in ("Time", "Iters", "FinalViol", "Violations")}
             for method in manifest["methods"]}
    for record in records:
        if matrix:
            observed.append(record["observed_fraction"])
        for sample in record["results"]:
            method = sample["Method"]
            history = sample["Violations"]
            iterations = int(sample["Iterations"] if matrix else sample["Iters"])
            violation = sample["Viol"] if matrix else sample["FinalViol"]
            raw = {
                "instance_id": record["instance_id"], "seed": record["seed"],
                "Method": method, "time_seconds": sample["Time"],
                "iterations": iterations, "reported_residual": violation,
                "last_history_residual": history[-1],
            }
            if matrix:
                raw.update(psd_projections=sample["PSD_Projs"], affine_projections=sample["Affine_Projs"])
                results.append({key: value for key, value in sample.items() if key != "Violations"})
            else:
                for key in stats[method]:
                    stats[method][key].append(sample[key])
            raw_rows.append(raw)
            histories[method].append(history)
    write_csv(output / f"table{number}_per_instance.csv", raw_rows)
    write_csv(output / f"table{number}_seeds.csv", [
        {"instance_id": rec["instance_id"], "seed": rec["seed"]} for rec in records
    ])

    summary = summarize(manifest, records)
    by_method = {row["Method"]: row for row in summary}
    table_order = [MATRIX_DEEP, "cCRM", "MAP", "Cimmino"] if matrix else [
        ELLIPSOID_METHODS[4], ELLIPSOID_METHODS[3], ELLIPSOID_METHODS[2],
        ELLIPSOID_METHODS[1], "cCRM", "MAP", "Cimmino",
    ]
    table = [by_method[method] for method in table_order]
    write_csv(output / f"table{number}.csv", table)
    write_csv(output / f"table{number}_all_methods.csv", summary)
    latex_table(output / f"table{number}.tex", table, benchmark)
    if not matrix:
        expanded_order = [*table_order[:4], ELLIPSOID_METHODS[5], *table_order[4:]]
        latex_table(output / "table2_all_methods.tex", [by_method[m] for m in expanded_order], benchmark)

    comparison_methods = [MATRIX_DEEP] if matrix else ELLIPSOID_METHODS[1:6]
    paired = []
    pairs = []
    config = read_json(ROOT / "experiment_config.json")
    for method in comparison_methods:
        baseline_iterations, method_iterations, baseline_times, method_times = [], [], [], []
        for record in records:
            samples = {sample["Method"]: sample for sample in record["results"]}
            baseline, candidate = samples["cCRM"], samples[method]
            count_key = "Iterations" if matrix else "Iters"
            bk, mk = baseline[count_key], candidate[count_key]
            bt, mt = baseline["Time"], candidate["Time"]
            baseline_iterations.append(bk)
            method_iterations.append(mk)
            baseline_times.append(bt)
            method_times.append(mt)
            pairs.append({
                "instance_id": record["instance_id"], "seed": record["seed"],
                "baseline": "cCRM", "Method": method,
                "baseline_iterations": bk, "method_iterations": mk, "iterations_saved": bk - mk,
                "baseline_time_seconds": bt, "method_time_seconds": mt, "time_saved_seconds": bt - mt,
            })
        with contextlib.redirect_stdout(io.StringIO()):
            result = mc.print_paired_statistics(
                f"{method} vs cCRM", baseline_iterations, method_iterations,
                baseline_times, method_times,
                n_bootstrap=config["bootstrap_resamples"], seed=config["bootstrap_seed"],
            )
        result = {"baseline": "cCRM", "Method": method, **result}
        paired.append(result)
    write_csv(output / f"table{number}_pairs.csv", pairs)
    write_json(output / f"table{number}_paired_statistics.json", paired)
    headline = []
    for result in paired:
        if matrix or result["Method"] in (ELLIPSOID_METHODS[2], ELLIPSOID_METHODS[4]):
            ik = result["iteration_reduction_ci95_pct"]
            rt = result["runtime_reduction_ci95_pct"]
            headline.append(
                f"% {result['Method']} relative to cCRM; n={result['n_pairs']} paired instances.\n"
                f"Iteration reduction: {result['iteration_reduction_pct']:.1f}\\% "
                f"(95\\% paired bootstrap CI: [{ik[0]:.1f}\\%, {ik[1]:.1f}\\%]).\n"
                f"Runtime reduction: {result['runtime_reduction_pct']:.1f}\\% "
                f"(95\\% paired bootstrap CI: [{rt[0]:.1f}\\%, {rt[1]:.1f}\\%]).\n"
            )
    (output / f"table{number}_headline_statistics.tex").write_text("\n".join(headline), encoding="utf-8")

    # Export actual pointwise plot data, including the original padding rule.
    plot_rows = []
    for method, values in histories.items():
        longest = max(map(len, values))
        padded = np.asarray([history + [history[-1]] * (longest - len(history)) for history in values])
        mean, sd = padded.mean(axis=0), padded.std(axis=0, ddof=0)
        for index, (avg, spread) in enumerate(zip(mean, sd)):
            plot_rows.append({"Method": method, "iteration": index + 1,
                              "mean_violation": float(avg), "sd_violation": float(spread),
                              "n_instances": len(values)})
    write_csv(output / ("fig2_data.csv" if matrix else "fig3_data.csv"), plot_rows)
    if matrix:
        mc.render_results(results, histories, observed, len(records), output,
                          n_matrix=manifest["config"]["n"], rank=manifest["config"]["rank"],
                          observation_probability=manifest["config"]["observation_probability"])
    else:
        ell.render_results(stats, len(records), output, n=manifest["config"]["n"])


def regenerate(data_root, output, benchmarks, allow_validation=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for benchmark in benchmarks:
        manifest, records = load_dataset(Path(data_root) / benchmark)
        if manifest["profile"] != "paper" and not allow_validation:
            raise ValueError("Validation data cannot be labeled as paper results; use --allow-validation explicitly")
        inputs[benchmark] = hashlib.sha256((Path(data_root) / benchmark / "manifest.json").read_bytes()).hexdigest()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            export_one(benchmark, manifest, records, output)
        summary = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines())
        (output / f"{benchmark}_summary.txt").write_text(summary + "\n", encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "reproduction_manifest.json")
    provenance = {
        "schema_version": 1,
        "data_manifest_sha256": inputs,
        "postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    write_json(output / "reproduction_manifest.json", provenance)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--benchmark", choices=["all", "matrix_completion", "ellipsoids"], default="all")
    parser.add_argument("--allow-validation", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Regenerate in a temporary directory and compare file hashes")
    args = parser.parse_args(argv)
    benchmarks = ["matrix_completion", "ellipsoids"] if args.benchmark == "all" else [args.benchmark]
    if args.verify:
        expected = read_json(args.output_dir / "reproduction_manifest.json")
        with tempfile.TemporaryDirectory() as temporary:
            actual = regenerate(args.data_dir, Path(temporary), benchmarks, args.allow_validation)
        if actual != expected:
            raise SystemExit("Regenerated outputs differ from the archived checksums")
        print("Verified: every regenerated table, statistic, plot-data file, PDF, and PNG matches its archived checksum.")
    else:
        regenerate(args.data_dir, args.output_dir, benchmarks, args.allow_validation)
        print(f"Regenerated tables, paired statistics, and figures in {args.output_dir}")


if __name__ == "__main__":
    main()
