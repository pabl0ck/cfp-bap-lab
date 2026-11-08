# Deep Centralization: reproducibility package

Numerical experiments for **Pablo Barros, Deep Centralization for the Circumcentered Reflection Method**.

Release tag: **`eccrm-repro-v1`**. Cite the tagged snapshot rather than the moving default branch:

<https://github.com/pabl0ck/cfp-bap-lab/tree/eccrm-repro-v1>

The package contains experiment scripts, all instance-generator seeds, lossless per-instance measurements and convergence histories, and a separate postprocessor for Tables 1 and 2, paired statistics, and the figures. No benchmark is rerun when regenerating the paper outputs from archived records.

## Archived measurements and provenance

The checked-in `data/` records are fresh complete runs of the archived scripts. Earlier unsaved timing samples cannot be reconstructed from published averages. The tables and figures in `artifacts/` are computed from these checked-in records; use those generated values and the recorded environment when preparing the manuscript associated with this release.

Each dataset manifest records the configuration, seed list, Python/platform and NumPy build details, thread limits, source SHA-256 hashes, and a checksum for every instance file. Each instance file contains its ID, seed, methods, elapsed times, iteration counts, reported residuals, and complete residual histories. JSON stores binary64 values losslessly through Python's round-trip float representation. Derived CSV files preserve full precision.

| Paper output | Dataset | Instances and seeds | Configuration |
| --- | --- | --- | --- |
| Table 1; matrix-completion figure (`fig2`) | `data/matrix_completion/` | 50; seeds 42–91 | 120 × 120; rank 10; observation probability 0.75; tolerance 0.01; limit 1500 |
| Table 2; ellipsoid figure (`fig3`) | `data/ellipsoids/` | 50; seeds 1234–1283 | dimension 3000; eigenvalues in [1,5]; center separation 1.9 in the Q norm; tolerance 1e-12; limit 600 |

`experiment_config.json` lists every seed explicitly. Both generators use `numpy.random.default_rng(seed)` (PCG64). Every method receives the same generated problem and initial point within an instance. The ellipsoid archive also retains the additional deep-kernel comparator shown in the figure; Table 2 contains the seven methods used in the paper table.

## Rebuild the paper outputs from saved measurements

Use Python 3.12. The manifests record the interpreter and numerical libraries used for the measurements. The dependency lock pins the environment used to regenerate the paper outputs.

```bash
git clone https://github.com/pabl0ck/cfp-bap-lab.git
cd cfp-bap-lab
git checkout eccrm-repro-v1
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell), then run:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

The postprocessor validates dataset completeness, method/instance pairing, record checksums, and source hashes before computing outputs. Its only inputs are the archived measurements and configuration. It does not call projection operators or instance generators.

Generated files:

| File | Content |
| --- | --- |
| `artifacts/table1.tex`, `table2.tex` | LaTeX tabular environments ready to insert into the paper; bold best means among methods that converged on every instance |
| `artifacts/table1.csv`, `table2.csv` | Unrounded means and standard deviations behind each paper table |
| `artifacts/table2_all_methods.tex` | Expanded ellipsoid table including the additional deep-kernel experiment |
| `artifacts/table1_per_instance.csv`, `table2_per_instance.csv` | One row per instance and method, including seed, measured runtime, iterations, and reported final residual |
| `artifacts/table1_pairs.csv`, `table2_pairs.csv` | Explicit matched-instance comparisons against cCRM |
| `artifacts/table1_paired_statistics.json`, `table2_paired_statistics.json` | Paired differences and bootstrap confidence intervals |
| `artifacts/fig2_data.csv`, `fig3_data.csv` | Exact pointwise curve means and standard deviations |
| `artifacts/fig2.pdf`, `fig2.png`, `fig3.pdf`, `fig3.png` | Regenerated figures |
| `artifacts/*_summary.txt` | Human-readable summaries and paired comparisons |
| `artifacts/reproduction_manifest.json` | Input and output checksums |

With the same rendering environment, verify byte-for-byte regeneration of every derived file:

```bash
python reproduce.py --verify
```

The numerical table entries, paired statistics, and curve values are obtained exactly from the stored records. New benchmark timings depend on hardware and load, and rerunning floating-point eigendecompositions can introduce platform-dependent roundoff. Figure PDF/PNG bytes and platform text formatting can also depend on the plotting environment; the exported curve data specifies the numerical figure independently of rendering. The repository fixes source and archived-data line endings to LF so checksum validation also works with Windows Git configurations.

## Paired statistics and aggregation conventions

For the baseline values b_i and candidate values m_i, the headline reduction is `100 * (1 - mean(m) / mean(b))`. It is the reduction of means, rather than the mean of per-instance percentage reductions. The archive also gives the mean and sample SD of the paired differences b_i − m_i.

Confidence intervals use 10,000 paired bootstrap resamples with seed **20261106**. A resample selects complete instance IDs jointly for both methods, iteration counts, and runtimes. The 2.5% and 97.5% quantiles give the percentile intervals. The same joint resamples estimate the runtime reduction minus iteration reduction in percentage points.

To preserve the experiment scripts' aggregation, Table 1 uses sample SD (`ddof=1`), while Table 2 uses population SD (`ddof=0`). These choices are recorded explicitly in the manifests. All paired-difference SDs use `ddof=1`. Pointwise figure SDs use `ddof=0`; shorter histories are extended by their final recorded value within each method before averaging.

## Measurement and stopping conventions

The numerical kernels, circumcenter safeguard, projection tolerances, method order, and timing boundaries are fixed in this release.

- Matrix completion: method order is Cimmino, MAP, cCRM, deep ecCRM. MAP times only its alternating-projection updates and excludes the extra projection used by its stopping check. Other methods use the timing boundaries in `matrix_completion.py`. The externally evaluated final feasibility gap is outside all timed regions. Projection-count columns retain the original scripts' per-iteration accounting; diagnostic projections are excluded.
- Ellipsoids: methods are executed sequentially in the order listed in each manifest. The timer encloses each original iteration routine. The reported residual is the final recorded stopping-test value: at the centralized point for circumcentered methods, at the new iterate for MAP, and at the input iterate of the final Cimmino pass. The latter routine may return an additional averaged point. These existing conventions are retained and documented rather than silently changed.
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` are set to 1 before NumPy is imported. Instance generation and archive writing are outside the measured algorithm times.

A run that reaches the iteration cap remains in the raw records and plots; it is not treated as a successful solve merely because a runtime is available. The generated tables highlight runtimes only among methods that reached tolerance on every instance.

## Rerun the experiments and archive new measurements

The full benchmarks take substantially longer than reprocessing the saved records. Use a new directory to keep the released data intact:

```bash
python run_experiments.py --data-dir rerun_data
python reproduce.py --data-dir rerun_data --output-dir rerun_artifacts
```

Equivalent per-benchmark entry points:

```bash
python matrix_completion.py --data-dir rerun_data
python ellipsoids.py --data-dir rerun_data
```

Records are checkpointed after every instance. An interrupted run can resume only when the configuration, measured source files, and environment agree:

```bash
python run_experiments.py --data-dir rerun_data --resume
```

For a quick installation and pipeline check:

```bash
python run_experiments.py --smoke
python reproduce.py --data-dir validation_data --output-dir validation_artifacts --allow-validation
python reproduce.py --data-dir validation_data --output-dir validation_artifacts --allow-validation --verify
python -m unittest discover -s tests -v
```

Smoke records use two small instances and are explicitly labeled validation data. They cannot be processed as paper results without the explicit `--allow-validation` option.

## Repository structure

- `matrix_completion.py`, `ellipsoids.py`: numerical methods, projectors, generators, paired summaries, and plotting routines.
- `run_experiments.py`: the measured experiment runner and per-instance checkpoints.
- `archive_io.py`: provenance, lossless storage, and data integrity checks.
- `reproduce.py`: standalone postprocessing from saved measurements.
- `experiment_config.json`: problem parameters, seed lists, and bootstrap parameters.
- `data/`: the complete measured archive and its manifests.
- `artifacts/`: all regenerated paper outputs and their checksums.
- `requirements.txt`: pinned Python dependencies.
- `tests/`: integrity, pairing, float round-trip, and paired-bootstrap checks.
- `paper/availability.tex`: version-pinned availability statement.
