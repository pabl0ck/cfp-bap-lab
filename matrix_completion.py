import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
from pathlib import Path
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Vector = np.ndarray
Operator = Callable[[Vector], Vector]
AlphaSchedule = Union[float, Callable[[int], float]]


def safe_format(mean: float, std: float, fmt: str = ".0f") -> str:
    if pd.isna(std):
        return f"{mean:{fmt}} ± NaN"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def circumcenter(
    z: Vector,
    v: Vector,
    w: Vector,
    tol: float = 1e-12,
) -> Vector:
    z = np.asarray(z)
    v = np.asarray(v)
    w = np.asarray(w)

    u1 = v - z
    u2 = w - z

    dot11 = np.dot(u1, u1)
    dot22 = np.dot(u2, u2)
    dot12 = np.dot(u1, u2)

    det = dot11 * dot22 - dot12**2

    if abs(det) <= tol * (dot11 + dot22 + tol):
        d_vw = np.dot(v - w, v - w)

        if dot11 >= dot22 and dot11 >= d_vw:
            return 0.5 * (z + v)

        if dot22 >= dot11 and dot22 >= d_vw:
            return 0.5 * (z + w)

        return 0.5 * (v + w)

    rhs1 = 0.5 * dot11
    rhs2 = 0.5 * dot22

    lam1 = (dot22 * rhs1 - dot12 * rhs2) / det
    lam2 = (dot11 * rhs2 - dot12 * rhs1) / det

    return z + lam1 * u1 + lam2 * u2


def eccrm_iterates(
    z0: Vector,
    P_X: Operator,
    P_Y: Operator,
    T: Operator,
    alpha: AlphaSchedule,
    max_iter: int,
    tol: float,
) -> Tuple[List[Vector], List[float]]:
    z = np.asarray(z0, dtype=float)

    traj = [z.copy()]
    viol_hist: List[float] = []

    for k in range(max_iter):
        alpha_k = float(alpha(k)) if callable(alpha) else float(alpha)

        t_z = T(z)
        px_tz = P_X(t_z)

        y = alpha_k * t_z + (1.0 - alpha_k) * px_tz

        # Since y lies on [Tz, P_X(Tz)], P_X(y) = P_X(Tz).
        p_x_y = px_tz
        p_y_y = P_Y(y)

        viol = max(
            np.linalg.norm(y - p_x_y),
            np.linalg.norm(y - p_y_y),
        )
        viol_hist.append(viol)

        if viol < tol:
            traj.append(y.copy())
            break

        r_x = 2.0 * p_x_y - y
        r_y = 2.0 * p_y_y - y

        z = circumcenter(y, r_x, r_y)
        traj.append(z.copy())

    return traj, viol_hist


def ap_iterates(
    z0: Vector,
    P_X: Operator,
    P_Y: Operator,
    max_iter: int,
    tol: float,
) -> Tuple[List[Vector], List[float], float]:
    """
    Alternating projections.

    The additional P_X evaluation used only to test feasibility is excluded
    from the reported algorithmic runtime and projection counts.
    """
    z = np.asarray(z0, dtype=float)

    traj = [z.copy()]
    viol_hist: List[float] = []
    algorithm_time = 0.0

    for _ in range(max_iter):
        # Time only the MAP iteration z_new = P_Y(P_X(z)).
        start_time = time.perf_counter()

        px = P_X(z)
        z_new = P_Y(px)

        algorithm_time += time.perf_counter() - start_time

        # External stopping test, excluded from time and projection counts.
        # Since z_new belongs to Y, delta(z_new) = dist(z_new, X).
        viol = np.linalg.norm(z_new - P_X(z_new))

        viol_hist.append(viol)
        traj.append(z_new.copy())

        if viol < tol:
            break

        z = z_new

    return traj, viol_hist, algorithm_time


def cimmino_iterates(
    z0: Vector,
    P_X: Operator,
    P_Y: Operator,
    max_iter: int,
    tol: float,
) -> Tuple[List[Vector], List[float]]:
    z = np.asarray(z0, dtype=float)

    traj = [z.copy()]
    viol_hist: List[float] = []

    for _ in range(max_iter):
        px = P_X(z)
        py = P_Y(z)

        # These projections are already required by the Cimmino iteration.
        viol = max(
            np.linalg.norm(z - px),
            np.linalg.norm(z - py),
        )
        viol_hist.append(viol)

        if viol < tol:
            break

        z = 0.5 * (px + py)
        traj.append(z.copy())

    return traj, viol_hist


class PSDConeProjector:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, z_vec: np.ndarray) -> np.ndarray:
        Z = z_vec.reshape((self.n, self.n))
        Z_sym = 0.5 * (Z + Z.T)

        eigenvalues, eigenvectors = np.linalg.eigh(Z_sym)
        eigenvalues_plus = np.maximum(eigenvalues, 0.0)

        X = eigenvectors @ (
            eigenvalues_plus[:, None] * eigenvectors.T
        )

        return X.ravel()


class AffineProjector:
    def __init__(
        self,
        n: int,
        indices: Tuple[np.ndarray, np.ndarray],
        values: np.ndarray,
    ):
        self.n = n
        self.indices = indices
        self.values = values

    def __call__(self, z_vec: np.ndarray) -> np.ndarray:
        z_out = z_vec.copy()
        Z_out = z_out.reshape((self.n, self.n))

        Z_out[self.indices] = self.values

        return z_out


def generate_matrix_completion_problem(
    n: int,
    rank: int,
    observation_probability: float,
    rng: np.random.Generator,
) -> Tuple[AffineProjector, np.ndarray, float]:
    """
    Generate one independent rank-r PSD matrix-completion problem and one
    symmetric starting matrix shared by all methods.
    """
    # Rank-r PSD target A = L L^T.
    L = rng.standard_normal((n, rank))
    X_star = L @ L.T

    # Sample the upper triangle and reflect it. Thus each matrix entry has
    # marginal observation probability approximately observation_probability.
    upper_mask = np.triu(
        rng.random((n, n)) < observation_probability
    )
    mask = upper_mask | upper_mask.T

    indices = np.where(mask)
    values = X_star[indices]

    P_Y = AffineProjector(
        n=n,
        indices=indices,
        values=values,
    )

    # Symmetric initial matrix, shared by every method on this problem.
    G = rng.standard_normal((n, n))
    G = 0.5 * (G + G.T)
    z0 = (1.0 * G).ravel()

    observed_fraction = np.count_nonzero(mask) / (n * n)

    return P_Y, z0, observed_fraction


def make_configs(
    P_X: Operator,
    P_Y: Operator,
):
    return [
        {
            "name": "Cimmino",
            "m_type": "cimmino",
            "T": None,
            "psd_per_iter": 1,
            "affine_per_iter": 1,
        },
        {
            "name": "MAP",
            "m_type": "ap",
            "T": None,
            # Only the projections defining the MAP iteration are counted.
            "psd_per_iter": 1,
            "affine_per_iter": 1,
        },
        {
            "name": "cCRM",
            "m_type": "eccrm",
            "T": lambda z, px=P_X, py=P_Y: py(px(z)),
            "psd_per_iter": 2,
            "affine_per_iter": 2,
        },
        {
            "name": r"ecCRM, Deep (T=P_Y P_X P_Y), $\alpha_k=0.5$",
            "m_type": "eccrm",
            "T": lambda z, px=P_X, py=P_Y: py(px(py(z))),
            "psd_per_iter": 2,
            "affine_per_iter": 3,
        },
    ]


def print_paired_statistics(
    label,
    baseline_iters,
    method_iters,
    baseline_times,
    method_times,
    n_bootstrap=10000,
    seed=20261106,
):
    """Summarize matched instances; positive differences favor the method."""
    b_k, m_k, b_t, m_t = [
        np.asarray(values, dtype=float)
        for values in (
            baseline_iters, method_iters, baseline_times, method_times
        )
    ]
    arrays = (b_k, m_k, b_t, m_t)
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("Paired inputs must be one-dimensional.")
    if len({len(values) for values in arrays}) != 1 or len(b_k) < 2:
        raise ValueError("At least two complete instance pairs are required.")
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Paired inputs contain missing or nonfinite values.")
    if np.any(b_k <= 0) or np.any(b_t <= 0):
        raise ValueError("Baseline iterations and runtimes must be positive.")

    differences_k = b_k - m_k
    differences_t = b_t - m_t
    reduction_k = 100.0 * (1.0 - m_k.mean() / b_k.mean())
    reduction_t = 100.0 * (1.0 - m_t.mean() / b_t.mean())

    # Resample entire instance pairs jointly for both metrics. This retains
    # within-instance dependence between methods, iteration counts and times.
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(b_k), size=(n_bootstrap, len(b_k)))
    bootstrap_k = 100.0 * (
        1.0 - m_k[ids].mean(axis=1) / b_k[ids].mean(axis=1)
    )
    bootstrap_t = 100.0 * (
        1.0 - m_t[ids].mean(axis=1) / b_t[ids].mean(axis=1)
    )
    ci_k = np.quantile(bootstrap_k, [0.025, 0.975])
    ci_t = np.quantile(bootstrap_t, [0.025, 0.975])
    ci_extra = np.quantile(bootstrap_t - bootstrap_k, [0.025, 0.975])

    print(f"\nPAIRED COMPARISON: {label} (n={len(b_k)})")
    print(
        "  Iterations saved, mean +/- SD: "
        f"{differences_k.mean():.3f} +/- {differences_k.std(ddof=1):.3f}"
    )
    print(
        f"  Reduction in mean iterations: {reduction_k:.2f}% "
        f"(95% paired bootstrap CI: {ci_k[0]:.2f}%, {ci_k[1]:.2f}%)"
    )
    print(
        "  Time saved (s), mean +/- SD: "
        f"{differences_t.mean():.4f} +/- {differences_t.std(ddof=1):.4f}"
    )
    print(
        f"  Reduction in mean runtime: {reduction_t:.2f}% "
        f"(95% paired bootstrap CI: {ci_t[0]:.2f}%, {ci_t[1]:.2f}%)"
    )
    print(
        "  Runtime reduction minus iteration reduction: "
        f"{reduction_t - reduction_k:.2f} percentage points "
        f"(95% paired bootstrap CI: {ci_extra[0]:.2f}, {ci_extra[1]:.2f})"
    )

    return {
        "n_pairs": len(b_k),
        "iterations_saved_mean": float(differences_k.mean()),
        "iterations_saved_sd": float(differences_k.std(ddof=1)),
        "time_saved_mean": float(differences_t.mean()),
        "time_saved_sd": float(differences_t.std(ddof=1)),
        "iteration_reduction_pct": float(reduction_k),
        "runtime_reduction_pct": float(reduction_t),
        "iteration_reduction_ci95_pct": tuple(float(x) for x in ci_k),
        "runtime_reduction_ci95_pct": tuple(float(x) for x in ci_t),
        "extra_runtime_reduction_pp": float(reduction_t - reduction_k),
        "extra_runtime_reduction_ci95_pp": tuple(float(x) for x in ci_extra),
    }


def run_feasibility_experiment(num_instances=50):
    """Run and archive the experiment; see run_experiments.py for the CLI."""
    from run_experiments import main
    main(["--benchmark", "matrix_completion", "--instances", str(num_instances)])


def render_results(results, trajectories, observed_fractions, num_instances,
                   output_dir, n_matrix=120, rank=10,
                   observation_probability=0.75):
    """Regenerate the original summary and figure from archived measurements."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_names = list(trajectories)
    df = pd.DataFrame(results)

    summary = (
        df.groupby("Method")
        .agg(
            Time_mean=("Time", "mean"),
            Time_std=("Time", "std"),
            Iters_mean=("Iterations", "mean"),
            Iters_std=("Iterations", "std"),
            Viol_mean=("Viol", "mean"),
            Viol_std=("Viol", "std"),
            Convergence_rate=("Converged", "mean"),
            PSD_Projs_mean=("PSD_Projs", "mean"),
            PSD_Projs_std=("PSD_Projs", "std"),
            Affine_Projs_mean=("Affine_Projs", "mean"),
            Affine_Projs_std=("Affine_Projs", "std"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        ["Convergence_rate", "Time_mean"],
        ascending=[False, True],
    )

    summary["Time (s)"] = summary.apply(
        lambda row: safe_format(
            row["Time_mean"],
            row["Time_std"],
            ".2f",
        ),
        axis=1,
    )

    summary["Iterations"] = summary.apply(
        lambda row: safe_format(
            row["Iters_mean"],
            row["Iters_std"],
            ".0f",
        ),
        axis=1,
    )

    summary["Final Viol"] = summary.apply(
        lambda row: (
            f"{row['Viol_mean']:.2e} "
            f"± {row['Viol_std']:.1e}"
        ),
        axis=1,
    )

    summary["PSD Projs"] = summary.apply(
        lambda row: safe_format(
            row["PSD_Projs_mean"],
            row["PSD_Projs_std"],
            ".0f",
        ),
        axis=1,
    )

    summary["Affine Projs"] = summary.apply(
        lambda row: safe_format(
            row["Affine_Projs_mean"],
            row["Affine_Projs_std"],
            ".0f",
        ),
        axis=1,
    )

    summary["Converged"] = summary["Convergence_rate"].apply(
        lambda value: f"{100.0 * value:.0f}%"
    )

    display_columns = [
        "Method",
        "Time (s)",
        "Iterations",
        "Final Viol",
        "PSD Projs",
        "Affine Projs",
        "Converged",
    ]

    print("\n" + "=" * 140)
    print(
        "RESULTS: PSD MATRIX COMPLETION "
        f"(n={n_matrix}, rank={rank}, "
        f"{100 * observation_probability:.0f}% observed, "
        f"{num_instances} independent problems)"
    )
    print("=" * 140)

    print(
        summary[display_columns].to_string(
            index=False
        )
    )

    # Align the two methods by their common instance IDs.
    paired_iterations = df.pivot(
        index="Instance", columns="Method", values="Iterations"
    )
    paired_times = df.pivot(
        index="Instance", columns="Method", values="Time"
    ).reindex(paired_iterations.index)
    deep_method = r"ecCRM, Deep (T=P_Y P_X P_Y), $\alpha_k=0.5$"
    print_paired_statistics(
        "Deep ecCRM vs cCRM",
        paired_iterations["cCRM"].to_numpy(),
        paired_iterations[deep_method].to_numpy(),
        paired_times["cCRM"].to_numpy(),
        paired_times[deep_method].to_numpy(),
    )


    print(
        "\nObserved-entry fraction: "
        f"{np.mean(observed_fractions):.4f} "
        f"± {np.std(observed_fractions):.4f}"
    )

    print(
        "Projection counts and runtimes exclude evaluations "
        "performed solely for stopping or final-residual tests."
    )

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Last-value extension within each method: a terminated run keeps its
    # final violation until the longest run of that method has terminated.
    # Means and standard deviations therefore use all instances at every k.
    deep_name = r"ecCRM, Deep (T=P_Y P_X P_Y), $\alpha_k=0.5$"
    display_labels = {
        "Cimmino": "Cimmino",
        "MAP": "MAP",
        "cCRM": "cCRM",
        deep_name: r"Deep ecCRM ($\alpha=1/2$)",
    }
    colors = {
        "Cimmino": "tab:blue",
        "MAP": "tab:orange",
        "cCRM": "tab:green",
        deep_name: "tab:red",
    }

    fig, (ax_mean, ax_spread) = plt.subplots(
        2, 1,
        figsize=(7.2, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    max_relative_sd = 0.0

    for method_name in method_names:
        violation_histories = trajectories[method_name]
        max_length = max(len(history) for history in violation_histories)
        padded_histories = np.array([
            history + [history[-1]] * (max_length - len(history))
            for history in violation_histories
        ])

        mean_violation = np.mean(padded_histories, axis=0)
        std_violation = np.std(padded_histories, axis=0)
        iteration_axis = np.arange(1, max_length + 1)
        color = colors[method_name]
        linestyle = "--" if method_name == deep_name else "-"

        ax_mean.semilogy(
            iteration_axis, mean_violation,
            color=color, linestyle=linestyle, linewidth=1.8,
            label=display_labels[method_name],
        )
        ax_mean.fill_between(
            iteration_axis,
            np.clip(mean_violation - std_violation, 1e-15, None),
            mean_violation + std_violation,
            color=color, alpha=0.20, linewidth=0,
        )

        if method_name in ("cCRM", deep_name):
            relative_sd = np.divide(
                100.0 * std_violation,
                mean_violation,
                out=np.zeros_like(std_violation),
                where=mean_violation > 0,
            )
            max_relative_sd = max(max_relative_sd, float(np.max(relative_sd)))
            ax_spread.plot(
                iteration_axis, relative_sd,
                color=color, linestyle=linestyle, linewidth=1.8,
                label=display_labels[method_name],
            )

    ax_mean.set_title(
        f"PSD matrix completion (mean over {num_instances} instances)"
    )
    ax_mean.set_ylabel("Feasibility violation")
    ax_mean.legend(loc="upper right", framealpha=0.95)
    ax_spread.set_xlabel("Iteration")
    ax_spread.set_ylabel("Relative SD (%)")
    ax_spread.set_ylim(0, max(1.0, 1.4 * max_relative_sd))
    ax_spread.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.95)

    for ax in (ax_mean, ax_spread):
        ax.grid(True, which="major", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.align_ylabels()
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.93)
    fig.savefig(output_dir / "fig2.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output_dir / "fig2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    from run_experiments import main
    main(["--benchmark", "matrix_completion", *sys.argv[1:]])
