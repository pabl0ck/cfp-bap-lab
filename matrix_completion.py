import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
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


def run_feasibility_experiment(
    num_instances: int = 50,
) -> None:
    n_matrix = 120
    rank = 10
    observation_probability = 0.75

    max_iter = 1500
    tol = 1e-2
    alpha = 0.5

    P_X = PSDConeProjector(n_matrix)

    method_names = [
        "Cimmino",
        "MAP",
        "cCRM",
        r"ecCRM, Deep (T=P_Y P_X P_Y), $\alpha_k=0.5$",
    ]

    results = []
    trajectories = {
        method_name: []
        for method_name in method_names
    }

    observed_fractions = []

    print(
        f"Running {num_instances} independent "
        "PSD matrix-completion problems..."
    )

    for instance in range(num_instances):
        rng = np.random.default_rng(42 + instance)

        P_Y, z0, observed_fraction = generate_matrix_completion_problem(
            n=n_matrix,
            rank=rank,
            observation_probability=observation_probability,
            rng=rng,
        )

        observed_fractions.append(observed_fraction)

        configs = make_configs(P_X, P_Y)

        for cfg in configs:
            if cfg["m_type"] == "ap":
                traj, viol_hist, elapsed_time = ap_iterates(
                    z0=z0,
                    P_X=P_X,
                    P_Y=P_Y,
                    max_iter=max_iter,
                    tol=tol,
                )

            else:
                start_time = time.perf_counter()

                if cfg["m_type"] == "cimmino":
                    traj, viol_hist = cimmino_iterates(
                        z0=z0,
                        P_X=P_X,
                        P_Y=P_Y,
                        max_iter=max_iter,
                        tol=tol,
                    )

                else:
                    traj, viol_hist = eccrm_iterates(
                        z0=z0,
                        P_X=P_X,
                        P_Y=P_Y,
                        T=cfg["T"],
                        alpha=alpha,
                        max_iter=max_iter,
                        tol=tol,
                    )

                elapsed_time = time.perf_counter() - start_time

            z_final = traj[-1]

            # External final evaluation. It is outside the timed region and is
            # not included in the projection counts.
            final_viol = max(
                np.linalg.norm(z_final - P_X(z_final)),
                np.linalg.norm(z_final - P_Y(z_final)),
            )

            iterations = len(viol_hist)

            results.append(
                {
                    "Instance": instance,
                    "Method": cfg["name"],
                    "Time": elapsed_time,
                    "Iterations": float(iterations),
                    "Viol": final_viol,
                    "Converged": final_viol <= tol,
                    "PSD_Projs": float(
                        iterations * cfg["psd_per_iter"]
                    ),
                    "Affine_Projs": float(
                        iterations * cfg["affine_per_iter"]
                    ),
                }
            )

            trajectories[cfg["name"]].append(viol_hist)

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

    print(
        "\nObserved-entry fraction: "
        f"{np.mean(observed_fractions):.4f} "
        f"± {np.std(observed_fractions):.4f}"
    )

    print(
        "Projection counts and runtimes exclude evaluations "
        "performed solely for stopping or final-residual tests."
    )

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    plt.figure(figsize=(8, 6))

    for method_name in method_names:
        violation_histories = trajectories[method_name]

        max_length = max(
            len(history)
            for history in violation_histories
        )

        padded_histories = np.array(
            [
                history
                + [history[-1]] * (max_length - len(history))
                for history in violation_histories
            ]
        )

        mean_violation = np.mean(
            padded_histories,
            axis=0,
        )
        std_violation = np.std(
            padded_histories,
            axis=0,
        )

        iteration_axis = np.arange(
            1,
            max_length + 1,
        )

        line = plt.semilogy(
            iteration_axis,
            mean_violation,
            linewidth=1.5,
            label=method_name,
        )[0]

        plt.fill_between(
            iteration_axis,
            np.clip(
                mean_violation - std_violation,
                1e-15,
                None,
            ),
            mean_violation + std_violation,
            color=line.get_color(),
            alpha=0.15,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Violation")
    plt.title(
        "PSD Matrix Completion "
        f"(Average over {num_instances} problems)"
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.3,
    )

    plt.legend(
        fontsize=9,
        borderpad=1.0,
        labelspacing=0.8,
        handlelength=2.0,
    )

    plt.tight_layout()

    plt.savefig(
        "matrixcompletion.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    run_feasibility_experiment(
        num_instances=50
    )
