import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Union

Vector = np.ndarray
Operator = Callable[[Vector], Vector]
AlphaSchedule = Union[float, Callable[[int], float]]


def circumcenter(z: Vector, v: Vector, w: Vector, tol: float = 1e-12) -> Vector:
    z, v, w = np.asarray(z), np.asarray(v), np.asarray(w)
    u1, u2 = v - z, w - z
    dot11, dot22, dot12 = np.dot(u1, u1), np.dot(u2, u2), np.dot(u1, u2)
    det = dot11 * dot22 - dot12**2

    if abs(det) <= tol * (dot11 + dot22 + tol):
        d_vw = np.dot(v - w, v - w)
        if dot11 >= dot22 and dot11 >= d_vw:
            return 0.5 * (z + v)
        if dot22 >= dot11 and dot22 >= d_vw:
            return 0.5 * (z + w)
        return 0.5 * (v + w)

    inv_det = 1.0 / det
    rhs1, rhs2 = 0.5 * dot11, 0.5 * dot22
    lam1 = inv_det * (dot22 * rhs1 - dot12 * rhs2)
    lam2 = inv_det * (dot11 * rhs2 - dot12 * rhs1)
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
        p_x_y = px_tz
        p_y_y = P_Y(y)

        viol = max(np.linalg.norm(y - p_x_y), np.linalg.norm(y - p_y_y))
        viol_hist.append(viol)

        if viol < tol:
            traj.append(y.copy())
            break

        r_x = 2.0 * p_x_y - y
        r_y = 2.0 * p_y_y - y
        z = circumcenter(y, r_x, r_y)
        traj.append(z.copy())

    return traj, viol_hist


def map_iterates(
    z0: Vector,
    P_X: Operator,
    P_Y: Operator,
    max_iter: int,
    tol: float,
) -> Tuple[List[Vector], List[float]]:
    z = np.asarray(z0, dtype=float)
    traj = [z.copy()]
    viol_hist: List[float] = []

    if max_iter <= 0:
        return traj, viol_hist

    px = P_X(z)
    for _ in range(max_iter):
        z_new = P_Y(px)

        # The stopping-test projection is also the next MAP input to P_Y.
        px = P_X(z_new)
        viol = np.linalg.norm(z_new - px)
        viol_hist.append(viol)
        traj.append(z_new.copy())

        if viol < tol:
            break

    return traj, viol_hist


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
        z_new = 0.5 * (px + py)

        viol = max(np.linalg.norm(z - px), np.linalg.norm(z - py))
        viol_hist.append(viol)
        traj.append(z_new.copy())

        if viol < tol:
            break

        z = z_new

    return traj, viol_hist


class EllipsoidProjector:
    def __init__(self, Q: np.ndarray, c: np.ndarray):
        self.Q = 0.5 * (Q + Q.T)
        self.c = c

        d, U = np.linalg.eigh(self.Q)
        if np.any(d <= 0):
            raise ValueError("Q must be positive definite.")

        self.d = d
        self.U = U

    def _phi(self, lam: float, y0: np.ndarray) -> float:
        denom = 1.0 + lam * self.d
        y = y0 / denom
        return np.sum(self.d * y**2) - 1.0

    def __call__(self, z_in: np.ndarray) -> np.ndarray:
        z = np.asarray(z_in, dtype=float)
        v = z - self.c
        y0 = self.U.T @ v

        val = np.sum(self.d * y0**2)
        if val <= 1.0 + 1e-12:
            return z.copy()

        lam_lo = 0.0
        lam_hi = 1.0
        phi_hi = self._phi(lam_hi, y0)

        it_guard = 0
        while phi_hi > 0.0 and it_guard < 60:
            lam_hi *= 2.0
            phi_hi = self._phi(lam_hi, y0)
            it_guard += 1

        for _ in range(60):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            phi_mid = self._phi(lam_mid, y0)

            if phi_mid > 0.0:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid

        lam = 0.5 * (lam_lo + lam_hi)
        denom = 1.0 + lam * self.d
        y = y0 / denom
        return self.c + self.U @ y


def make_spd(
    n: int,
    rng: np.random.Generator,
    anisotropy: float = 10.0,
) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    lam_min = 1.0
    lam_max = anisotropy
    eigvals = lam_min + (lam_max - lam_min) * rng.random(n)

    return Q.T @ np.diag(eigvals) @ Q


def generate_ellipsoid_instance(rng: np.random.Generator, n: int):
    x_star = rng.standard_normal(n)

    Q = make_spd(n, rng, anisotropy=5.0)

    w = rng.standard_normal(n)
    v = w / np.sqrt(w @ Q @ w)

    rho = 0.95
    d = rho * v

    c1 = x_star - d
    c2 = x_star + d

    P_X = EllipsoidProjector(Q, c1)
    P_Y = EllipsoidProjector(Q, c2)

    u0 = rng.standard_normal(n)
    u0 /= np.linalg.norm(u0)
    z0 = x_star + 8.0 * u0
    return z0, P_X, P_Y


def run_single_instance(
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    n: int,
):
    z0, P_X, P_Y = generate_ellipsoid_instance(rng, n)
    T_func = lambda z, px=P_X, py=P_Y: py(px(z))
    T_deep = lambda z, px=P_X, py=P_Y: py(px(py(z)))

    results = []

    schedules = [
        (
            r"cCRM",
            0.5,
            T_func,
        ),
        (
            r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-1/2}$",
            lambda k: 1.0 / np.sqrt(k + 2),
            T_func,
        ),
        (
            r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-1}$",
            lambda k: 1.0 / (k + 2),
            T_func,
        ),
        (
            r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-2}$",
            lambda k: 1.0 / ((k + 2) ** 2),
            T_func,
        ),
        (
            r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=0.9^{k+1}$",
            lambda k: 0.9 ** (k + 1),
            T_func,
        ),
        (
            r"ecCRM, Deep ($T=P_Y P_X P_Y$), $\alpha_k=0.9^{k+1}$",
            lambda k: 0.9 ** (k + 1),
            T_deep,
        ),
    ]

    for method_name, alpha, kernel in schedules:
        start = time.perf_counter()

        traj, viol_hist = eccrm_iterates(
            z0,
            P_X,
            P_Y,
            kernel,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )

        elapsed = time.perf_counter() - start
        final_viol = viol_hist[-1] if viol_hist else np.nan

        results.append({
            "Method": method_name,
            "Time": elapsed,
            "Iters": len(viol_hist),
            "FinalViol": final_viol,
            "Violations": viol_hist,
        })

    # MAP baseline
    start = time.perf_counter()

    traj, viol_hist = map_iterates(
        z0,
        P_X,
        P_Y,
        max_iter=max_iter,
        tol=tol,
    )

    elapsed = time.perf_counter() - start
    final_viol = viol_hist[-1] if viol_hist else np.nan

    results.append({
        "Method": r"MAP",
        "Time": elapsed,
        "Iters": len(viol_hist),
        "FinalViol": final_viol,
        "Violations": viol_hist,
    })

    # Cimmino baseline
    start = time.perf_counter()

    traj, viol_hist = cimmino_iterates(
        z0,
        P_X,
        P_Y,
        max_iter=max_iter,
        tol=tol,
    )

    elapsed = time.perf_counter() - start
    final_viol = viol_hist[-1] if viol_hist else np.nan

    results.append({
        "Method": r"Cimmino",
        "Time": elapsed,
        "Iters": len(viol_hist),
        "FinalViol": final_viol,
        "Violations": viol_hist,
    })

    return results


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


def run_ellipsoid_experiment(num_trials=50):
    """Run and archive the experiment; see run_experiments.py for the CLI."""
    from run_experiments import main
    main(["--benchmark", "ellipsoids", "--instances", str(num_trials)])


def render_results(stats, num_trials, output_dir, n=3000):
    """Regenerate the original summaries and figure without rerunning solvers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    display_labels = {
        "cCRM": "cCRM",
        r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-1/2}$":
            r"ecCRM, $\alpha_k=(k+2)^{-1/2}$",
        r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-1}$":
            r"ecCRM, $\alpha_k=(k+2)^{-1}$",
        r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-2}$":
            r"ecCRM, $\alpha_k=(k+2)^{-2}$",
        r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=0.9^{k+1}$":
            r"ecCRM, $\alpha_k=0.9^{k+1}$",
        r"ecCRM, Deep ($T=P_Y P_X P_Y$), $\alpha_k=0.9^{k+1}$":
            r"Deep ecCRM, $\alpha_k=0.9^{k+1}$",
        "MAP": "MAP",
        "Cimmino": "Cimmino",
    }

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(7.2, 4.5), sharey=True,
    )
    handles = []
    labels = []
    longest_history = 1

    # Preserve the existing last-value extension within each method.
    # Every pointwise mean includes all runs of that method.
    for index, method in enumerate(stats.keys()):
        viols = [v for v in stats[method]["Violations"] if len(v) > 0]
        if not viols:
            continue

        max_len = max(len(v) for v in viols)
        longest_history = max(longest_history, max_len)
        padded_viols = np.array([
            v + [v[-1]] * (max_len - len(v))
            for v in viols
        ])
        mean_viol = np.mean(padded_viols, axis=0)
        iters = np.arange(1, max_len + 1)
        label = display_labels.get(method, method)
        is_deep = "Deep" in method
        if is_deep:
            color = "black"
        elif method == "MAP":
            color = "tab:brown"
        elif method == "Cimmino":
            color = "tab:pink"
        else:
            color = f"C{index}"
        linestyle = "--" if is_deep else "-"

        for ax in (ax_full, ax_zoom):
            line, = ax.semilogy(
                iters, mean_viol,
                color=color, linestyle=linestyle, linewidth=1.7,
                label=label,
            )
            if ax is ax_full:
                handles.append(line)
                labels.append(label)

    ax_full.set_xlim(0, longest_history)
    ax_zoom.set_xlim(0, 60)
    ax_full.set_title("Full range")
    ax_zoom.set_title(r"Detail: $0 \leq k \leq 60$")
    ax_full.set_ylabel("Feasibility violation")
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("Iteration")
        ax.grid(True, which="major", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Ellipsoids (mean over {num_trials} instances)",
        fontsize=11, y=0.98,
    )
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=3, fontsize=9, frameon=False,
        columnspacing=1.2, handlelength=2.2,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, bottom=0.34, top=0.87, wspace=0.12,
    )
    fig.savefig(output_dir / "fig3.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output_dir / "fig3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Two ellipsoids in R^{n}")
    print(f"(Averaged over {num_trials} runs)")
    print(
        f"{'Method':<75} | "
        f"{'Iters (Mean ± Std)':<20} | "
        f"{'Time (s) (Mean ± Std)':<25} | "
        f"{'Final Viol':<10}"
    )
    print("-" * 140)

    for method in stats.keys():
        iters_mean = np.mean(stats[method]["Iters"])
        iters_std = np.std(stats[method]["Iters"])

        time_mean = np.mean(stats[method]["Time"])
        time_std = np.std(stats[method]["Time"])

        viol_mean = np.mean(stats[method]["FinalViol"])
        viol_std = np.std(stats[method]["FinalViol"])

        print(
            f"{method:<75} | "
            f"{iters_mean:>6.1f} ± {iters_std:<6.1f} | "
            f"{time_mean:>8.3f} ± {time_std:<8.3f} | "
            f"{viol_mean:.2e} ± {viol_std:.1e}"
        )

    # Each list was appended in the same trial order, so index i denotes
    # the same generated instance for every method.
    for method in stats:
        if method in ("cCRM", "MAP", "Cimmino"):
            continue
        print_paired_statistics(
            f"{display_labels.get(method, method)} vs cCRM",
            stats["cCRM"]["Iters"],
            stats[method]["Iters"],
            stats["cCRM"]["Time"],
            stats[method]["Time"],
        )


if __name__ == "__main__":
    import sys
    from run_experiments import main
    main(["--benchmark", "ellipsoids", *sys.argv[1:]])
