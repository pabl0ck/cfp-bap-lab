import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
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

    for _ in range(max_iter):
        px = P_X(z)
        z_new = P_Y(px)

        viol = np.linalg.norm(z_new - P_X(z_new))
        viol_hist.append(viol)
        traj.append(z_new.copy())

        if viol < tol:
            break

        z = z_new

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


def run_single_instance(
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    n: int,
):
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

    T_func = lambda z, px=P_X, py=P_Y: py(px(z))

    results = []

    schedules = [
        (
            r"cCRM",
            0.5,
        ),
        (
            r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-1/2}$",
            lambda k: 1.0 / np.sqrt(k + 2),
        ),
        (
            r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-1}$",
            lambda k: 1.0 / (k + 2),
        ),
        (
            r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=(k+2)^{-2}$",
            lambda k: 1.0 / ((k + 2) ** 2),
        ),
        (
            r"ecCRM, Standard ($T=P_Y P_X$), $\alpha_k=0.9^{k+1}$",
            lambda k: 0.9 ** (k + 1),
        ),
    ]

    for method_name, alpha in schedules:
        start = time.perf_counter()

        traj, viol_hist = eccrm_iterates(
            z0,
            P_X,
            P_Y,
            T_func,
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


def run_ellipsoid_experiment(num_trials: int = 50):
    n = 3000
    max_iter = 600
    tol = 1e-12

    stats = {}

    for trial in range(num_trials):
        rng = np.random.default_rng(seed=1234 + trial)
        results = run_single_instance(rng, max_iter=max_iter, tol=tol, n=n)

        for res in results:
            key = res["Method"]

            if key not in stats:
                stats[key] = {
                    "Time": [],
                    "Iters": [],
                    "FinalViol": [],
                    "Violations": [],
                }

            stats[key]["Time"].append(res["Time"])
            stats[key]["Iters"].append(res["Iters"])
            stats[key]["FinalViol"].append(res["FinalViol"])
            stats[key]["Violations"].append(res["Violations"])

    plt.rcParams.update({
        'font.size': 14,           # Global font size
        'axes.titlesize': 16,      # Title size
        'axes.labelsize': 14,      # X and Y label size
        'xtick.labelsize': 12,     # X tick labels
        'ytick.labelsize': 12,     # Y tick labels
        'legend.fontsize': 10,     # Legend font size
    })

    plt.figure(figsize=(9, 6))

    for method in stats.keys():
        viols = [v for v in stats[method]["Violations"] if len(v) > 0]
        if not viols:
            continue

        max_len = max(len(v) for v in viols)
        padded_viols = np.array([
            v + [v[-1]] * (max_len - len(v))
            for v in viols
        ])

        mean_viol = np.mean(padded_viols, axis=0)
        std_viol = np.std(padded_viols, axis=0)
        iters = np.arange(1, max_len + 1)

        line = plt.semilogy(
            iters,
            mean_viol,
            linewidth=1.5,
            label=method,
        )[0]

        color = line.get_color()

        plt.fill_between(
            iters,
            np.clip(mean_viol - std_viol, 1e-15, None),
            mean_viol + std_viol,
            color=color,
            alpha=0.15,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Violation (max dist to X, Y)")
    plt.title(
        f"ecCRM, MAP, and Cimmino on Ellipsoids "
        f"(Avg. over {num_trials} runs)"
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(
    fontsize=9,           # Increases the text size
    borderpad=1.0,         # Adds more space inside the box edges
    labelspacing=0.8,      # Adds more vertical space between the method names
    handlelength=2.0       # Makes the colored line samples longer
    )
    plt.tight_layout()
    plt.show()

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


if __name__ == "__main__":
    run_ellipsoid_experiment(num_trials=50)
