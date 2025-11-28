import time
import numpy as np
import pandas as pd
from typing import Callable, Sequence, List

# ==========================================
# 1. SOLVER ENGINE
# ==========================================
Vector = np.ndarray
Operator = Callable[[Vector], Vector]

def circumcenter(z: Vector, v: Vector, w: Vector, tol: float = 1e-12) -> Vector:
    z, v, w = np.asarray(z), np.asarray(v), np.asarray(w)
    u1, u2 = v - z, w - z
    dot11, dot22, dot12 = np.dot(u1, u1), np.dot(u2, u2), np.dot(u1, u2)
    det = dot11 * dot22 - dot12**2
    
    if abs(det) <= tol * (dot11 + dot22 + tol):
        # Fallback
        d_vw = np.dot(v - w, v - w)
        if dot11 >= dot22 and dot11 >= d_vw: return 0.5 * (z + v)
        if dot22 >= dot11 and dot22 >= d_vw: return 0.5 * (z + w)
        return 0.5 * (v + w)

    inv_det = 1.0 / det
    rhs1, rhs2 = 0.5 * dot11, 0.5 * dot22
    lam1 = inv_det * (dot22 * rhs1 - dot12 * rhs2)
    lam2 = inv_det * (dot11 * rhs2 - dot12 * rhs1)
    return z + lam1 * u1 + lam2 * u2

def eccrm_iterates(z0: Vector, P_X: Operator, P_Y: Operator, T: Operator, 
                   alphas: Sequence[float], max_iter: int, tol: float) -> List[Vector]:
    z = np.asarray(z0, dtype=float)
    traj = [z.copy()]
    
    for k in range(max_iter):
        alpha = alphas[k] if k < len(alphas) else alphas[-1]
        
        t_z = T(z)
        px_tz = P_X(t_z)
        y = alpha * t_z + (1.0 - alpha) * px_tz
        p_x_y = px_tz
        p_y_y = P_Y(y)
        
        viol = max(np.linalg.norm(y - p_x_y), np.linalg.norm(y - p_y_y))
        if viol < tol:
            traj.append(y.copy())
            break
            
        r_x = 2.0 * p_x_y - y
        r_y = 2.0 * p_y_y - y
        z = circumcenter(y, r_x, r_y)
        traj.append(z.copy())
        
    return traj

# ==========================================
# 2. PROJECTORS
# ==========================================

class PSDConeProjector:
    def __init__(self, n: int):
        self.n = n
    
    def __call__(self, z_vec: np.ndarray) -> np.ndarray:
        Z = z_vec.reshape((self.n, self.n))
        Z_sym = 0.5 * (Z + Z.T)
        w, V = np.linalg.eigh(Z_sym)
        w_plus = np.maximum(w, 0.0)
        X = V @ (w_plus[:, None] * V.T)
        return X.ravel()

class AffineProjector:
    def __init__(self, n: int, indices: tuple, values: np.ndarray):
        self.n = n
        self.idx = indices
        self.vals = values
        
    def __call__(self, z_vec: np.ndarray) -> np.ndarray:
        z_out = z_vec.copy()
        Z_out = z_out.reshape((self.n, self.n))
        Z_out[self.idx] = self.vals
        return z_out

# ==========================================
# 3. EXPERIMENT: FEASIBILITY
# ==========================================

# PSDConeProjector(n): projector onto n×n PSD cone, acting on vec'd matrices
# AffineProjector(n, indices, values): projector onto observed entries constraints
# eccrm_iterates(z0, P_X, P_Y, T, alpha_list, max_iter, tol): returns trajectory list


def run_feasibility_experiment():
    # ------------------------------------------------------------
    # PROBLEM SETUP: single matrix completion instance (fixed)
    # ------------------------------------------------------------
    n_matrix = 100
    n_vec = n_matrix * n_matrix

    rng_problem = np.random.default_rng(42)

    rank = 5
    L = rng_problem.standard_normal((n_matrix, rank))
    X_star = L @ L.T

    mask = rng_problem.random((n_matrix, n_matrix)) < 0.4
    indices = np.where(mask)
    values = X_star[indices]

    P_X = PSDConeProjector(n_matrix)
    P_Y = AffineProjector(n_matrix, indices, values)

    # ------------------------------------------------------------
    # METHODS / OPERATORS
    # ------------------------------------------------------------
    configs = [
        {
            "name": "Cheap (T=P_Y)",
            "T": lambda z, py=P_Y: py(z),
            "projs_per_iter": 3,
        },
        {
            "name": "Standard (T=P_Y P_X)",
            "T": lambda z, px=P_X, py=P_Y: py(px(z)),
            "projs_per_iter": 4,
        },
        {
            "name": "Deep (T=P_Y P_X P_Y)",
            "T": lambda z, px=P_X, py=P_Y: py(px(py(z))),
            "projs_per_iter": 5,
        },
    ]

    # Stepsizes and seeds (same matrix, different starting points)
    alpha_values = [0.25, 0.5, 0.75]
    seeds = range(10)

    results = []

    # ------------------------------------------------------------
    # MAIN LOOP: same matrix, multiple seeds, methods, alphas
    # ------------------------------------------------------------
    for seed in seeds:
        rng = np.random.default_rng(seed)
        z0 = rng.standard_normal(n_vec) * 10.0  # start far away

        for cfg in configs:
            T_func = cfg["T"]

            for alpha in alpha_values:
                start_t = time.perf_counter()

                traj = eccrm_iterates(
                    z0, P_X, P_Y, T_func,
                    [alpha] * 5000,
                    max_iter=5000,
                    tol=1e-2,
                )

                end_t = time.perf_counter()

                z_f = traj[-1]
                viol = max(
                    np.linalg.norm(z_f - P_X(z_f)),
                    np.linalg.norm(z_f - P_Y(z_f)),
                )

                iters = len(traj) - 1
                time_elapsed = end_t - start_t
                projs_approx = iters * cfg["projs_per_iter"]

                results.append({
                    "Method": cfg["name"],
                    "Alpha": float(alpha),
                    "Time": time_elapsed,
                    "Iterations": float(iters),
                    "Viol": viol,
                    "Projections_Approx": float(projs_approx),
                })

    df = pd.DataFrame(results)

    # ------------------------------------------------------------
    # AGGREGATED SUMMARY (converged first, then non-converged)
    # ------------------------------------------------------------
    summary = (
        df.groupby(["Method", "Alpha"])
          .agg(
              Time=("Time", "mean"),
              Iterations=("Iterations", "mean"),
              Viol=("Viol", "mean"),
              Projections_Approx=("Projections_Approx", "mean"),
          )
          .reset_index()
    )

    # Convergence flag: Viol <= 0.01
    summary["ConvergedFlag"] = summary["Viol"] <= 0.01

    # Sort: converged (True) first, then by Time; non-converged at the bottom
    summary = summary.sort_values(
        ["ConvergedFlag", "Time"],
        ascending=[False, True],
    ).drop(columns=["ConvergedFlag"])

    print("\n" + "=" * 70)
    print(f"RESULTS: PSD FEASIBILITY (N={n_matrix}x{n_matrix}, same matrix, 10 seeds)")
    print("=" * 70)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x: .7f}",
        )
    )

    # ------------------------------------------------------------
    # DETAILED STATS + RELATIVE SPEEDUP FOR ALPHA = 0.5
    # (same converged-first ordering)
    # ------------------------------------------------------------
    alpha0 = 0.5
    summary_a0 = summary[np.isclose(summary["Alpha"], alpha0)].copy()

    if not summary_a0.empty:
        # Baseline: Standard cCRM, regardless of convergence flag
        base_time = float(
            summary_a0.loc[
                summary_a0["Method"].str.contains("Standard"),
                "Time",
            ].iloc[0]
        )

        summary_a0["Speedup_vs_Standard"] = base_time / summary_a0["Time"]

        # Re-apply converged-first ordering within alpha=0.5
        summary_a0["ConvergedFlag"] = summary_a0["Viol"] <= 0.01
        summary_a0 = summary_a0.sort_values(
            ["ConvergedFlag", "Time"],
            ascending=[False, True],
        ).drop(columns=["ConvergedFlag"])

        print("\nDetailed stats for alpha = 0.5 (converged first):")
        print(
            summary_a0.to_string(
                index=False,
                float_format=lambda x: f"{x: .7f}",
            )
        )
    else:
        print("\nNo configurations for alpha = 0.5 found in summary.")


if __name__ == "__main__":
    run_feasibility_experiment()
