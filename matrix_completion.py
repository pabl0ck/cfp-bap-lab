import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List, Union, Tuple

Vector = np.ndarray
Operator = Callable[[Vector], Vector]
AlphaSchedule = Union[float, Callable[[int], float]]

def safe_format(mean, std, fmt=".0f"):
    if pd.isna(std):
        return f"{mean:{fmt}} ± NaN"
    return f"{mean:{fmt}} ± {std:{fmt}}"

def circumcenter(z: Vector, v: Vector, w: Vector, tol: float = 1e-12) -> Vector:
    z, v, w = np.asarray(z), np.asarray(v), np.asarray(w)
    u1, u2 = v - z, w - z
    dot11, dot22, dot12 = np.dot(u1, u1), np.dot(u2, u2), np.dot(u1, u2)
    det = dot11 * dot22 - dot12**2

    if abs(det) <= tol * (dot11 + dot22 + tol):
        # Fallback
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
    viol_hist = []

    for k in range(max_iter):
        if callable(alpha):
            alpha_k = float(alpha(k))
        else:
            alpha_k = float(alpha)

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

# --- Alternating Projections (MAP) ---
def ap_iterates(
    z0: Vector,
    P_X: Operator,
    P_Y: Operator,
    max_iter: int,
    tol: float,
) -> Tuple[List[Vector], List[float]]:
    z = np.asarray(z0, dtype=float)
    traj = [z.copy()]
    viol_hist = []

    for _ in range(max_iter):
        px = P_X(z)
        z_new = P_Y(px)
        
        # Standardized residual: distance to the PSD cone
        viol = np.linalg.norm(z_new - P_X(z_new))
        viol_hist.append(viol)
        
        if viol < tol:
            traj.append(z_new.copy())
            break
            
        z = z_new
        traj.append(z.copy())

    return traj, viol_hist

# --- Cimmino (Simultaneous Projections) ---
def cimmino_iterates(
    z0: Vector,
    P_X: Operator,
    P_Y: Operator,
    max_iter: int,
    tol: float,
) -> Tuple[List[Vector], List[float]]:
    z = np.asarray(z0, dtype=float)
    traj = [z.copy()]
    viol_hist = []

    for _ in range(max_iter):
        px = P_X(z)
        py = P_Y(z)
        z_new = 0.5 * px + 0.5 * py
        
        # Free residual: max distance to the sets from current iterate
        viol = max(np.linalg.norm(z - px), np.linalg.norm(z - py))
        viol_hist.append(viol)
        
        if viol < tol:
            traj.append(z_new.copy())
            break
            
        z = z_new
        traj.append(z.copy())

    return traj, viol_hist

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

def run_feasibility_experiment(n_seeds):
    n_matrix = 100
    n_vec = n_matrix * n_matrix
    num_seeds = n_seeds

    rng_problem = np.random.default_rng(42)

    rank = 10
    L = rng_problem.standard_normal((n_matrix, rank))
    X_star = L @ L.T

    mask = rng_problem.random((n_matrix, n_matrix)) < 0.6
    mask = mask | mask.T  # Force symmetric observations
    indices = np.where(mask)
    values = X_star[indices]

    P_X = PSDConeProjector(n_matrix)
    P_Y = AffineProjector(n_matrix, indices, values)

    configs = [
        {
            "name": "Cimmino",
            "m_type": "cimmino",
            "T": None,
            "psd_per_iter": 1,
            "affine_per_iter": 1,
        },
        {
            "name": "Alternating Projections (MAP)",
            "m_type": "ap",
            "T": None,
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

    alpha_values = [0.5]
    
    results = []
    trajectories = {cfg["name"]: {a: [] for a in alpha_values} for cfg in configs}

    print(f"Running PSD Matrix Completion over {num_seeds} instances...")
    
    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)
        z0 = rng.standard_normal(n_vec) * 10.0

        for cfg in configs:
            alphas_to_run = alpha_values if cfg["m_type"] == "eccrm" else [alpha_values[0]]
            
            for alpha in alphas_to_run:
                start_t = time.perf_counter()

                if cfg["m_type"] == "ap":
                    traj, viol_hist = ap_iterates(z0, P_X, P_Y, max_iter=2000, tol=1e-2)
                elif cfg["m_type"] == "cimmino":
                    traj, viol_hist = cimmino_iterates(z0, P_X, P_Y, max_iter=2000, tol=1e-2)
                else:
                    traj, viol_hist = eccrm_iterates(
                        z0=z0, P_X=P_X, P_Y=P_Y, T=cfg["T"], alpha=alpha, max_iter=2000, tol=1e-2
                    )

                end_t = time.perf_counter()

                z_f = traj[-1]
                viol = max(
                    np.linalg.norm(z_f - P_X(z_f)),
                    np.linalg.norm(z_f - P_Y(z_f)),
                )

                iters = len(traj) - 1
                time_elapsed = end_t - start_t
                
                psd_approx = iters * cfg["psd_per_iter"]
                affine_approx = iters * cfg["affine_per_iter"]
                
                alpha_val = float(alpha) if cfg["m_type"] == "eccrm" else np.nan

                results.append({
                    "Method": cfg["name"],
                    "Alpha": alpha_val,
                    "Time": time_elapsed,
                    "Iterations": float(iters),
                    "Viol": viol,
                    "PSD_Projs": float(psd_approx),
                    "Affine_Projs": float(affine_approx),
                })
                
                trajectories[cfg["name"]][alpha].append(viol_hist)

    df = pd.DataFrame(results)

    summary = (
    df.groupby(["Method", "Alpha"], dropna=False)
      .agg(
          Time_mean=("Time", "mean"),
          Time_std=("Time", "std"),
          Iters_mean=("Iterations", "mean"),
          Iters_std=("Iterations", "std"),
          Viol_mean=("Viol", "mean"),
          PSD_Projs_mean=("PSD_Projs", "mean"),   
          PSD_Projs_std=("PSD_Projs", "std"),     
          Affine_Projs_mean=("Affine_Projs", "mean"), 
          Affine_Projs_std=("Affine_Projs", "std"),   
      )
      .reset_index()
    )

    summary["ConvergedFlag"] = summary["Viol_mean"] <= 0.01

    summary = summary.sort_values(
        ["ConvergedFlag", "Time_mean"],
        ascending=[False, True],
    ).drop(columns=["ConvergedFlag"])

    summary["Time (s)"] = summary.apply(lambda r: safe_format(r['Time_mean'], r['Time_std'], ".2f"), axis=1)
    summary["Iterations"] = summary.apply(lambda r: safe_format(r['Iters_mean'], r['Iters_std']), axis=1)
    summary["PSD_Projs"] = summary.apply(lambda r: safe_format(r['PSD_Projs_mean'], r['PSD_Projs_std']), axis=1)
    summary["Affine_Projs"] = summary.apply(lambda r: safe_format(r['Affine_Projs_mean'], r['Affine_Projs_std']), axis=1)

    display_cols = ["Method", "Alpha", "Iterations", "Time (s)", "Viol_mean", "PSD_Projs", "Affine_Projs"]
    
    print("\n" + "=" * 105)
    print(f"RESULTS: PSD FEASIBILITY (N={n_matrix}x{n_matrix}, 60% observed, {num_seeds} seeds)")
    print("=" * 105)
    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x: .4f}",
            na_rep="-"
        )
    )

    plt.rcParams.update({
        'font.size': 14,           # Global font size
        'axes.titlesize': 16,      # Title size
        'axes.labelsize': 14,      # X and Y label size
        'xtick.labelsize': 12,     # X tick labels
        'ytick.labelsize': 12,     # Y tick labels
        'legend.fontsize': 10,     # Legend font size
    })
    
    plt.figure(figsize=(8, 6))
    target_alpha = 0.5 
    
    for cfg in configs:
        method_name = cfg["name"]
        viols = trajectories[method_name][target_alpha]
        
        max_len = max(len(v) for v in viols)
        padded_viols = np.array([v + [v[-1]] * (max_len - len(v)) for v in viols])
        
        mean_viol = np.mean(padded_viols, axis=0)
        std_viol = np.std(padded_viols, axis=0)
        iters_arr = np.arange(1, max_len + 1)
        
        label_text = method_name
        
        line = plt.semilogy(
            iters_arr, 
            mean_viol, 
            linewidth=1.5, 
            label=label_text
        )[0]
        
        color = line.get_color()
        plt.fill_between(
            iters_arr,
            np.clip(mean_viol - std_viol, 1e-15, None),
            mean_viol + std_viol,
            color=color,
            alpha=0.15
        )

    plt.xlabel("Iteration")
    plt.ylabel("Violation (max dist to X, Y)")
    plt.title(f"PSD Matrix Completion: Descent Trajectories (Avg over {num_seeds} runs)")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(
    fontsize=9,           # Increases the text size
    borderpad=1.0,         # Adds more space inside the box edges
    labelspacing=0.8,      # Adds more vertical space between the method names
    handlelength=2.0       # Makes the colored line samples longer
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_feasibility_experiment(50)
