  import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Union

# ==========================================
# 1. MATH & SOLVER ENGINE (Generic)
# ==========================================
Vector = np.ndarray
Operator = Callable[[Vector], Vector]
AlphaSchedule = Union[float, Callable[[int], float]]

def circumcenter(z: Vector, v: Vector, w: Vector, tol: float = 1e-12) -> Vector:
    z, v, w = np.asarray(z), np.asarray(v), np.asarray(w)
    u1, u2 = v - z, w - z
    dot11, dot22, dot12 = np.dot(u1, u1), np.dot(u2, u2), np.dot(u1, u2)
    det = dot11 * dot22 - dot12**2

    if abs(det) <= tol * (dot11 + dot22 + tol):
        # Fallback: choose the "most separated" pair
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
    tol: float
) -> Tuple[List[Vector], List[float]]:
    z = np.asarray(z0, dtype=float)
    traj = [z.copy()]
    viol_hist: List[float] = []

    for k in range(max_iter):
        # Get alpha_k
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

# ==========================================
# 2. ANALYTIC ELLIPSOID PROJECTOR
# ==========================================

class EllipsoidProjector:
    """
    Projector onto the ellipsoid:
        E = { x | (x - c)^T Q (x - c) <= 1 },
    where Q is symmetric positive definite.

    Uses eigen-decomposition and a 1D bisection on the KKT parameter.
    No external solver needed.
    """
    def __init__(self, Q: np.ndarray, c: np.ndarray):
        self.Q = 0.5 * (Q + Q.T)  # ensure symmetry
        self.c = c
        # Q = U^T diag(d) U with d_i > 0
        d, U = np.linalg.eigh(self.Q)
        if np.any(d <= 0):
            raise ValueError("Q must be positive definite.")
        self.d = d
        self.U = U

    def _phi(self, lam: float, y0: np.ndarray) -> float:
        """
        Constraint value g(y(lam)) - 1, where
        y_i(lam) = y0_i / (1 + lam * d_i)
        and g(y) = sum d_i y_i^2.
        """
        denom = 1.0 + lam * self.d
        y = y0 / denom
        return np.sum(self.d * y**2) - 1.0

    def __call__(self, z_in: np.ndarray) -> np.ndarray:
        z = np.asarray(z_in, dtype=float)
        v = z - self.c
        # Rotate to eigenbasis
        y0 = self.U.T @ v

        # Check if inside ellipsoid
        val = np.sum(self.d * y0**2)
        if val <= 1.0 + 1e-12:
            return z.copy()

        # Outside: solve for lam > 0 with phi(lam) = 0 via bisection
        lam_lo = 0.0
        lam_hi = 1.0
        phi_hi = self._phi(lam_hi, y0)
        # Increase lam_hi until phi(lam_hi) < 0
        it_guard = 0
        while phi_hi > 0.0 and it_guard < 60:
            lam_hi *= 2.0
            phi_hi = self._phi(lam_hi, y0)
            it_guard += 1

        # Bisection
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
        x = self.c + self.U @ y
        return x

# ==========================================
# 3. EXPERIMENT: INTERSECTION OF TWO ELLIPSOIDS
# ==========================================

def make_spd(n: int, rng: np.random.Generator, anisotropy: float = 10.0) -> np.ndarray:
    """
    Generate a random SPD matrix with controlled anisotropy and random orientation.
    - anisotropy: roughly the ratio between largest and smallest eigenvalues.
    """
    # Random orthogonal matrix (eigenvectors)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # Random eigenvalues in [lam_min, lam_max]
    lam_min = 1.0
    lam_max = anisotropy
    eigvals = lam_min + (lam_max - lam_min) * rng.random(n)

    return Q.T @ np.diag(eigvals) @ Q

def run_single_instance(rng: np.random.Generator, max_iter: int, tol: float):
    """
    Build one random ellipsoid pair + z0, run both alpha regimes,
    return list of result dicts (including violations for plotting).
    """
    n = 1000

    # 1. Build two "normal size" ellipsoids with small, nearly tangent intersection
    x_star = rng.standard_normal(n)

    # One SPD with some anisotropy – same for both ellipsoids
    Q = make_spd(n, rng, anisotropy=20.0)

    # Choose direction v with Q-norm 1: v^T Q v = 1
    w = rng.standard_normal(n)
    v = w / np.sqrt(w @ Q @ w)

    # Make x_star *almost* on the boundary of both ellipsoids
    rho = 0.99  # closer to 1 => smaller intersection
    d = rho * v

    # Centers on opposite sides of x_star along v
    c1 = x_star - d
    c2 = x_star + d

    P_X = EllipsoidProjector(Q, c1)
    P_Y = EllipsoidProjector(Q, c2)

    # 2. Initial point far away
    z0 = rng.standard_normal(n)
    z0 = z0 * (10.0 / np.linalg.norm(z0))

    # 3. Kernel + alpha configurations
    configs = [
        {"name": "Standard (T = P_Y P_X)", "T": lambda z, px=P_X, py=P_Y: py(px(z))},
    ]

    alpha_const = 0.5
    alpha_schedule = lambda k: 1.0 / (k + 2)  # k = 0,1,2,...

    results = []

    for cfg in configs:
        for alpha, alpha_name in [
            (alpha_const, "alpha = 1/2"),
            (alpha_schedule, "alpha_k = 1/(k+2)"),
        ]:
            start = time.perf_counter()
            traj, viol_hist = eccrm_iterates(
                z0, P_X, P_Y, cfg["T"],
                alpha=alpha, max_iter=max_iter, tol=tol
            )
            elapsed = time.perf_counter() - start
            final_viol = viol_hist[-1] if viol_hist else np.nan
            method_label = f"{cfg['name']} ({alpha_name})"

            results.append({
                "Method": method_label,
                "Time": elapsed,
                "Iters": len(viol_hist),
                "FinalViol": final_viol,
                "Violations": viol_hist,
            })

    return results

def run_ellipsoid_experiment(num_trials: int = 10):
    max_iter = 2000
    tol = 1e-12

    # To accumulate stats
    stats = {}
    last_results = None  # for plotting a typical instance

    for trial in range(num_trials):
        rng = np.random.default_rng(seed=1234 + trial)  # reproducible but varying
        results = run_single_instance(rng, max_iter=max_iter, tol=tol)
        last_results = results  # keep the last one

        plt.figure(figsize=(6, 4))
        for res in last_results:
            v = np.array(res["Violations"])
            if len(v) == 0:
                continue
            iters = np.arange(1, len(v) + 1)
            plt.semilogy(iters, v, marker='o', markersize=1, linewidth=1, label=res["Method"])

        plt.xlabel("Iteration")
        plt.ylabel("Violation (max dist to X, Y)")
        plt.title("ecCRM on Two Ellipsoids (one instance)")
        plt.grid(True, which="both", linestyle="--", alpha=0.1)
        plt.legend()
        plt.tight_layout()
        plt.show()

        for res in results:
            key = res["Method"]
            if key not in stats:
                stats[key] = {"Time": [], "Iters": [], "FinalViol": []}
            stats[key]["Time"].append(res["Time"])
            stats[key]["Iters"].append(res["Iters"])
            stats[key]["FinalViol"].append(res["FinalViol"])

    # Print averaged table
    print(f"Two ellipsoids in R^{2000} (normal size, small, nearly tangent intersection)")
    print(f"(Averaged over {num_trials} runs)")
    print(f"{'Method':<45} | {'Iters':<6} | {'Time(s)':<8} | {'Final Viol':<10}")
    print("-" * 85)

    # Fixed order for nice display
    method_order = [
        "Standard (T = P_Y P_X) (alpha = 1/2)",
        "Standard (T = P_Y P_X) (alpha_k = 1/(k+2))",
    ]

    for method in method_order:
        if method not in stats:
            continue
        iters_mean = np.mean(stats[method]["Iters"])
        time_mean = np.mean(stats[method]["Time"])
        viol_mean = np.mean(stats[method]["FinalViol"])

        print(f"{method:<45} | {int(round(iters_mean)): <6} | {time_mean:<8.4f} | {viol_mean:.2e}")

if __name__ == "__main__":
    run_ellipsoid_experiment(num_trials=10)
