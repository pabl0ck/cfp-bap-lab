# Projection Methods Benchmarks

Experiments with projection methods for **convex feasibility** and the **Best Approximation Problem (BAP)**, including the extended centralized Circumcentered Reflection Method (ecCRM) and related classical algorithms.

This repository contains the numerical experiments accompanying:

* **P. Barros**, *Deep Centralization for the Circumcentered Reflection Method*, to appear, 2025.
* **R. Behling, Y. Bello-Cruz, A. N. Iusem, L. R. Santos**, *On the Centralization of the Circumcentered Reflection Method*, Mathematical Programming, 205:337–371, 2024.

### Current Focus

* **PSD-cone feasibility** via rank-deficient matrix completion to showcase the computational efficiency of **deep kernels**.
* High-dimensional **ellipsoid intersections** to validate the superlinear acceleration of **vanishing step sizes**.

*(Future extensions will cover additional geometries and methods such as Halpern and Dykstra).*

---

## 1. Overview

This repository provides small, self-contained numerical experiments for projection-based algorithms in convex analysis. It emphasizes:

* Two-set and multi-set convex feasibility problems.
* Comparing different projection pipelines and centralization schemes.

The core of this repository is the implementation of the **extended centralized Circumcentered Reflection Method (ecCRM)**. In ecCRM, the "centralization" step is driven by an admissible operator $T$ and a relaxation parameter $\alpha_k$. This modularity decouples the **geometric engine** (PCRM) from the design of the **projection kernel** and the **step-size schedule**.

---

## 2. Implemented Experiments

### 2.1 PSD Matrix Completion (Benefit of Deep Kernels)

We consider the positive semidefinite (PSD) matrix completion problem:

> Find $Z \in \mathcal{S}_+^n$ such that $Z_{ij} = A_{ij}$ for $(i,j) \in \Omega$

where $\mathcal{S}_+^n$ is the PSD cone and $\Omega$ encodes the observed entries (an affine subspace). Because this geometry is rank-deficient and lacks an interior, superlinear convergence is theoretically impossible. This makes it an ideal benchmark for comparing severe sublinear stagnation.

We implemented ecCRM alongside classical baselines (Alternating Projections and Cimmino) using different kernel operators:

* `T = P_Y P_X` (**Standard / cCRM**)
* `T = P_Y P_X P_Y` (**Deep / modular ecCRM**)

**Key Results (Averaged over 50 instances, $100 \times 100$ matrices, 60% observed):**

* Classical methods like MAP and Cimmino fail to converge within a 2000-iteration limit.
* Even though the Deep kernel uses more projections per iteration, it only adds cheap memory operations (affine projections), while the $O(n^3)$ eigenvalue decompositions remain exactly two per iteration.
* At a fixed step size of $\alpha = 0.5$, the Deep kernel achieves a **37.1% reduction in iterations** and a **36.8% reduction in overall runtime** compared to standard cCRM.

This demonstrates that the choice of kernel $T$ is not merely a cosmetic modeling detail - it fundamentally alters the linear rate constant and structural performance of the algorithm.

### 2.2 High-Dimensional Ellipsoids (Acceleration via Vanishing Steps)

To validate the superlinear convergence theory of ecCRM, we evaluate the intersection of two nearly tangent, anisotropic ellipsoids in $\mathbb{R}^{2000}$. Because the intersection has a nonempty interior, the methods are guaranteed to converge superlinearly.

We compared a fixed-step approach against dynamic, vanishing step-size schedules ($\alpha_k \to 0$):

* Fixed step: `α_k = 0.5` (Standard cCRM)
* Sublinear decay: `α_k = 1/(k+2)`
* Geometric decay: `α_k = 0.9^{k+1}`

**Key Results (Averaged over 50 runs, tolerance $10^{-12}$):**

* While sublinear schedules consistently reduce iterations by roughly 19% over the standard fixed-step baseline, aggressive geometric decay drastically accelerates convergence.
* The geometric schedule `α_k = 0.9^{k+1}` yields a **60.2% reduction in iterations** and a **57.5% reduction in total runtime**, entirely dwarfing classical MAP and Cimmino baselines.

These tests confirm that **both** parameters of ecCRM - the kernel $T$ and the sequence $\alpha_k$ - play essential and complementary roles in achieving optimal algorithmic performance.

---

## 3. Repository Structure

* `matrix_completion.py`: Generates the PSD matrix completion benchmark, running ecCRM with shallow/deep kernels alongside MAP and Cimmino.
* `ellipsoids.py`: Generates the high-dimensional ellipsoid intersection benchmark to test fixed vs. vanishing step-size schedules.

---

## 4. Quickstart & Requirements

**Minimal dependencies:**

* Python $\ge$ 3.9
* `numpy`
* `pandas`
* `matplotlib` (for generating convergence plots and descent trajectories)
* `scipy` (recommended for numerical linear algebra optimizations)

**Installation:**

```bash
git clone https://github.com/pabl0ck/cfp-bap-lab.git
cd cfp-bap-lab
pip install numpy pandas scipy matplotlib

```

**Running the experiments:**

```bash
# Run the Deep Kernel matrix completion benchmark
python matrix_completion.py

# Run the vanishing step-size ellipsoid benchmark
python ellipsoids.py

```
