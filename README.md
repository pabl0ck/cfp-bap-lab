# Projection Methods Benchmarks

Experiments with projection methods for **convex feasibility** and the **Best Approximation Problem (BAP)**, including the extended centralized Circumcentered Reflection Method (ecCRM) and related algorithms.

This repository contains code accompanying:

- Pablo Barros, *Extended, Modular Centralized Circumcentered Reflection Method*, to appear, 2025.  
- R. Behling, Y. Bello-Cruz, A. N. Iusem, L. R. Santos, *On the Centralization of the Circumcentered Reflection Method*, Mathematical Programming, 205:337–371, 2024.

Current focus:

- PSD-cone feasibility via **matrix completion** (rank-deficient) to showcase the usefulness of **deep kernels**.  
- High-dimensional **ellipsoid intersections** to validate the effect of **vanishing step sizes**.  

Future extensions will cover more geometries and methods (Halpern, Dykstra, etc.).

---

## 1. Overview

This repository collects small, self-contained numerical experiments for projection-based algorithms in convex analysis, with an emphasis on:

- Two-set and multi-set **convex feasibility** problems;  
- The **Best Approximation Problem (BAP)** over intersections of convex sets;  
- Comparing different **projection pipelines** and **centralization schemes**.

The main example is an implementation of the **extended centralized Circumcentered Reflection Method (ecCRM)**, where the “centralization” step is driven by an admissible operator \(T\). This decouples:

- the **geometric engine** (PCRM/cCRM), from  
- the design of the **projection kernel** \(T\) and the **step-size schedule** \((\alpha_k)\).

Code for all experiments is available here: <https://github.com/pabl0ck/cfp-bap-lab>

---

## 2. Implemented Experiments

### 2.1 PSD Matrix Completion (PSD-Cone Feasibility)

We consider the positive semidefinite (PSD) matrix completion problem

> find \(Z \in \mathcal{S}_+^n\) such that \(Z_{ij} = A_{ij}\) for \((i,j) \in \Omega\),

where \(\mathcal{S}_+^n\) is the PSD cone and \(\Omega\) encodes observed entries (an affine subspace).  
This geometry is **rank-deficient** and lacks interior, so **superlinear convergence is not expected**, but one can still compare linear rates and constants.

We implement ecCRM with several choices of kernel operator \(T\):

- `T = P_Y`         (“Cheap”)  
- `T = P_Y P_X`     (“Standard / cCRM`)  
- `T = P_Y P_X P_Y` (“Deep / modular ecCRM”)

where:

- \(P_X\) is projection onto the PSD cone,  
- \(P_Y\) is projection onto the affine constraints (observed entries).

On a benchmark with 10 random instances (e.g., \(n = 100\), rank \(r = 5\), 40% observed entries), we compare the **Standard** kernel `T = P_Y P_X` against the **Deep** kernel `T = P_Y P_X P_Y` under several fixed step sizes \(\alpha\).

Key observation:

- Even though the Deep kernel uses **more projections per iteration**, it consistently reduces **total runtime** and **iteration count**.
- At the best fixed step size \(\alpha = 0.5\), the Deep kernel achieves about **9% lower CPU time** and **9% fewer iterations** than the standard cCRM kernel.

This shows that the **choice of kernel \(T\)** is not a cosmetic modeling detail: it has a **structural impact** on performance even in geometries where superlinear convergence is impossible.

---

### 2.2 High-Dimensional Ellipsoids (Vanishing Step Sizes)

To validate the superlinear convergence theory for ecCRM with **vanishing step sizes**, we consider the intersection of two nearly tangent, anisotropic ellipsoids in \(\mathbb{R}^{2000}\) (with moderate condition numbers).

Here the intersection has **nonempty interior**, so both cCRM and ecCRM are guaranteed to converge at least **linearly**. We compare:

- cCRM with a **fixed step** `α = 0.5`, versus  
- ecCRM with a **vanishing step-size sequence** `α_k = 1/(k+2)`.

Numerical results (averaged over multiple runs) show that the **vanishing step-size schedule**:

- Reduces the **iteration count** by roughly **15%**, and  
- Reduces the **total runtime** by about **20%**,  
- While reaching the **same final feasibility violation** (on the order of \(10^{-13}\)).

This confirms that:

1. The **kernel \(T\)** (e.g., Deep vs Standard) matters for the **linear phase** and constants;  
2. The **step-size sequence \((\alpha_k)\)** is crucial to unlock the **superlinear regime** predicted by theory.

In other words, **both parameters of ecCRM — the kernel \(T\) and the sequence \((\alpha_k)\)** — play essential and complementary roles.

---

## 3. Repository Structure

- `matrix_completion.py`  
  - Script for PSD matrix completion experiments with different kernels.  
- `ellipsoids.py`  
  - Script for high-dimensional ellipsoid intersection and vanishing step-size tests.  

Refer to the individual scripts for more details.

---

## 4. Requirements

Minimal dependencies:

- Python ≥ 3.9  
- [NumPy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)

Additional (recommended) packages for some experiments:

- [SciPy](https://scipy.org/) – for numerical linear algebra and optimization in some projection routines.  
- [Matplotlib](https://matplotlib.org/) – for convergence plots and diagnostics.

Install with:

```bash
pip install numpy pandas scipy matplotlib
