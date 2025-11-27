# Projection Methods Benchmarks

Experiments with projection methods for **convex feasibility** and the **Best Approximation Problem (BAP)**, including ecCRM and related algorithms.

This repository contains code accompanying:

- Pablo Barros, *Extended, Modular Centralized Circumcentered Reflection Method*, to appear.  
- R. Behling, Y. Bello-Cruz, A. N. Iusem, L. R. Santos, *On the Centralization of the Circumcentered Reflection Method*, Mathematical Programming, 205:337–371, 2024.

Current focus: PSD-cone feasibility via matrix completion; future extensions will cover more geometries and methods (Halpern, Dykstra, etc.).

---

## 1. Overview

This repository collects small, self-contained numerical experiments for projection-based algorithms in convex analysis, with an emphasis on:

- Two-set and multi-set **convex feasibility** problems;
- The **Best Approximation Problem (BAP)** over intersections of convex sets;
- Comparing different **projection pipelines** and centralization schemes.

The main example is an implementation of the **extended centralized circumcentered reflection method (ecCRM)** on a PSD matrix-completion feasibility problem, as developed in:

> P. Barros, *Extended, Modular Centralized Circumcentered Reflection Method*, to appear.

The experiment compares several choices of the “kernel” operator \(T\), such as:

- `T = P_Y`         (“Cheap”)  
- `T = P_Y P_X`     (“Standard”)  
- `T = P_Y P_X P_Y` (“Deep”)

where \(P_X\) is the projector onto the PSD cone and \(P_Y\) is the projector onto the affine constraints (observed entries).  
The underlying centralized CRM ideas follow:

> R. Behling, Y. Bello-Cruz, A. N. Iusem, L. R. Santos, *On the Centralization of the Circumcentered Reflection Method*, Mathematical Programming, 205:337–371, 2024.

---

## 2. Requirements

Minimal dependencies:

- Python ≥ 3.9  
- [NumPy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)

Install with:

```bash
pip install numpy pandas
