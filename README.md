# Screening Policy Optimization under Cost Constraints

Welcome to the repository for **Screening Policy Optimization for Two Diseases**, a simulation and optimization framework for comparing **independent** and **unified** screening models under cost constraints.

This project uses **Monte Carlo simulation**, **Bayesian inference**, and **convex optimization** to derive optimal screening policies for patients characterized by risk factors \(x_1, x_2\). Policies are evaluated based on **expected survival time gains** and **screening costs**.

---

## 🚀 Overview

We model two types of screening strategies:

- **Independent Model:** Diseases are screened separately.
- **Unified Model:** Joint screening decisions are made based on both risk factors.

Each patient is represented by a 2D risk vector \(x \in [0,1]^2\), generated from a \(\text{Beta}(a, b)\) distribution. For each model, we:

1. Simulate disease events and diagnoses.
2. Estimate expected survival gains and screening costs.
3. Solve an optimization problem to find the best policy under a budget constraint.

---

## 📁 Project Structure

- `independent_model.py` – Simulates separate screening for each disease and saves survival gains & costs.
- `unified_model.py` – Simulates joint screening for both diseases.
- `optimization.py` – Solves a constrained optimization problem to derive screening policies.
- `plot_kappas.py` – Provides visualizations comparing policy effectiveness.

---

## 🔧 Code Breakdown

### independent_model.py

Simulates two diseases separately:
- Uses **Bayesian updates** for disease detection.
- Computes expected survival under policy \(a = (a_1, a_2)\) where each \(a_i \in \{0, 1\}\).
- Uses `jax.vmap` and `jax.jit` for performance.
- Saves gains/costs to `comparison2/independent_data/`.

### unified_model.py

Models disease interaction:
- A single Bayesian update step covers both diseases.
- Calculates gain/cost vectors for all \(a \in \{00, 10, 01, 11\}\).
- Saves results to `comparison4/unified_data/`.

### optimization.py

Performs policy optimization:
- **Independent**: Separately optimizes for each disease (2 policies \(a_i \in \{0,1\}\)).
- **Unified**: Solves a **linear program** over 4 actions (00–11).
- Constraints:
  - Total expected cost \(\leq B\)
  - Sum of action probabilities per \(x\) = 1
- Saves policies to `saved_policies/`.

### plot_kappas.py

Generates insightful plots:
- Survival vs. cost trade-offs.
- Decision boundaries \(\arg\max_a \kappa(a) = \text{gain} - \lambda \cdot \text{cost}\).
- Delta-plots: \(\Delta\text{Gain}, \Delta\text{Cost}, \Delta\kappa\).

---

## 📊 Results and Interpretation

- We visualize optimal actions over \(x_1, x_2\) space.
- We compare the **joint policy** vs. the **combined independent policy**.
- Useful for understanding trade-offs in medical screening protocols under limited resources.

---

## 💾 How to Run

1. Install dependencies:
```bash
pip install jax jaxlib cvxpy matplotlib scipy
```
2. Run simulations:
```bash
python unified_model.py
python independent_model.py
```
3. Run optimization:
```bash
python optimization.py
```
4. Plot visualizations:
```bash
python plot_kappas.py
```

---

## 📌 Future Work

- Add **temporal decision models** (e.g. MDPs)
- Expand to \(k > 2\) diseases
- Incorporate patient-specific cost models

---

## 🧠 Authors & Acknowledgments

Developed as part of a study on optimal health screening policy design using machine learning and convex optimization techniques.

Feel free to contribute, star ⭐, and fork 🍴!
