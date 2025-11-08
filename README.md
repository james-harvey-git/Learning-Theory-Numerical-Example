# Learning-Theory-Numerical-Example
A numerical example using polynomial ridge regression to demonstrate the difference between Approximation/Estimation error and Bias/Variance decomposition.

# Polynomial Ridge Regression — Bias/Variance vs Approximation/Estimation

This repository contains the reproducible code used to generate the figures in the *Numerical Example: Polynomial Ridge Regression* section of the **Learning Theory Assessment** presentation.  
It demonstrates how ridge regularisation affects bias, variance, approximation, and estimation error, and explores their relationship to optimisation error when comparing across learning procedures.

---

## 📘 Overview

The numerical experiment investigates how introducing a ridge regularisation term affects the bias–variance and approximation–estimation decompositions for polynomial regression models.

We model a simple cubic data-generating process:

$$
\[
y = x^3 + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2), \; \sigma = 0.3
\]
$$

For each ridge strength \( \lambda \), we:
1. Generate 60 random training datasets (\(n = 20\) each).
2. Train a polynomial ridge regressor of degree \(d \in \{1, 5\}\).
3. Evaluate bias, variance, approximation, estimation, and optimisation error components on a large test set (\(n_{\text{test}} = 1000\)).
4. Plot the resulting decompositions to visualise how each component evolves with λ.

---

## 🧠 Conceptual background

The goal is to compare two related decompositions of model generalisation error:

| Decomposition | Terms | Focus |
|----------------|--------|--------|
| **Bias / Variance** | \( \text{Risk} = \text{Bias} + \text{Variance} + \text{Noise} \) | Examines the *learner* (training procedure) |
| **Approximation / Estimation / Optimisation** | \( \text{Excess Risk} = \text{Approx} + \text{Estimation} + \text{Optimisation} \) | Examines the *function class* |

In this experiment:
- **Risk** is measured under MSE loss and kept fixed across λ.
- **Regularisation** (ridge penalty) changes only the *learning rule*, not the underlying function class.
- **Approximation error** and estimation error therefore remains constant, while **bias**, **variance**, and **optimisation error** evolve with λ.

---

## 🧩 Code structure

All logic is contained in a single Python script (`ridge_bias_variance.py`).

### Main components:
| Function | Description |
|-----------|--------------|
| `f0(x)` | True data-generating function \(x^3\). |
| `fit_ridge_poly(deg, lam, x, y)` | Fits a polynomial ridge regression model of degree `deg` and regularisation strength `lam`. |
| `risk_from_preds(preds)` | Estimates population (true) risk using a Monte Carlo approximation + σ² term. |
| `best_in_class(deg)` | Approximates the best-in-class model (Bayes/OLS limit) using a large synthetic dataset. |

---

## 🧮 How to reproduce

### 1. Requirements
Install the following dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

### 2. Run the script

Simply execute:

```bash
python ridge_bias_variance.py
```

This will:
- Train ridge models for polynomial degrees 1 and 5.
- Sweep over λ values in [1e-4, 1e2].
- Reproduce all figures used in the presentation slides.

### 3. Output

The script will display the following plots for each degree:
	1.	Bias vs Approximation Error
	- Shows how regularisation introduces estimation bias.
	2.	Variance vs Estimation Error
	- Illustrates how variance decreases while estimation bias + optimisation error increase.
	3.	Optimisation Error 
	- Plots optimisation error to examine how it changes with regularisation.
	4.	Bias, Variance, and Total Excess Risk
	- Combines main terms on a single figure.
	5.	Decomposition Consistency Check
	- Verifies total_excess_risk ≈ approx + est_err + diff.

These correspond directly to the figures following the slide titled
“Numerical Example: Polynomial Ridge Regression.”

---

## Interpretation highlights

	- For small λ, ridge ≈ OLS → bias ≈ approximation, variance high, optimisation ≈ 0.
	- As λ increases, estimation bias and optimisation error grow, variance decreases.
	- The total excess risk forms the expected U-shape vs λ.
	- A negative cross-procedure “optimisation error” indicates that ridge outperforms OLS in terms of population risk (better generalisation).

---

## Reproducibility checklist

	- Deterministic random seed (rng = np.random.default_rng(7)).
	- Fixed test grid and identical training data generation across runs.
	- Fully self-contained script using only open-source dependencies.

