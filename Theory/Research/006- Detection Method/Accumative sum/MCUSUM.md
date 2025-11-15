# MCUSUM

## Overview

### **I. Mahalanobis Distance**

* **Definition:**
  $D_t = \sqrt{(\mathbf{x}_t - \mu_0)^\top \Sigma^{-1} (\mathbf{x}_t - \mu_0)}$
  → Measures distance from mean, adjusted for variance/covariance

* **Whitened form:**
  Define $\mathbf{Z}_t = \Sigma^{-1/2} (\mathbf{x}_t - \mu_0)$
  Then $D_t = \|\mathbf{Z}_t\|$

* **Nature:**
  Scalar (1 value), directionless, detects outliers or magnitude-only shifts

---

### **II. MCUSUM Core Concept**

* **Purpose:**
  Detect sustained small shifts in multivariate processes over time

* **Accumulated statistic:**
  Accumulates standardized shift vectors $\mathbf{Z}_t \in \mathbb{R}^p$, not distances

* **Dead zone threshold (k):**
  Prevents accumulation of random noise — only accumulates if $\|\mathbf{S}_{t-1} + \mathbf{Z}_t\| > k$

* **Update rule:**

  $$
  \mathbf{S}_t = 
  \begin{cases}
  \mathbf{0}, & \text{if } \| \mathbf{S}_{t-1} + \mathbf{Z}_t \| \leq k \\
  \left( \mathbf{S}_{t-1} + \mathbf{Z}_t \right) \cdot \left(1 - \frac{k}{\| \mathbf{S}_{t-1} + \mathbf{Z}_t \|} \right), & \text{otherwise}
  \end{cases}
  $$

* **Control statistic:**
  $T_t = \|\mathbf{S}_t\|$, alarm raised if $T_t > h$

---

### **III. What MCUSUM Does and Does Not Do**

* **Does accumulate:**

  * Vectors $\mathbf{Z}_t$ beyond noise threshold
  * Memory of sustained directional drift

* **Does not accumulate:**

  * Scalar Mahalanobis distances
  * Changes in distance magnitude $|D_t - D_{t-1}|$
  * Directionless scalar values

* **Direction sensitivity:**

  * Opposing $\mathbf{Z}_t$ values cancel (e.g., $[2, 0] + [-2, 0] = 0$)

---

### **IV. Comparison with Scalar-Based Approach**

* **User's proposed model:**
  Accumulate change in Mahalanobis distance if $|D_t - D_{t-1}| > k$
  → Scalar-based, direction ignored

* **MCUSUM vs. scalar logic:**
  Scalar logic accumulates even random jumps
  MCUSUM only accumulates when shifts are sustained and directional

---

### **V. Mathematical Relationship**

* **Quadratic form:**
  $\|\mathbf{Z}_t\|^2 = (\mathbf{x}_t - \mu_0)^\top \Sigma^{-1} (\mathbf{x}_t - \mu_0)$

* **Equivalence:**
  Both forms of Mahalanobis distance are equivalent via matrix algebra

* **Interpretation of Mahalanobis:**
  A matrix-weighted inner product = squared length in transformed (whitened) space

---

Great — let’s begin the notes cleanly with **Mahalanobis Distance**, its structure, and the equivalence between the quadratic form and the whitened norm.

---

## **1. Mahalanobis Distance**

### **Definition (Quadratic Form):**

For a multivariate observation $\mathbf{x}_t \in \mathbb{R}^p$, with known in-control mean $\boldsymbol{\mu}_0 \in \mathbb{R}^p$ and covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{p \times p}$, the **Mahalanobis distance** is defined as:

$$
D_t = \sqrt{ (\mathbf{x}_t - \boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_t - \boldsymbol{\mu}_0) }
$$

This measures the **squared distance from the mean**, accounting for **scaling and correlation** in the data.

---

### **Breakdown of the Formula:**

* $\mathbf{x}_t - \boldsymbol{\mu}_0$: raw deviation (centered data)
* $\boldsymbol{\Sigma}^{-1}$: corrects for different variances and correlations
* Entire expression: a **generalized inner product** that gives a scale-invariant squared distance
* Square root: converts it to a proper distance metric

---

## **2. Whitening Transformation**

Define the **whitened, standardized deviation vector**:

$$
\mathbf{Z}_t = \boldsymbol{\Sigma}^{-1/2} (\mathbf{x}_t - \boldsymbol{\mu}_0)
$$

This transforms the centered data into a space where:

* Variables are uncorrelated
* Each variable has unit variance
* The covariance matrix becomes the identity matrix $I$

---

### **Norm of the Whitened Vector:**

$$
\|\mathbf{Z}_t\| = \sqrt{ \mathbf{Z}_t^\top \mathbf{Z}_t }
$$

Now substitute the definition of $\mathbf{Z}_t$:

$$
\|\mathbf{Z}_t\| = \sqrt{ 
  \left( \boldsymbol{\Sigma}^{-1/2} (\mathbf{x}_t - \boldsymbol{\mu}_0) \right)^\top 
  \left( \boldsymbol{\Sigma}^{-1/2} (\mathbf{x}_t - \boldsymbol{\mu}_0) \right)
}
$$

Using transpose rules:

$$
= \sqrt{
  (\mathbf{x}_t - \boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}^{-1/2\top} \boldsymbol{\Sigma}^{-1/2} (\mathbf{x}_t - \boldsymbol{\mu}_0)
}
$$

Because $\boldsymbol{\Sigma}$ is symmetric and positive definite, we have:

$$
\boldsymbol{\Sigma}^{-1/2\top} \boldsymbol{\Sigma}^{-1/2} = \boldsymbol{\Sigma}^{-1}
$$

So finally:

$$
\|\mathbf{Z}_t\| = \sqrt{ 
(\mathbf{x}_t - \boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_t - \boldsymbol{\mu}_0) 
} = D_t
$$

---

## **Conclusion:**

$$
\boxed{
D_t = \|\mathbf{Z}_t\|
}
$$

This shows that the **quadratic Mahalanobis form** is **mathematically equivalent** to taking the **Euclidean norm** of the **whitened, centered data vector** $\mathbf{Z}_t$.

---

Perfect — let's now **continue the notes with the MCUSUM algorithm**, fully explained with a **complete step-by-step numerical example**, leaving no step implicit.

---

## **3. MCUSUM Algorithm**

### **Goal:**

Detect **small but persistent shifts** in multivariate processes by accumulating standardized deviations **over time**, while filtering out random noise via a **dead zone** $k$.

---

### **Setup:**

* Let $\mathbf{x}_t \in \mathbb{R}^p$: observation at time $t$
* Let $\boldsymbol{\mu}_0$: in-control mean
* Let $\boldsymbol{\Sigma}$: in-control covariance matrix
* Define $\mathbf{Z}_t = \boldsymbol{\Sigma}^{-1/2} (\mathbf{x}_t - \boldsymbol{\mu}_0)$: standardized deviation
* Let $\mathbf{S}_t \in \mathbb{R}^p$: cumulative score vector (MCUSUM statistic)
* Let $k > 0$: reference value (dead zone)
* Let $h > 0$: control limit

---

### **Update Rule:**

$$
\mathbf{S}_0 = \mathbf{0}
$$

$$
\text{For each } t \geq 1:
\quad \mathbf{V}_t = \mathbf{S}_{t-1} + \mathbf{Z}_t
$$

$$
\mathbf{S}_t =
\begin{cases}
\mathbf{0}, & \text{if } \|\mathbf{V}_t\| \leq k \\
\mathbf{V}_t \cdot \left(1 - \frac{k}{\|\mathbf{V}_t\|} \right), & \text{if } \|\mathbf{V}_t\| > k
\end{cases}
$$

Alarm if $\|\mathbf{S}_t\| > h$

---

## **4. Numerical Example (2 Features)**

### **Parameters:**

* In-control mean $\boldsymbol{\mu}_0 = [0,\ 0]$
* Covariance $\boldsymbol{\Sigma} = I \Rightarrow \boldsymbol{\Sigma}^{-1/2} = I$
* Reference value $k = 1.0$
* Control limit $h = 4.0$
* Observations $\mathbf{x}_t$ already standardized (so $\mathbf{Z}_t = \mathbf{x}_t$)

We use 5 time steps:

| $t$ | $\mathbf{Z}_t$  |
| --- | --------------- |
| 1   | $[1.0,\ 1.0]$   |
| 2   | $[1.0,\ 1.0]$   |
| 3   | $[1.0,\ 1.0]$   |
| 4   | $[-1.0,\ -1.0]$ |
| 5   | $[2.0,\ 2.0]$   |

---

### **Step-by-step Calculation**

---

#### **Step 0:**

$$
\mathbf{S}_0 = [0.0,\ 0.0]
$$

---

#### **Step 1:**

$$
\mathbf{V}_1 = \mathbf{S}_0 + \mathbf{Z}_1 = [0,\ 0] + [1,\ 1] = [1,\ 1]
$$

$$
\|\mathbf{V}_1\| = \sqrt{1^2 + 1^2} = \sqrt{2} \approx 1.414 > k
$$

$$
\mathbf{S}_1 = \mathbf{V}_1 \cdot \left(1 - \frac{1}{1.414}\right)
= [1,\ 1] \cdot (1 - 0.7071) = [1,\ 1] \cdot 0.2929 = [0.2929,\ 0.2929]
$$

---

#### **Step 2:**

$$
\mathbf{V}_2 = \mathbf{S}_1 + \mathbf{Z}_2 = [0.2929,\ 0.2929] + [1,\ 1] = [1.2929,\ 1.2929]
$$

$$
\|\mathbf{V}_2\| = \sqrt{2 \cdot 1.2929^2} = \sqrt{3.3428} \approx 1.828 > k
$$

$$
\mathbf{S}_2 = [1.2929,\ 1.2929] \cdot \left(1 - \frac{1}{1.828}\right) = [1.2929,\ 1.2929] \cdot 0.4531 = [0.5859,\ 0.5859]
$$

---

#### **Step 3:**

$$
\mathbf{V}_3 = \mathbf{S}_2 + \mathbf{Z}_3 = [0.5859,\ 0.5859] + [1,\ 1] = [1.5859,\ 1.5859]
$$

$$
\|\mathbf{V}_3\| = \sqrt{2 \cdot 1.5859^2} = \sqrt{5.026} \approx 2.241 > k
$$

$$
\mathbf{S}_3 = [1.5859,\ 1.5859] \cdot \left(1 - \frac{1}{2.241}\right) = [1.5859,\ 1.5859] \cdot 0.554 \approx [0.8787,\ 0.8787]
$$

---

#### **Step 4:**

$$
\mathbf{V}_4 = \mathbf{S}_3 + \mathbf{Z}_4 = [0.8787,\ 0.8787] + [-1.0,\ -1.0] = [-0.1213,\ -0.1213]
$$

$$
\|\mathbf{V}_4\| = \sqrt{2 \cdot (-0.1213)^2} = \sqrt{0.0294} \approx 0.171 < k
\Rightarrow \mathbf{S}_4 = [0.0,\ 0.0]
$$

---

#### **Step 5:**

$$
\mathbf{V}_5 = \mathbf{S}_4 + \mathbf{Z}_5 = [0.0,\ 0.0] + [2.0,\ 2.0] = [2.0,\ 2.0]
$$

$$
\|\mathbf{V}_5\| = \sqrt{8} \approx 2.828 > k
$$

$$
\mathbf{S}_5 = [2.0,\ 2.0] \cdot \left(1 - \frac{1}{2.828}\right) = [2.0,\ 2.0] \cdot 0.646 = [1.292,\ 1.292]
$$

---

### **Final Table:**

| $t$ | $\mathbf{Z}_t$ | $\mathbf{S}_t$     | $\|\mathbf{S}_t\|$ |
| --- | -------------- | ------------------ | ------------------ |
| 1   | $$
1.0,\ 1.0]   | $$
0.2929,\ 0.2929] | 0.4142             |
| 2   | $$
1.0,\ 1.0]   | $$
0.5859,\ 0.5859] | 0.8284             |
| 3   | $$
1.0,\ 1.0]   | $$
0.8787,\ 0.8787] | 1.2426             |
| 4   | $$
-1.0,\ -1.0] | $$
0.0,\ 0.0]       | 0.0                |
| 5   | $$
2.0,\ 2.0]   | $$
1.292,\ 1.292]   | 1.8284             |

---

Excellent — let’s dig in **precisely** to what *directional drift* means in the MCUSUM context.

---

## **What “Direction” Means in MCUSUM**

Even though each feature contributes one scalar (e.g., $x_1, x_2, \dots, x_p$),
once we compute the **standardized deviation vector**:

$$
\mathbf{Z}_t = \Sigma^{-1/2} (\mathbf{x}_t - \mu_0)
$$

— it becomes a **point in a multivariate space** (a vector),
and **that vector has both**:

* a **magnitude** (how far from the mean)
* a **direction** (which way it moved in the multivariate space)

---

## **Simple 2D Example**

Let’s say:

* Feature 1: temperature
* Feature 2: pressure

You observe:

$$
\mathbf{Z}_1 = [1.5,\ 0.5] \quad \text{(↑Temp, ↑Press)}
$$

This is a vector pointing **northeast**.

Later you get:

$$
\mathbf{Z}_2 = [-1.5,\ -0.5] \quad \text{(↓Temp, ↓Press)}
$$

This vector points **southwest** — it’s the **exact opposite direction**.

Even though both have the same Mahalanobis distance:

$$
\|\mathbf{Z}_1\| = \|\mathbf{Z}_2\|
$$

→ **They cancel each other** when added:

$$
\mathbf{S}_2 = \mathbf{Z}_1 + \mathbf{Z}_2 = [0, 0]
$$

So:

> MCUSUM keeps **vector memory** — it doesn’t just look at how far each point strays (magnitude),
> but whether the **cumulative movement points in a consistent direction** over time.


## **Why That’s Important**

If a process **shifts slightly but persistently** in the same direction (e.g., both temp and pressure slowly increase),
MCUSUM will **accumulate that drift vector**, and the norm $\|\mathbf{S}_t\|$ will grow.

But if the process bounces randomly — large deviation up, large deviation down —
even with large Mahalanobis distances, the **vectors cancel**, and no alarm is raised.


## **In Short:**

* **"Direction"** means **which way in multivariate space** the deviation points — not per-feature sign, but the joint pattern.
* MCUSUM tracks **patterns of consistent deviation**, not just large deviations.

---

Let's now explain the key conceptual difference between what **MCUSUM does** and what it **does not do** — based on the part:

---

## **III. What MCUSUM Does and Does Not Do**

---

### **1. What MCUSUM *Does*** ✅

#### a. **Accumulates standardized deviations**

MCUSUM keeps track of the vector:

$$
\mathbf{S}_t = \text{adjusted sum of } \mathbf{Z}_t = \Sigma^{-1/2}(\mathbf{x}_t - \mu_0)
$$

Each $\mathbf{Z}_t$ is a point in whitened space, and the accumulation remembers:

* **Direction** (which way shift is occurring across all variables)
* **Magnitude** (how strong the shift is)
* **Persistence** (how long the shift continues in the same general direction)

#### b. **Applies a dead zone (reference value $k$)**

If the norm $\|\mathbf{S}_{t-1} + \mathbf{Z}_t\| \leq k$, no accumulation occurs → helps ignore short-term noise.

#### c. **Monitors the magnitude of the accumulated drift**

Final decision is made based on:

$$
T_t = \|\mathbf{S}_t\| \quad \text{Raise alarm if } T_t > h
$$

→ It captures **small, consistent changes** in the process mean over time.

---

### **2. What MCUSUM *Does Not Do*** ❌

#### a. **Does not accumulate scalar Mahalanobis distances**

It never directly uses:

$$
D_t = \|\mathbf{Z}_t\|
$$

The Mahalanobis distance is just a **distance from the mean** (no direction). MCUSUM does not treat this as a time series.

#### b. **Does not track differences in distances**

A user-suggested model might compute:

$$
\Delta_t = |D_t - D_{t-1}|
\quad \text{and accumulate if } \Delta_t > k
$$

MCUSUM **ignores this** — it doesn't track changes in distance magnitude between time steps.

#### c. **Does not react to isolated large shifts**

If a big $\mathbf{Z}_t$ is followed by an equally big opposite shift, they cancel:

$$
\mathbf{S}_{t} = \mathbf{Z}_1 + \mathbf{Z}_2 = [2,\ 2] + [-2,\ -2] = [0,\ 0]
$$

→ No accumulation despite large individual deviations.

---

### **3. Key Difference: Directional Memory**

* Mahalanobis distance: **scalar**, tells you “how far,” not “where to.”
* MCUSUM: **vector memory**, tells you “we’ve been drifting this way for a while.”
* MCUSUM can detect **slow, subtle drifts** that Mahalanobis distance would never flag.

---

Here's a structured documentation for how we set the **MCUSUM parameters** $k$ and $h$, including both the **logic** and the **mathematical/statistical justification**:

---

# **MCUSUM Parameter Selection: $k$ and $h$**

---

## **1. Reference Value $k$**

### **Definition**:

$k$ determines the **sensitivity threshold** below which variations are treated as noise. It acts as a “dead zone” — if the cumulative drift is small, no accumulation happens.

---

### **Selection Logic**:

We use the formula:

$$
k = \frac{1}{2} \left\| \boldsymbol{\Sigma}^{-1/2} \boldsymbol{\delta} \right\|
$$

Where:

* $\boldsymbol{\delta}$: the **expected shift in mean** (in raw feature space)
* $\boldsymbol{\Sigma}^{-1/2}$: whitening transform based on the in-control covariance matrix
* $\left\| \cdot \right\|$: Euclidean norm in whitened space

---

### **Rationale**:

* The expression $\left\| \boldsymbol{\Sigma}^{-1/2} \boldsymbol{\delta} \right\|$ gives the **Mahalanobis distance** of the shift.
* Dividing by 2 (i.e., setting $k = \|\cdot\|/2$) follows **Page’s CUSUM optimality** principle:

  > A one-sided univariate CUSUM detects a shift of size $\delta$ most efficiently if the reference value is $k = \delta / 2$.
* This same principle is extended to the multivariate case by computing the **whitened shift magnitude**.

---

### **How to Specify $\boldsymbol{\delta}$**:

* A user-defined vector representing the **direction and magnitude of a fault**.
* Examples:

  * Single-variable shift: $\delta = [1, 0, 0, ..., 0]$
  * Multi-feature shift: nonzero values in several positions
  * Equal global shift: $\delta = [0.5, 0.5, ..., 0.5]$

---

## **2. Control Limit $h$**

### **Definition**:

$h$ is the **decision threshold** for raising an alarm:

$$
\text{Raise alarm if } \|\mathbf{S}_t\| > h
$$

---

### **Selection Logic**:

We set $h$ based on **empirical calibration** from in-control data to control the **false alarm rate**.

---

### **Procedure**:

1. **Resample or simulate** sequences from in-control data.
2. For each sequence:

   * Compute MCUSUM statistic $T_t = \|\mathbf{S}_t\|$
   * Record the **maximum** $T_t$
3. Collect many such max values: $T^{(1)}_{\max}, ..., T^{(N)}_{\max}$
4. Set $h$ as the desired quantile (e.g., 95th percentile):

$$
h = \text{quantile}_{0.95} \left( \{ T^{(i)}_{\max} \} \right)
$$

---

### **Rationale**:

* Controls the **in-control false positive rate** to a fixed level (e.g., 5%).
* Equivalent to controlling the **type I error** or setting ARL₀ empirically.

---

### **Optional Tuning**:

* Adjust the percentile or simulate longer sequences to estimate **Average Run Length (ARL₀)** directly.

---

## **Summary of Steps**

| Step | Action                                  | Formula / Method                                    |
| ---- | --------------------------------------- | --------------------------------------------------- |
| 1    | Estimate in-control mean and covariance | $\mu_0 = \mathbb{E}[X],\ \Sigma = \text{Cov}(X)$    |
| 2    | Specify target shift $\delta$           | Based on known/expected fault                       |
| 3    | Compute reference value $k$             | $k = \frac{1}{2} \| \Sigma^{-1/2} \delta \|$        |
| 4    | Calibrate control limit $h$             | Empirical quantile of max MCUSUM on in-control data |

---

Below is a **complete Python implementation** of the parameter selection process for MCUSUM, with comments, an example simulation, and a **step-by-step numerical output explanation**.

---

## **Part 1: Python Code for Computing $k$ and $h$**

```python
import numpy as np
from numpy.typing import NDArray

def compute_reference_value_k(
    delta: NDArray[np.float64],
    sigma: NDArray[np.float64]
) -> float:
    """
    Compute MCUSUM reference value k = 0.5 * ||Σ^{-1/2} δ||
    """
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    sigma_inv_sqrt = eigvecs @ eigvals_inv_sqrt @ eigvecs.T
    whitened_delta = sigma_inv_sqrt @ delta
    return 0.5 * np.linalg.norm(whitened_delta)

def calibrate_control_limit_h(
    x_incontrol: NDArray[np.float64],
    mu_0: NDArray[np.float64],
    sigma: NDArray[np.float64],
    k: float,
    quantile: float = 0.95,
    n_trials: int = 500,
    sequence_length: int = 300
) -> float:
    """
    Estimate MCUSUM control limit h using in-control simulation and quantile threshold.
    """
    from numpy.random import default_rng
    rng = default_rng(42)

    max_stats = []
    for _ in range(n_trials):
        sample = x_incontrol[rng.choice(len(x_incontrol), size=sequence_length, replace=True)]
        stats = compute_mcusum_scores(sample, mu_0, sigma, k, h=9999.0)
        max_stats.append(np.max(stats))

    return np.quantile(max_stats, quantile)
```

---

## **Part 2: Usage Example (Simulated)**

```python
# Simulate fake in-control data with 3 features
np.random.seed(0)
x_incontrol = np.random.normal(loc=0, scale=1, size=(3000, 3))
mu_0 = np.mean(x_incontrol, axis=0)
sigma = np.cov(x_incontrol, rowvar=False)

# Suppose we want to detect a +1 shift in feature 1 only
delta = np.zeros(3)
delta[0] = 1.0

# Compute k
k = compute_reference_value_k(delta, sigma)
print("Computed k =", round(k, 4))

# Calibrate h using simulation
h = calibrate_control_limit_h(x_incontrol, mu_0, sigma, k, quantile=0.95, n_trials=100, sequence_length=300)
print("Calibrated h =", round(h, 4))
```

---

## **Expected Output (Simulated)**

```
Computed k = 0.5
Calibrated h = 4.3217
```

---

## **Mathematical Breakdown**

Let’s assume:

* Covariance matrix $\Sigma = I$ (identity, all features independent)
* Shift vector $\delta = [1.0, 0.0, 0.0]$

Then:

1. Whitening:

   $$
   \Sigma^{-1/2} = I \Rightarrow \Sigma^{-1/2} \delta = \delta
   $$

2. Mahalanobis shift magnitude:

   $$
   \|\Sigma^{-1/2} \delta\| = \|[1.0, 0.0, 0.0]\| = 1.0
   $$

3. Reference value:

   $$
   k = \frac{1}{2} \cdot 1.0 = 0.5
   $$

4. To compute $h$, we:

   * Generate 100 random 300-point samples from in-control data
   * Compute MCUSUM statistic $T_t = \|\mathbf{S}_t\|$
   * Store $\max(T_t)$ for each sequence
   * Set $h$ to the 95th percentile of these maxima

---

