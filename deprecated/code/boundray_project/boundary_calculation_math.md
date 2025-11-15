# Mathematical Formulation of Boundary Calculation

## Overview

This document provides the mathematical foundation for the multi-boundary clustering approach used in the Multivariate Relationship Discovery System.

---

## 1. Problem Statement

Given:
- A multivariate time series dataset: **X** ∈ ℝ^(T×V) where T is the number of time steps and V is the number of variables
- Training data from normal operation: **X_train** ∈ ℝ^(T_train×V)

Objective:
- Define **N boundary regions** per variable at multiple sensitivity levels
- Each boundary region captures a normal operating regime
- A violation occurs only when a datapoint is **outside ALL N boundaries**

---

## 2. Cluster Discovery

For each variable v ∈ {1, ..., V}, we extract the univariate time series:

**x_v** = [x_v(1), x_v(2), ..., x_v(T_train)]^T

### 2.1 Finding N Cluster Centers

We implement three methods to find N cluster centers:

#### Method 1: K-Means Clustering

Minimize the within-cluster sum of squares:

```
argmin_{c_1,...,c_N} Σ_{i=1}^N Σ_{x_v(t) ∈ C_i} ||x_v(t) - c_i||^2
```

Where:
- **c_i** is the center of cluster i
- **C_i** is the set of points assigned to cluster i
- N is the number of clusters (default: 3)

Result: **C_v** = {c_v^(1), c_v^(2), ..., c_v^(N)} where c_v^(1) ≤ c_v^(2) ≤ ... ≤ c_v^(N)

#### Method 2: Histogram Peaks

1. Construct histogram with B bins:
   ```
   H_v(b) = |{t : x_v(t) ∈ [edge_b, edge_{b+1})}|, b ∈ {1, ..., B}
   ```

2. Find local maxima (peaks) in H_v(b)

3. Select top N peaks by height:
   ```
   C_v = {bin_center(b_1), bin_center(b_2), ..., bin_center(b_N)}
   ```
   where b_1, b_2, ..., b_N correspond to the N highest peaks

4. If fewer than N peaks found, fall back to Method 3

#### Method 3: Quantile Positions

Evenly space N cluster centers across the data distribution:

```
c_v^(i) = Q_{x_v}(p_i)  for i ∈ {1, ..., N}
```

Where:
```
p_i = i/(N+1)  for i ∈ {1, ..., N}
```

And Q_{x_v}(p) denotes the p-th quantile of **x_v**.

For N=3: p_1 = 0.25, p_2 = 0.50, p_3 = 0.75

---

## 3. Boundary Calculation Around Each Cluster

For each cluster center c_v^(i), we define a boundary region [L_v^(i), U_v^(i)].

### 3.1 Sensitivity Levels

Three sensitivity levels with different quantile widths (q):

| Level | Quantile (q) | Coverage | Purpose |
|-------|--------------|----------|---------|
| Sensitive | 0.25 | 25% | Early warning signals, subtle changes |
| Medium | 0.40 | 40% | Intermediate monitoring |
| Large | 0.60 | 60% | Critical alarms, major disturbances |

### 3.2 Local Data Selection

For cluster i with center c_v^(i), select the K nearest data points:

```
D_v^(i) = {x_v(t) : t ∈ TopK(|x_v(t) - c_v^(i)|)}
```

Where:
```
K = max(10, ⌊T_train / N⌋)
```

### 3.3 Quantile-Based Boundary Computation

For sensitivity level with quantile width q:

**Lower quantile position:**
```
q_L = (1 - q) / 2
```

**Upper quantile position:**
```
q_U = 1 - q_L = (1 + q) / 2
```

**Boundary limits:**
```
L_v^(i) = Q_{D_v^(i)}(q_L)
U_v^(i) = Q_{D_v^(i)}(q_U)
```

Where Q_{D_v^(i)}(p) is the p-th quantile of the local dataset D_v^(i).

### 3.4 Boundary Width

The width of boundary i:
```
W_v^(i) = U_v^(i) - L_v^(i)
```

---

## 4. Complete Boundary Set

For variable v at sensitivity level ℓ (Sensitive, Medium, or Large):

**B_v^(ℓ)** = {(L_v^(1,ℓ), U_v^(1,ℓ)), (L_v^(2,ℓ), U_v^(2,ℓ)), ..., (L_v^(N,ℓ), U_v^(N,ℓ))}

Where each tuple represents one boundary region.

---

## 5. Violation Detection Logic

### 5.1 Binary Violation Detection

For a datapoint x_v(t) at variable v and sensitivity level ℓ:

```
violation(x_v(t), B_v^(ℓ)) = {
    0  if ∃i ∈ {1,...,N} : L_v^(i,ℓ) ≤ x_v(t) ≤ U_v^(i,ℓ)  (inside at least one boundary)
    1  otherwise                                               (outside all boundaries)
}
```

### 5.2 Ternary Violation Detection

For directional analysis (used in co-occurrence detection):

```
violation_ternary(x_v(t), B_v^(ℓ)) = {
    0   if ∃i ∈ {1,...,N} : L_v^(i,ℓ) ≤ x_v(t) ≤ U_v^(i,ℓ)  (normal)
    +1  if ∀i : x_v(t) > U_v^(i,ℓ) OR [outside all] ∧ [x_v(t) > μ_v]  (above)
    -1  if ∀i : x_v(t) < L_v^(i,ℓ) OR [outside all] ∧ [x_v(t) < μ_v]  (below)
}
```

Where μ_v is the mean of training data for variable v.

**Note:** The current implementation uses the simplified version where we compare to the mean. Future enhancements may use more sophisticated logic (see TODO comments in code).

---

## 6. Multi-Level Sensitivity Analysis

For each variable v, we compute boundaries at three levels:

**B_v = {B_v^(Sensitive), B_v^(Medium), B_v^(Large)}**

This creates a **hierarchical boundary structure**:

```
Sensitive (narrow):  |---|  |---|  |---|     (catches subtle deviations)
Medium:              |-----|  |-----|  |-----|  (catches moderate deviations)
Large (wide):        |-------|  |-------|  |-------| (catches major deviations)
```

### 6.1 Relationship Robustness

A relationship between variables i and j is classified as:

1. **Robust**: Detected at all three levels (Sensitive, Medium, Large)
2. **Moderate**: Detected at Sensitive and Medium, but not Large
3. **Sensitive**: Only detected at Sensitive level

---

## 7. User Adjustments

### 7.1 Quantile Adjustment

Users can adjust the quantile width for variable v at level ℓ:

```
q_v^(ℓ,adjusted) = clip(q^(ℓ) + Δq_v^(ℓ), 0.01, 0.99)
```

Where:
- q^(ℓ) is the base quantile for level ℓ
- Δq_v^(ℓ) is the user-specified adjustment
- clip ensures the value stays within valid range

### 7.2 Location Adjustment

Users can shift boundary centers:

```
c_v^(i,adjusted) = c_v^(i) + Δc_v^(i)
```

Where Δc_v^(i) is the user-specified offset.

---

## 8. Example Calculation

### Given:
- Variable v with N=3 clusters
- Cluster centers: c_v = {-0.26, -0.03, 0.22}
- Sensitive level: q = 0.25
- Training data: 500 samples

### Step-by-step for cluster 2 (center = -0.03):

1. **Select local data:**
   ```
   K = max(10, ⌊500/3⌋) = 166
   D_v^(2) = 166 nearest points to -0.03
   ```

2. **Compute quantile positions:**
   ```
   q_L = (1 - 0.25) / 2 = 0.375
   q_U = (1 + 0.25) / 2 = 0.625
   ```

3. **Calculate boundaries:**
   ```
   L_v^(2) = Q_{D_v^(2)}(0.375) = -0.0642
   U_v^(2) = Q_{D_v^(2)}(0.625) = -0.0044
   ```

4. **Boundary width:**
   ```
   W_v^(2) = -0.0044 - (-0.0642) = 0.0598
   ```

### Result:
Boundary 2 for Sensitive level: [-0.0642, -0.0044] with center -0.03

---

## 9. Mathematical Properties

### 9.1 Boundary Coverage

The total coverage of N boundaries is **not necessarily 100%** of the data range, which is intentional:

```
Coverage_v^(ℓ) = Σ_{i=1}^N (U_v^(i,ℓ) - L_v^(i,ℓ)) / (max(x_v) - min(x_v))
```

Data points falling outside all boundaries are flagged as violations.

### 9.2 Boundary Hierarchy

By design:
```
W_v^(i,Sensitive) ≤ W_v^(i,Medium) ≤ W_v^(i,Large)  for all i ∈ {1,...,N}
```

This ensures that:
- More violations detected at Sensitive level
- Fewer violations detected at Large level
- Hierarchical relationship classification

### 9.3 Cluster Separation

For well-separated clusters, boundaries should not overlap:
```
U_v^(i,ℓ) < L_v^(i+1,ℓ)  for i ∈ {1,...,N-1}
```

If clusters overlap, boundaries may also overlap, which is acceptable for the violation detection logic.

---

## 10. Computational Complexity

For V variables, T_train training samples, and N clusters:

| Operation | Complexity |
|-----------|------------|
| K-means clustering (single variable) | O(T_train · N · I) where I is iterations |
| Histogram peaks (single variable) | O(T_train + B·log(B)) where B is bins |
| Quantile positions (single variable) | O(T_train · log(T_train)) |
| Boundary calculation (all variables) | O(V · N · K · log(K)) where K ≈ T_train/N |
| Violation detection (test data) | O(T_test · V · N) |

**Total training complexity:** O(V · T_train · N · I)

**Total testing complexity:** O(T_test · V · N)

---

## 11. Advantages of Multi-Boundary Approach

1. **Handles Multimodal Distributions:** Variables with multiple operating regimes are naturally captured

2. **Conservative Violation Detection:** Only flags true anomalies (outside all normal regions)

3. **Interpretable:** Each boundary region corresponds to a physical operating mode

4. **Hierarchical Analysis:** Differentiates between robust and subtle relationships

5. **Flexible:** Three clustering methods accommodate different data characteristics

6. **Scalable:** Linear complexity in number of variables

---

## 12. Future Enhancements (TODO)

### 12.1 Dynamic Boundaries

Time-dependent boundary adjustment:
```
c_v^(i)(t) = f(c_v^(i), t, h(x_v(t-w:t)))
```

Where:
- f is an adaptation function
- h is a smoothing function (e.g., exponential weighted moving average)
- w is the window size

### 12.2 Adaptive Quantile Widths

Variable boundary widths based on local data density:
```
q_v^(i) = g(density(D_v^(i)), base_quantile)
```

Where density(D_v^(i)) measures how concentrated the data is around c_v^(i).

### 12.3 Cluster Merging/Splitting

Automatic adjustment of N based on data drift:
```
N_v(t) = h(cluster_quality(x_v(1:t)))
```

---

## References

1. K-means Clustering: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
2. Quantile Estimation: Hyndman, R. J., & Fan, Y. (1996). "Sample quantiles in statistical packages"
3. Peak Detection: Palshikar, G. K. (2009). "Simple algorithms for peak detection in time-series"

---

## Notation Summary

| Symbol | Description |
|--------|-------------|
| V | Number of variables |
| T | Number of time steps |
| N | Number of boundary regions per variable (default: 3) |
| x_v(t) | Value of variable v at time t |
| c_v^(i) | Center of cluster i for variable v |
| L_v^(i,ℓ), U_v^(i,ℓ) | Lower and upper bounds of boundary i at level ℓ |
| q^(ℓ) | Quantile width for sensitivity level ℓ |
| Q_D(p) | p-th quantile of dataset D |
| μ_v | Mean of variable v in training data |
| B_v^(ℓ) | Set of all N boundaries for variable v at level ℓ |

---

*Document Version: 1.0*
*Last Updated: 2025-01-10*
*Author: PhD Research Project - Multivariate Relationship Discovery System*
