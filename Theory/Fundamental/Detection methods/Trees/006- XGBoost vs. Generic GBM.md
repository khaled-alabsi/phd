The term **Gradient Boosted Trees** (or Gradient Boosting Machines, GBMs) refers to a general **algorithmic framework** for building predictive models, while **XGBoost** is a **specific implementation** of this framework with optimizations.

---

## **1. Gradient Boosted Trees (GBT / GBM)**

**Definition**: An ensemble technique that builds decision trees sequentially. Each tree corrects the residuals (errors) of the previous trees using gradient descent.

**Key Concepts**:

* Uses **gradient of a loss function** to fit new trees.
* Trees are **shallow** (often called decision stumps).
* Final model is a **sum of all trees**.
* Supports regression, classification, ranking, etc.

**Known Implementations**:

* Scikit-learn’s `GradientBoostingClassifier` / `Regressor`
* LightGBM
* CatBoost
* XGBoost (yes, it's one too)

---

## **2. XGBoost**

**Definition**: e**X**treme **G**radient **Boost**ing — a high-performance library implementing gradient boosting with engineering and mathematical enhancements.

**Differences (vs. generic GBMs)**:

| Feature                   | Generic GBM                   | XGBoost                                 |
| ------------------------- | ----------------------------- | --------------------------------------- |
| Tree construction         | Typically level-wise          | **Greedy, depth-first** (more flexible) |
| Regularization            | Not always included           | **L1 and L2** regularization supported  |
| Loss function support     | Custom losses need effort     | Built-in + supports custom losses       |
| Parallelization           | Limited                       | **Optimized** for parallelism           |
| Missing values handling   | Must be preprocessed manually | **Handled internally**                  |
| Shrinkage (learning rate) | Supported                     | Supported                               |
| Pruning                   | Not always                    | **Post-pruning** (to avoid overfitting) |
| Histogram optimization    | Rare                          | Optional (enabled in later versions)    |
| Sparsity awareness        | Rare                          | **Built-in**                            |
| Out-of-core computation   | Not common                    | **Supported**                           |
| Performance               | Slower                        | **Highly optimized (C++ core)**         |

---

If you want to **understand the theory** of boosting, start with **gradient boosting**.
If you want **state-of-the-art performance**, **XGBoost** is a go-to tool because it implements GBM **plus many engineering optimizations**.

---

Let's walk through a **numerical example** to illustrate the difference between:

* **Standard Gradient Boosted Trees (GBT)** — naïve/vanilla version (like scikit-learn)
* **XGBoost** — optimized version with regularization, weighted splits, and more

We'll use a **simple regression** example.

---

## **Dataset**

We use a tiny dataset:

| ID | x | y |
| -- | - | - |
| 1  | 1 | 3 |
| 2  | 2 | 5 |
| 3  | 3 | 7 |
| 4  | 4 | 9 |

**True function**: $y = 2x + 1$

We'll train for 1 boosting round (1 tree), max depth = 1 (decision stump), learning rate = 1.0

---

## **Step 1: Initial Prediction**

Both models start with initial prediction:

* For **regression with squared error**, this is the **mean of y**.

$$
\hat{y}^{(0)} = \frac{3 + 5 + 7 + 9}{4} = 6
$$

| ID | y | Initial prediction $\hat{y}^{(0)}$ | Residual $y - \hat{y}^{(0)}$ |
| -- | - | ---------------------------------- | ---------------------------- |
| 1  | 3 | 6                                  | -3                           |
| 2  | 5 | 6                                  | -1                           |
| 3  | 7 | 6                                  | +1                           |
| 4  | 9 | 6                                  | +3                           |

---

## **Step 2: Build Tree to Predict Residuals**

### **Vanilla Gradient Boosting (sklearn-style)**

Use residuals as labels, and fit a regression stump:

#### Split at x = 2.5:

* Left (x=1,2): residuals = $$
-3, -1], mean = -2
* Right (x=3,4): residuals = $$
+1, +3], mean = +2

So the tree is:

```
if x <= 2.5: predict -2
else:        predict +2
```

### **XGBoost Approach**

XGBoost uses **second-order info** (gradient + hessian).

For squared error:

* Gradient $g_i = \hat{y}^{(0)} - y_i$
* Hessian $h_i = 1$ (constant for squared error)

So same as residuals: g = $$
-3, -1, +1, +3], h = $$
1, 1, 1, 1]

It computes **gain** for a split:

$$
\text{Gain} = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
$$

Where:

* $G_L$, $G_R$: sum of gradients on left/right
* $H_L$, $H_R$: sum of hessians
* $\lambda$: regularization (say 0)

At x = 2.5:

* Left: g = $$
-3, -1], G\_L = -4, H\_L = 2
* Right: g = $$
+1, +3], G\_R = +4, H\_R = 2

$$
\text{Gain} = \frac{(-4)^2}{2} + \frac{(+4)^2}{2} - \frac{(0)^2}{4} = 8 + 8 - 0 = 16
$$

So same split selected, but **split chosen based on gain**, not just MSE reduction.

**Leaf values** in XGBoost are computed as:

$$
w_j = -\frac{G_j}{H_j + \lambda}
$$

* Left: $w = -(-4)/2 = 2$
* Right: $w = -(+4)/2 = -2$

Note: same prediction as before, but **computed differently and allows regularization**.

---

## **Step 3: Final Prediction**

### **Vanilla GBT**:

$$
\hat{y}^{(1)} = \hat{y}^{(0)} + \eta \cdot \text{leaf value}
$$

| x | y | Tree Prediction | Final Prediction |
| - | - | --------------- | ---------------- |
| 1 | 3 | -2              | 6 + (-2) = 4     |
| 2 | 5 | -2              | 6 + (-2) = 4     |
| 3 | 7 | +2              | 6 + 2 = 8        |
| 4 | 9 | +2              | 6 + 2 = 8        |

### **XGBoost**:

Same formula, just different internal mechanics for how leaf values were calculated. If same tree is selected and no regularization, predictions match vanilla.

---

## **Key Differences in Example**

| Feature                      | Vanilla GBT       | XGBoost                                  |
| ---------------------------- | ----------------- | ---------------------------------------- |
| How splits are chosen        | Reduce MSE        | Maximize **Gain** using gradients        |
| How leaf values are computed | Mean of residuals | Weighted formula with gradients/hessians |
| Regularization               | None              | **Built-in** L2 (λ) and γ for pruning    |
| Pruning                      | Not done          | Yes — removes low-gain splits            |
| Speed / performance          | Generic           | Optimized (histograms, sparsity, etc.)   |
a