
# LightGBM

1. Understand Gradient Boosting Basics

   * Learn the concept of boosting and gradient boosting trees.
   * Study how weak learners (decision trees) are combined sequentially.

2. Study Decision Tree Fundamentals

   * Understand how decision trees split data.
   * Learn about leaf nodes, splits, and how trees model nonlinear relationships.

3. Learn Key Differences of LightGBM

   * Understand LightGBM’s use of Gradient-based One-Side Sampling (GOSS).
   * Learn about Exclusive Feature Bundling (EFB) for dimensionality reduction.
   * Study histogram-based splitting vs. traditional methods.

4. Understand LightGBM’s Tree Growth

   * Learn about leaf-wise tree growth and how it differs from level-wise.
   * Study benefits and risks (overfitting) of leaf-wise growth.

5. Explore LightGBM Parameters

   * Study core parameters: num\_leaves, max\_depth, learning\_rate, boosting\_type.
   * Understand parameters for regularization, feature fraction, bagging fraction.

6. Hands-on with LightGBM Library

   * Install LightGBM in Python or preferred language.
   * Learn API basics for training, prediction, and evaluation.

7. Experiment with Dataset Examples

   * Use standard datasets (e.g., Kaggle datasets) to train LightGBM models.
   * Tune parameters and observe effects on model performance.

8. Learn Model Evaluation Techniques

   * Understand metrics for classification and regression (accuracy, AUC, RMSE).
   * Study cross-validation and early stopping.

9. Understand Feature Importance in LightGBM

   * Learn how to extract and interpret feature importance.

10. Explore Advanced Topics

    * Study handling categorical features natively in LightGBM.
    * Learn about custom objective functions and metrics.
    * Explore integration with GPU for acceleration.

11. Analyze Use Cases and Applications

    * Review common use cases in competitions and industry.

12. Practice Optimization and Deployment

    * Learn best practices for hyperparameter tuning (GridSearch, Bayesian optimization).
    * Understand model saving/loading and deployment basics.

---


### Step 1: Understand Gradient Boosting Basics

* **Boosting**: An ensemble technique that combines multiple weak learners (usually shallow trees) sequentially, where each new learner tries to correct the errors of the combined previous learners.

* **Gradient Boosting**: Uses gradients of a loss function to optimize the model. At each step, a new tree is fit to the negative gradient (residual errors) of the loss function from the current ensemble prediction.

* **Key idea**: Minimize a differentiable loss function by adding trees that predict residuals in a stage-wise fashion.

**Mathematical intuition**:
Given dataset $(x_i, y_i)$, model prediction at step $m$ is:

$$
F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)
$$

where

* $F_{m-1}(x)$ is prediction from previous steps,
* $h_m(x)$ is the new weak learner fit to the residuals,
* $\eta$ is the learning rate (step size).

---

### Step 2: Study Decision Tree Fundamentals

* **Decision Trees**: A tree structure where internal nodes split data on features, and leaf nodes provide predictions.

* **Splits**: Nodes split data based on feature thresholds, aiming to reduce impurity (e.g., minimize variance in regression, maximize information gain in classification).

* **Leaves**: Terminal nodes that output predicted values (e.g., average target value in regression).

* **Tree building**: Greedy, recursive splitting selecting the best feature and threshold at each node.

* **Tree depth and leaves**: Controls complexity and capacity to fit data.

---

**Tree building** To illustrate how **greedy splitting** works in decision trees compared to a non-greedy (global optimal) splitting approach, let's do a simple numerical example.


### Setup

Suppose we have this small dataset:

| Index | Feature $X$ | Target $y$ |
| ----- | ----------- | ---------- |
| 1     | 1           | 10         |
| 2     | 2           | 12         |
| 3     | 3           | 20         |
| 4     | 4           | 22         |

---

### Greedy Splitting (What Decision Trees Actually Do)

* At the root node, the algorithm tries **all possible splits on $X$**.
* It picks the split with the **best immediate variance reduction**.
* It **does not look ahead** or consider future splits.

Possible splits:

* $X \leq 1.5$ → Left: $\{1\}$, Right: $\{2,3,4\}$
* $X \leq 2.5$ → Left: $\{1,2\}$, Right: $\{3,4\}$
* $X \leq 3.5$ → Left: $\{1,2,3\}$, Right: $\{4\}$

Calculate variance reductions:

Parent variance:

$$
\bar{y} = \frac{10 + 12 + 20 + 22}{4} = 16
$$

$$
\text{Var(parent)} = \frac{(10-16)^2 + (12-16)^2 + (20-16)^2 + (22-16)^2}{4} = \frac{36 + 16 + 16 + 36}{4} = 26
$$

---

1. Split $X \leq 1.5$

* Left mean = 10, variance = 0 (one sample)
* Right mean = $\frac{12 + 20 + 22}{3} = 18$, variance = $\frac{(12-18)^2 + (20-18)^2 + (22-18)^2}{3} = \frac{36 + 4 + 16}{3} = 18.67$
* Weighted variance = $\frac{1}{4} \times 0 + \frac{3}{4} \times 18.67 = 14$
* Variance reduction = $26 - 14 = 12$

---

2. Split $X \leq 2.5$

* Left mean = $\frac{10 + 12}{2} = 11$, variance = $\frac{(10-11)^2 + (12-11)^2}{2} = 1$
* Right mean = $\frac{20 + 22}{2} = 21$, variance = $\frac{(20-21)^2 + (22-21)^2}{2} = 1$
* Weighted variance = $\frac{2}{4} \times 1 + \frac{2}{4} \times 1 = 1$
* Variance reduction = $26 - 1 = 25$

---

3. Split $X \leq 3.5$

* Left mean = $\frac{10 + 12 + 20}{3} = 14$, variance = $\frac{(10-14)^2 + (12-14)^2 + (20-14)^2}{3} = \frac{16 + 4 + 36}{3} = 18.67$
* Right mean = 22, variance = 0
* Weighted variance = $\frac{3}{4} \times 18.67 + \frac{1}{4} \times 0 = 14$
* Variance reduction = $26 - 14 = 12$

---

Greedy choice: Split at $X \leq 2.5$ (highest immediate reduction: 25)


Non-Greedy (Global Optimal) Splitting (Hypothetical)

* Suppose we try to optimize the tree by looking at **all possible sequences of splits** (look ahead to 2 levels).

* Consider the best two-level split structure instead of only the root split.


### Example:

* Suppose splitting first at $X \leq 1.5$ (variance reduction 12) is not best immediately, but if we look ahead:

* On the right child $\{2,3,4\}$, split at $X \leq 3.5$:

  * Left node: $\{2,3\}$, mean = 16, variance = $\frac{(12-16)^2 + (20-16)^2}{2} = 16$
  * Right node: $\{4\}$, mean = 22, variance = 0

* Total variance after two splits:

  * Left leaf: $\{1\}$, variance = 0
  * Right-left leaf: $\{2,3\}$, variance = 16
  * Right-right leaf: $\{4\}$, variance = 0

Weighted variance:

$$
\frac{1}{4} \times 0 + \frac{2}{4} \times 16 + \frac{1}{4} \times 0 = 8
$$

Variance reduction after two splits = $26 - 8 = 18$

---

### Summary

| Approach                | Variance Reduction | Split(s)                                       |
| ----------------------- | ------------------ | ---------------------------------------------- |
| Greedy (one step)       | 25                 | $X \leq 2.5$                                   |
| Non-greedy (look ahead) | 18                 | $X \leq 1.5$, then $X \leq 3.5$ on right child |

Here greedy picks the best immediate split $X \leq 2.5$ (variance reduction 25), ignoring that the two-level split starting with $X \leq 1.5$ achieves a smaller immediate but potentially better tree overall.


**Key:**

* Greedy chooses splits that are best **at the current node only** without looking ahead.
* Non-greedy (exhaustive) would consider future splits but is computationally infeasible for large datasets.

### Step 3: Learn Key Differences of LightGBM

LightGBM improves traditional gradient boosting with several unique techniques:

* **Gradient-based One-Side Sampling (GOSS)**
  Instead of using all data points for splitting, GOSS keeps all instances with large gradients (hard-to-predict samples) and randomly samples from instances with small gradients (easy samples). This reduces data size while preserving accuracy and speeds up training.
[Read more.](<FAQ/Gradient-based One-Side Sampling GOSS.md>)
* **Exclusive Feature Bundling (EFB)**
  Many features are mutually exclusive (rarely nonzero at the same time). EFB bundles these sparse features into fewer combined features to reduce dimensionality and speed up training without losing information.
  [Read more.](<FAQ/Exclusive Feature Bundling EFB.md>)

* **Histogram-based Splitting**
  LightGBM bins continuous feature values into discrete bins (histogram). This speeds up split finding by working on bins instead of raw values, saving memory and computation.
  [Read more.](<FAQ/Histogram-based Splitting.md>)

* **Leaf-wise Tree Growth with Depth Limit**
  Unlike level-wise growth used in many GBMs (grow all nodes at one depth before going deeper), LightGBM grows the tree leaf-wise by splitting the leaf with maximum loss reduction. This can reduce loss faster but risks overfitting if uncontrolled.
  [Read more.](<FAQ/Leaf-wise Tree Growth with Depth Limit.md>)

These innovations allow LightGBM to be faster and more memory-efficient while maintaining or improving accuracy compared to traditional gradient boosting implementations.

---

### Step 4: Understand LightGBM’s Tree Growth

LightGBM builds trees **leaf-wise**, unlike most traditional gradient boosting libraries which use **level-wise** growth.

#### 1. **Level-wise (e.g., XGBoost, scikit-learn)**

* Expands all nodes at a given depth before going deeper.
* Grows tree symmetrically.
* Easier to control overfitting using `max_depth`.
* Slower to converge because it doesn't always pick the best split first.

#### 2. **Leaf-wise (LightGBM)**

* Chooses the leaf with the **highest loss reduction** and splits it — not all leaves at once.
* Grows tree **asymmetrically**.
* Converges faster because it always makes the most impactful split.
* **Higher risk of overfitting**, especially on small datasets. Requires regularization or depth constraints.

**Visual difference**:

* Level-wise:

  ```
  Depth 0     A  
             / \
  Depth 1   B   C  
           / \ / \
  ```
* Leaf-wise:

  ```
  Step 1     A  
             |
           Best split  
             |
             B  
           /   \
       best    less good
  ```

#### Controlling Overfitting

* Use `max_depth`, `num_leaves`, `min_data_in_leaf`, and `min_gain_to_split` to limit complexity.


---

### Step 5: Explore LightGBM Parameters

Here are the **most important LightGBM parameters** categorized by purpose:

---

#### **Core Parameters**

| Parameter       | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `boosting_type` | Type of boosting: `"gbdt"` (default), `"dart"`, `"goss"`, `"rf"`. |
| `objective`     | Task type: `"regression"`, `"binary"`, `"multiclass"`, etc.       |
| `metric`        | Evaluation metric: `"l2"`, `"auc"`, `"binary_logloss"`, etc.      |
| `num_class`     | Number of classes (required for multiclass).                      |

---

#### **Tree Structure Parameters**

| Parameter           | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `num_leaves`        | Max number of leaves per tree. **Too high → overfitting**. |
| `max_depth`         | Max depth. Used to limit tree growth.                      |
| `min_data_in_leaf`  | Minimum number of samples per leaf. Helps regularize.      |
| `min_gain_to_split` | Minimum loss reduction required to make a split.           |

---

#### **Learning Control**

| Parameter              | Description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| `learning_rate`        | Shrinkage applied to each tree’s contribution. Lower = slower but better. |
| `num_iterations`       | Number of boosting rounds (trees).                                        |
| `early_stopping_round` | Stop training if no improvement over N rounds.                            |

---

#### **Sampling Parameters**

| Parameter          | Description                                       |
| ------------------ | ------------------------------------------------- |
| `feature_fraction` | Randomly select a fraction of features per tree.  |
| `bagging_fraction` | Randomly select a fraction of rows per iteration. |
| `bagging_freq`     | Frequency of applying bagging (0 = off).          |

---

#### **Regularization**

| Parameter   | Description                        |
| ----------- | ---------------------------------- |
| `lambda_l1` | L1 regularization term on weights. |
| `lambda_l2` | L2 regularization term on weights. |

---

#### **Categorical Features**

| Parameter             | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `categorical_feature` | Specify categorical columns (LightGBM handles them natively). |

---

You can control model complexity and overfitting mostly via:

* `num_leaves`
* `min_data_in_leaf`
* `max_depth`
* `feature_fraction`
* `bagging_fraction`
* `lambda_l1`, `lambda_l2`

--

### Step 6: Hands-on with LightGBM Library (Python)

#### 1. **Installation**

```bash
pip install lightgbm
```

To enable GPU support:

```bash
pip install lightgbm --install-option=--gpu
```

#### 2. **Basic Usage: Regression Example**

```python
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Tuple

# Generate example data
def load_data() -> Tuple:
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# Train LightGBM model
model: LGBMRegressor = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse: float = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
```

#### 3. **Basic Usage: Classification Example**

```python
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf: LGBMClassifier = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc: float = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
```

#### 4. **Key Methods**

* `.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=10)`
* `.predict(X)`
* `.score(X, y)`
* `.feature_importances_`


---

LightGBM is **closer to XGBoost** than to classical Gradient Boosting.

### Reasoning:

| Feature                | Gradient Boosting (original)            | XGBoost                                | LightGBM                                       |
| ---------------------- | --------------------------------------- | -------------------------------------- | ---------------------------------------------- |
| **Core Algorithm**     | Gradient Boosting Decision Trees (GBDT) | Optimized GBDT                         | Optimized GBDT                                 |
| **Split Finding**      | Level-wise                              | Level-wise                             | Leaf-wise with depth limit                     |
| **Speed Optimization** | None                                    | Block structure, cache, sparsity-aware | Histogram-based, GOSS                          |
| **Sampling**           | Bootstrap (optional)                    | Weighted sampling                      | GOSS (Gradient-based One-Side Sampling)        |
| **Parallelism**        | Limited                                 | Yes (feature/block-wise)               | Yes (histogram-based, fast histogram building) |
| **Regularization**     | Limited (learning rate, shrinkage)      | L1, L2                                 | L1, L2, more advanced                          |
| **Missing Values**     | Not handled natively                    | Handled automatically                  | Handled automatically                          |

### Interpretation:

* XGBoost and LightGBM both extend GBDT but in **different directions**:

  * XGBoost focuses on **robustness and stability**.
  * LightGBM focuses on **speed and scalability**.
* LightGBM uses several ideas similar to or even inspired by XGBoost (e.g., regularization, histogram-based trees), but introduces **more radical changes** like:

  * **Leaf-wise tree growth** (more greedy)
  * **GOSS + EFB (Exclusive Feature Bundling)**

### Conclusion (technical):

If you want to compare design philosophy and implementation, **LightGBM is a more aggressively optimized variant of GBDT like XGBoost, but it departs further from classical GBDT than XGBoost does.** So it is **technically closer to XGBoost** in the sense of purpose and optimization layer.
