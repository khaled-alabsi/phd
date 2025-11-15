**SHAP** and **LIME** are model-agnostic tools for interpreting machine learning models—especially useful in fault diagnosis and process monitoring where **transparency** is critical.

---

## 1. **SHAP (SHapley Additive exPlanations)**

### What it is:

* A unified framework based on **game theory** to explain the output of any ML model.
* Assigns each feature an importance score (positive or negative) for a specific prediction.

### How it works:

* Computes the contribution of each feature by averaging over all possible feature combinations (Shapley values).
* Captures both main effects and interaction effects.

### Why use it:

* **Globally interpretable**: shows which features are most important across all predictions.
* **Locally interpretable**: explains individual predictions (e.g., why fault 3 was predicted).
* **Consistent and theoretically grounded**.

### Example in TE process:

If a Random Forest predicts **Fault A**, SHAP might show:

* Reflux Flow: +0.25 (pushes prediction toward Fault A)
* Feed Temp: -0.10 (pulls prediction away)

### Tools:

* `shap` Python package (`pip install shap`)
* Integrates well with tree models (XGBoost, LightGBM, Random Forest)

---

## 2. **LIME (Local Interpretable Model-agnostic Explanations)**

### What it is:

* Focuses on explaining **individual predictions** by training a simple local model around that point.

### How it works:

* Perturbs the input (e.g., modifies sensor readings slightly)
* Fits a local interpretable model (e.g., linear regression) on the perturbed samples
* Measures how the black-box model reacts

### Why use it:

* Useful when model is **nonlinear and non-interpretable** (like deep neural networks)
* Provides **human-readable explanations**: "This fault was predicted mainly due to high Temp and low Flow"

### Tools:

* `lime` Python package (`pip install lime`)
* Supports classifiers and regressors from scikit-learn, Keras, etc.

---

## Comparison

| Aspect                  | SHAP                          | LIME                                |
| ----------------------- | ----------------------------- | ----------------------------------- |
| Explanation Type        | Additive, consistent          | Local linear approximation          |
| Output                  | Feature contributions         | Feature weights for local surrogate |
| Speed                   | Slower for large models       | Faster for local regions            |
| Global Interpretability | Yes                           | No                                  |
| Best For                | Tree models, critical systems | Deep models, quick local insight    |

---

## When to Use in Fault Diagnosis

* Use **SHAP** if you're using tree-based models (Random Forest, XGBoost) and want **global + local explanations**.
* Use **LIME** for **individual case investigations** especially when using DNNs or CNNs.
* Use both to audit high-risk predictions and to satisfy transparency requirements in regulated environments (e.g., chemical or pharma production).

Let me know if you want a working code example using TE-like data and XGBoost with SHAP.
