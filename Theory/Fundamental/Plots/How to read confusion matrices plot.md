### Reading a Normalized Confusion Matrix (Row-Normalized)


#### 1. **Definition**

A **normalized confusion matrix** (with `normalize='true'` in scikit-learn) shows the **proportion of predictions per actual class**.

* **Rows = Actual Classes**
* **Columns = Predicted Classes**
* **Cell $$
i]$$
j]** = Fraction of class *i* samples predicted as class *j*
* Row sum = 1.0

---

#### 2. **Example Matrix (4 Classes)**

| Actual \ Predicted | A    | B    | C    | D    |
| ------------------ | ---- | ---- | ---- | ---- |
| **A**              | 0.80 | 0.10 | 0.05 | 0.05 |
| **B**              | 0.00 | 0.70 | 0.20 | 0.10 |
| **C**              | 0.05 | 0.10 | 0.80 | 0.05 |
| **D**              | 0.10 | 0.05 | 0.10 | 0.75 |

---

#### 3. **Reading Rows (Actual → Predicted)**

Each row answers:

> "Given the actual class, how was it predicted?"

* **Row A**: 80% of A predicted as A, 10% as B, 5% as C, 5% as D
* **Row B**: 70% of B predicted correctly, 20% confused with C
* Diagonal = Correct predictions
* Off-diagonal = Misclassifications

**✔ Compare across a row** to see confusion patterns.

---

#### 4. **Reading Columns (Predicted ← Actual)**

Each column shows how samples were **assigned to a predicted class**.

Example: Column B

* Predicted B came from:

  * 10% of A
  * 70% of B
  * 10% of C
  * 5% of D

BUT:

* **✖ Do not compare across columns**
* Columns are **not normalized** (rows are)
* You **can’t say** what fraction of all B predictions are correct without knowing class counts

---

#### 5. **Valid Interpretations**

| Element       | Interpretation                       | Valid                       |
| ------------- | ------------------------------------ | --------------------------- |
| Row values    | Conditional probabilities per actual | ✔                           |
| Column values | Mixed sources of prediction          | ✖ (if normalized over rows) |
| Diagonal      | Correct predictions per class        | ✔                           |
| Row sum       | Always = 1.0                         | ✔                           |
| Column sum    | Not meaningful in row-normalized     | ✖                           |

---

#### 6. **Plot Interpretation**

* **Diagonal cells** should ideally be dark (closer to 1.0)
* **Off-diagonal** cells highlight misclassifications
* Use darker color to indicate higher percentage (e.g., seaborn `Blues`)

---

#### 7. **Alternatives**

If you want:

* **Prediction-wise normalization** → use `normalize='pred'`
* **Overall fraction of all samples** → use `normalize='all'`

---

#### 8. **Final Tip**

> **Row-normalized confusion matrices** are best for analyzing **per-class behavior** of the model — especially in **imbalanced** classification problems.
