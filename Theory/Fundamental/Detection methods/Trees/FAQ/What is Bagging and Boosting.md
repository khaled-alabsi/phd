### **1. Bagging (Bootstrap Aggregating)**

**Goal**: Reduce variance (overfitting)

#### **How it works**:

* **Bootstrap sampling**: Generate multiple random datasets by sampling *with replacement* from the original dataset.
* **Train multiple models**: Fit a base model (e.g., decision tree) on each of these datasets independently and in parallel.
* **Aggregate predictions**:

  * For **classification**: majority vote
  * For **regression**: average of predictions

#### **Key Characteristics**:

* Models are **independent** of each other.
* **Parallelizable**.
* Helps in **reducing variance**.
* Common algorithm: **Random Forest**.

#### **Example**:

Imagine training 100 decision trees on 100 different bootstrap samples. Each tree votes on the final class; the most voted class wins.

---

### **2. Boosting**

**Goal**: Reduce bias and variance (underfitting and overfitting)

#### **How it works**:

* Train models **sequentially**, where each model focuses on correcting the errors of the previous one.
* Weights or gradients are used to give more importance to the samples that were misclassified or had high error.
* Combine models using a **weighted sum** or other aggregation technique.

#### **Key Characteristics**:

* Models are **dependent** on each other.
* **Not parallelizable** due to sequential training.
* Helps in **reducing bias** (and also variance to some extent).
* Common algorithms: **AdaBoost**, **Gradient Boosting**, **XGBoost**, **LightGBM**, **CatBoost**.

#### **Example**:

Start with a weak learner (e.g., a shallow tree). Evaluate its errors, and train a second model that tries to fix those errors. Continue this process for many rounds, combining all learners at the end.

---

### **Comparison Table**

| Feature           | Bagging                       | Boosting                       |
| ----------------- | ----------------------------- | ------------------------------ |
| Training          | Parallel (independent models) | Sequential (dependent models)  |
| Data Sampling     | Bootstrap samples             | Entire data (with reweighting) |
| Focus             | Reducing variance             | Reducing bias (and variance)   |
| Aggregation       | Voting / averaging            | Weighted combination           |
| Overfitting Risk  | Lower (especially with trees) | Higher (but controllable)      |
| Speed             | Faster (parallel)             | Slower (sequential)            |
| Example Algorithm | Random Forest                 | XGBoost, AdaBoost, LightGBM    |

