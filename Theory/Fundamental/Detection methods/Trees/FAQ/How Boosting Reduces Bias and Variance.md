### **Bias**

* **Definition**: Error due to overly simplistic assumptions in the model.
* **High bias**: The model is too simple to capture the underlying patterns.
* **Symptoms**: Underfitting — performs poorly on both training and test data.

**Example**: A linear model trying to fit non-linear data. It will miss the complexity and make consistently wrong predictions.



---

##### —**"Bias"** also has different meanings depending on the context.

##### **1. In Machine Learning**

As explained earlier, **bias** means a model’s error from overly simplistic assumptions (like assuming linearity in non-linear data). It's part of the **bias-variance tradeoff**.


##### **2. In General Language / News / Social Context**

In these contexts, **bias** means **a tendency or prejudice toward or against something**, often in an unfair or unbalanced way.

**Examples**:

* *"The news channel has a political bias."*
  → It favors one side in political reporting.
* *"There is gender bias in hiring."*
  → One gender is unfairly favored.

This kind of bias is **not mathematical**. It’s about **subjective or systemic favoring** or discrimination.

##### **3. In Neural Networks (Weights and Biases)**

Here, **bias** is closer to a kind of **weight**:

* Each neuron computes:
  **output = activation(weight × input + bias)**
* The **bias** is a trainable scalar added to the weighted input.
* It lets the neuron shift the activation function left or right, helping the model learn patterns **that don't pass through the origin (0)**.

So in this context, **bias is literally a numeric offset**, similar to intercept in linear regression.



##### **Comparison Table**

| Context            | Meaning of "Bias"                     |
| ------------------ | ------------------------------------- |
| ML (bias-variance) | Model error from wrong assumptions    |
| News/Social        | Favoritism or prejudice               |
| Neural Networks    | Numeric value added to input (offset) |


---

### **Variance**

* **Definition**: Error due to sensitivity to small fluctuations in the training data.
* **High variance**: The model learns noise instead of the signal.
* **Symptoms**: Overfitting — performs well on training data but poorly on test data.

**Example**: A deep decision tree memorizing training points instead of generalizing.

---

### **Total Error = Bias² + Variance + Irreducible Error**

---

## **How Boosting Reduces Bias and Variance**

### **1. Reducing Bias**

Boosting starts with a weak learner (often a shallow tree) that underfits. It then:

* Identifies where the model is making errors.
* Trains a new learner **specifically on those errors**.
* Repeats this process, creating a series of learners that together form a much more accurate (and less biased) model.

Each new model **compensates for the bias** of the previous one. The combination becomes increasingly flexible and better at modeling the data.

### **2. Reducing Variance**

Boosting reduces variance **through averaging**, but in a more controlled and directed way than bagging. Unlike bagging, which just aggregates random models, boosting:

* Focuses learning on the "hard" parts of the data.
* Shrinks the contribution of each learner using a **learning rate** (in gradient boosting), which prevents any single model from dominating.

By sequentially combining many weak models (low-variance), boosting builds a strong learner that is both **low-bias** and **low-variance**.

---

### **Illustrative Analogy**

* **Bias** is like using a dull knife: you can’t cut accurately.
* **Variance** is like using a scalpel with shaky hands: you can cut too much or in the wrong direction.
* **Boosting** gives you a sharp knife and a steady hand by progressively refining the cut (model) in small, guided steps.
