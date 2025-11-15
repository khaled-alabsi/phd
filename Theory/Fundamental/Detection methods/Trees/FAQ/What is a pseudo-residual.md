### What does "Calculating (pseudo) residuals" mean in XGBoost?

In gradient boosting (like in XGBoost), we improve the model by **adding a new tree that fixes the mistakes** of the current model. These mistakes are called **residuals** — but more precisely, they are often **pseudo-residuals** because they depend on the loss function.

---

### What is a pseudo-residual?

It is the **negative derivative** of the loss function with respect to the model's current prediction.

---

### General Formula

$$
\tilde{r}_i = -\frac{\partial \mathcal{L}(y_i, \hat{y}_i)}{\partial \hat{y}_i}
$$

**Definitions:**

* $\tilde{r}_i$: the pseudo-residual for data point $i$
* $\mathcal{L}(y_i, \hat{y}_i)$: the loss function, which measures how wrong the prediction $\hat{y}_i$ is compared to the true value $y_i$
* $y_i$: the true target value of the $i$-th data point
* $\hat{y}_i$: the current prediction of the model for the $i$-th data point
* $\frac{\partial}{\partial \hat{y}_i}$: means "derivative with respect to the prediction"

---

### Examples

#### Case 1: Regression with squared error loss

Loss function:

$$
\mathcal{L}(y_i, \hat{y}_i) = \frac{1}{2}(y_i - \hat{y}_i)^2
$$

Then the derivative:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \hat{y}_i - y_i
$$

So the pseudo-residual is:

$$
\tilde{r}_i = -(\hat{y}_i - y_i) = y_i - \hat{y}_i
$$

Here, it **is the same as the true residual**.

---

#### Case 2: Binary classification with logistic loss

Assume labels $y_i \in \{-1, +1\}$, and current prediction is $\hat{y}_i$

Loss function:

$$
\mathcal{L}(y_i, \hat{y}_i) = \log\left(1 + \exp(-2 y_i \hat{y}_i)\right)
$$

Derivative:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{-2 y_i}{1 + \exp(2 y_i \hat{y}_i)}
$$

So pseudo-residual:

$$
\tilde{r}_i = \frac{2 y_i}{1 + \exp(2 y_i \hat{y}_i)}
$$

This is **not the same** as $y_i - \hat{y}_i$, so we call it a **pseudo-residual**.
