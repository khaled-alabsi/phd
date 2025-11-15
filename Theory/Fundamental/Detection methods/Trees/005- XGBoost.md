# XGBoost

## Steps on high level

### **Step 1: Initialize the Model**

In **XGBoost**, Step 1 initializes the model with a **constant prediction**. This step sets the starting point before any trees are built.


##### **What this formula does**

$$
\hat{y}^{(0)} = \arg\min_c \sum_{i=1}^{n} L(y_i, c)
$$

* $\hat{y}^{(0)}$: the initial prediction for every data point
* $c$: a constant value (the same for all data points at this stage)
* $L(y_i, c)$: the loss between the true label $y_i$ and the constant prediction $c$
* The sum aggregates the total loss over all training samples
* $\arg\min_c$: choose the value of $c$ that minimizes the total loss


##### **Examples by task**

##### **Regression with squared error loss**

$$
L(y_i, c) = (y_i - c)^2
$$

Then:

$$
\hat{y}^{(0)} = \arg\min_c \sum_{i=1}^{n} (y_i - c)^2
$$

This is minimized when $c$ is the **mean** of $y$:

$$
\hat{y}^{(0)} = \frac{1}{n} \sum_{i=1}^{n} y_i
$$

#### **Binary classification with logistic loss**

$$
L(y_i, c) = \log(1 + \exp(-y_i c))
$$

Then:

$$
\hat{y}^{(0)} = \arg\min_c \sum_{i=1}^{n} \log(1 + \exp(-y_i c))
$$

Minimized at:

$$
\hat{y}^{(0)} = \log\left(\frac{p}{1 - p}\right)
$$

where $p$ is the proportion of positive class in the dataset. This is the **log-odds**.


##### **Why this step is needed**

Before fitting any trees, XGBoost starts with a simple, constant model. This gives a base to compute:

* the **pseudo-residuals** in the next step (which guide the first tree)
* the **direction and magnitude** of the correction needed by the first tree

This is like taking a first guess at the solution before using boosting to refine it iteratively.


---

### **Step 2: Iterate for T Boosting Rounds**

For each boosting round $t = 1, \dots, T$:

#### **Step 2.1: Compute [Pseudo-Residuals](FAQ/What%20is%20a%20pseudo-residual.md)**

* Compute gradients and optionally second-order derivatives (Hessians):

  $$
  g_i^{(t)} = \frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i^{(t)} = \frac{\partial^2 L(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}
  $$

---

#### **Step 2.2: Fit a Regression Tree**

* Use $(x_i, g_i, h_i)$ to build a tree that splits based on features to minimize:

  $$
  \text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
  $$

  * $G_L, H_L$: gradient and Hessian sums of left child
  * $G_R, H_R$: of right child
  * $\lambda$: L2 regularization
  * $\gamma$: complexity penalty

---

#### **Step 2.3: Compute Leaf Weights**

* For each leaf $j$:

  $$
  w_j = -\frac{G_j}{H_j + \lambda}
  $$

---

#### **Step 2.4: Update Predictions**

* For all samples $i$:

  $$
  \hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)
  $$

  * $\eta$: learning rate
  * $f_t$: tree's prediction (leaf score) for sample $i$

---

### **Step 3: Final Prediction**

* The final output is the sum over all tree outputs:

  $$
  \hat{y}_i = \sum_{t=1}^{T} \eta \cdot f_t(x_i)
  $$
* If classification: apply softmax/sigmoid to get class probabilities

---

### **Optional: Regularization**

* The objective includes a regularization term:

  $$
  \mathcal{L}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)
  $$

  where:

  $$
  \Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
  $$

---

## Example

### **Dataset: Loan Default Prediction**

**Goal**: Predict whether a customer will default on a loan (`default = 1`) or not (`default = 0`).

**Features** (all numerical and continuous for tree-splitting):

| ID | credit\_score | annual\_income | debt\_to\_income | default |
| -- | ------------- | -------------- | ---------------- | ------- |
| 1  | 720           | 65000          | 0.25             | 0       |
| 2  | 680           | 72000          | 0.45             | 1       |
| 3  | 710           | 82000          | 0.32             | 0       |
| 4  | 690           | 61000          | 0.40             | 1       |
| 5  | 730           | 90000          | 0.20             | 0       |

We'll use a **binary logistic loss**:

$$
L(y, \hat{y}) = - \left[ y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \right]
\quad \text{with} \quad \hat{p} = \sigma(\hat{y})
$$

Where:

* $y \in \{0, 1\}$
* $\hat{y}$: raw score before sigmoid
* $\sigma(\hat{y}) = \frac{1}{1 + e^{-\hat{y}}}$: sigmoid to convert to probability

---

## Step 1: **Initialize the model**

We start with the same raw prediction score for all examples:

$$
\hat{y}_i^{(0)} = \log\left( \frac{\bar{y}}{1 - \bar{y}} \right)
$$

where $\bar{y}$ is the mean of target labels.

From data:

* Labels: 0, 1, 0, 1, 0 → $\bar{y} = \frac{2}{5} = 0.4$

So:

$$
\hat{y}^{(0)} = \log\left( \frac{0.4}{0.6} \right) = \log\left(\frac{2}{3}\right) \approx -0.405
$$

This is our base score: **all initial predictions = -0.405** (raw log-odds), which gives:

$$
\hat{p}_i = \sigma(-0.405) \approx 0.4
$$

---

## Step 2: Compute Pseudo-Residuals

We now compute the **first-order (gradient)** and **second-order (Hessian)** derivatives of the loss function with respect to the raw prediction $\hat{y}^{(0)}$.

For binary logistic loss:

$$
g_i = \frac{\partial L}{\partial \hat{y}_i} = \hat{p}_i - y_i
$$

$$
h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2} = \hat{p}_i (1 - \hat{p}_i)
$$

Since we initialized all $\hat{y}_i^{(0)} = -0.405$, we get:

$$
\hat{p}_i = \sigma(-0.405) \approx 0.4
$$

So for all samples:

$$
h_i = 0.4 \cdot 0.6 = 0.24
$$

Now calculate the gradients:

| ID | $y_i$ | $\hat{p}_i$ | $g_i = \hat{p}_i - y_i$ | $h_i = 0.24$ |
| -- | ----- | ----------- | ----------------------- | ------------ |
| 1  | 0     | 0.4         | 0.4                     | 0.24         |
| 2  | 1     | 0.4         | -0.6                    | 0.24         |
| 3  | 0     | 0.4         | 0.4                     | 0.24         |
| 4  | 1     | 0.4         | -0.6                    | 0.24         |
| 5  | 0     | 0.4         | 0.4                     | 0.24         |

We now have gradients and Hessians for each sample. These values will be used to build the first decision tree.

Let's explain **predicted probability** step-by-step and compute it **numerically** for two rows.

---

### Logistic Regression in XGBoost (Raw Score to Probability)

XGBoost does **not** directly predict class labels. It predicts a raw score $\hat{y}_i \in \mathbb{R}$, which must be converted to a probability using the **sigmoid function**:

$$
\hat{p}_i = \sigma(\hat{y}_i) = \frac{1}{1 + e^{-\hat{y}_i}}
$$

At initialization, all samples get the same base score:

$$
\hat{y}^{(0)} = \log\left( \frac{\bar{y}}{1 - \bar{y}} \right) = \log\left( \frac{0.4}{0.6} \right) = \log\left( \frac{2}{3} \right) \approx -0.405
$$

So we use this $\hat{y}^{(0)} = -0.405$ to compute $\hat{p}_i$.

---

### Step-by-Step: Row 1

* $\hat{y}_1 = -0.405$

* Predicted probability:

  $$
  \hat{p}_1 = \frac{1}{1 + e^{-(-0.405)}} = \frac{1}{1 + e^{0.405}} \approx \frac{1}{1 + 1.499} \approx \frac{1}{2.499} \approx 0.4
  $$

* $y_1 = 0$, so:

  $$
  g_1 = \hat{p}_1 - y_1 = 0.4 - 0 = 0.4
  $$

  $$
  h_1 = \hat{p}_1 (1 - \hat{p}_1) = 0.4 \cdot 0.6 = 0.24
  $$

---

### Step-by-Step: Row 2

* $\hat{y}_2 = -0.405$ (same for all in round 0)

* Predicted probability:

  $$
  \hat{p}_2 = \frac{1}{1 + e^{0.405}} \approx 0.4
  $$

* $y_2 = 1$, so:

  $$
  g_2 = 0.4 - 1 = -0.6
  $$

  $$
  h_2 = 0.4 \cdot 0.6 = 0.24
  $$

---

### Full Table (with all samples)

| ID | $y_i$ | $\hat{y}_i^{(0)}$ | $\hat{p}_i$ | $g_i = \hat{p}_i - y_i$ | $h_i = \hat{p}_i (1 - \hat{p}_i)$ |
| -- | ----- | ----------------- | ----------- | ----------------------- | --------------------------------- |
| 1  | 0     | -0.405            | 0.4         | 0.4                     | 0.24                              |
| 2  | 1     | -0.405            | 0.4         | -0.6                    | 0.24                              |
| 3  | 0     | -0.405            | 0.4         | 0.4                     | 0.24                              |
| 4  | 1     | -0.405            | 0.4         | -0.6                    | 0.24                              |
| 5  | 0     | -0.405            | 0.4         | 0.4                     | 0.24                              |


The **predicted probability formula** in XGBoost comes from **logistic regression**, which models probabilities using the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

This function maps any real number $z \in \mathbb{R}$ to the interval $(0, 1)$, which makes it suitable for modeling probabilities.

---

### Why This Formula?

Because the **logistic model** assumes the **log-odds** (also called the *logit*) of the positive class is a linear function of the input:

$$
\text{logit}(p) = \log\left(\frac{p}{1 - p}\right) = z
$$

Then solving for $p$, the probability:

$$
\frac{p}{1 - p} = e^z \Rightarrow p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}}
$$

So the sigmoid is simply the inverse of the logit function.

---

### Why Exponential?

* The **exponential function** arises because the odds (and log-odds) involve ratios, and taking the inverse of the log-odds leads to $e^z$.
* It ensures that the function is **smooth and differentiable**—needed for gradient boosting.
* The sigmoid has the property that its **gradient and Hessian are simple**:

  $$
  \sigma'(z) = \sigma(z)(1 - \sigma(z))
  $$

This makes it ideal for use in optimization algorithms like XGBoost.

The **raw score** from XGBoost (or logistic regression) is **not** a probability. It’s a value on the real number line $\hat{y} \in (-\infty, +\infty)$.

The **sigmoid function** converts this raw score into a **probability** in the interval $(0, 1)$:

$$
\hat{p} = \frac{1}{1 + e^{-\hat{y}}}
$$

So:

* **XGBoost predicts a score**, not a probability.
* **Sigmoid maps this score into a probability.**

This is why we say: **“convert the score to a probability,”** not “calculate the probability directly.”


### Summary:

* $\sigma(z) = \frac{1}{1 + e^{-z}}$ converts scores to probabilities.
* Derived from modeling **log-odds linearly**.
* Exponential ensures correct mapping and nice math properties for optimization.

---

## **Step 2.2: Fit the First Regression Tree by Trying Splits and Computing Gain**

We will:

1. Try a split on one feature (e.g., `credit_score`)
2. Divide the dataset into left/right based on a threshold
3. Compute **Gain** using:

   $$
   \text{Gain} = \frac{1}{2} \left( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right) - \gamma
   $$
4. Repeat for other split points and features

We'll assume:

* $\lambda = 1$
* $\gamma = 0$ (to simplify first tree step)

---

### Data Recap:

| ID | credit\_score | annual\_income | debt\_to\_income | $g_i$ | $h_i$ |
| -- | ------------- | -------------- | ---------------- | ----- | ----- |
| 1  | 720           | 65000          | 0.25             | 0.4   | 0.24  |
| 2  | 680           | 72000          | 0.45             | -0.6  | 0.24  |
| 3  | 710           | 82000          | 0.32             | 0.4   | 0.24  |
| 4  | 690           | 61000          | 0.40             | -0.6  | 0.24  |
| 5  | 730           | 90000          | 0.20             | 0.4   | 0.24  |

---

### Try splitting on `credit_score < 700`

#### Left Child: IDs 2 and 4

(680 and 690 < 700)

* $G_L = -0.6 + (-0.6) = -1.2$
* $H_L = 0.24 + 0.24 = 0.48$

#### Right Child: IDs 1, 3, 5

(720, 710, 730 ≥ 700)

* $G_R = 0.4 + 0.4 + 0.4 = 1.2$
* $H_R = 0.24 × 3 = 0.72$

Now plug into the Gain formula:

$$
\text{Gain} = \frac{1}{2} \left( \frac{(-1.2)^2}{0.48 + 1} + \frac{(1.2)^2}{0.72 + 1} - \frac{0^2}{1.2 + 1} \right)
$$

* Left: $\frac{1.44}{1.48} \approx 0.973$
* Right: $\frac{1.44}{1.72} \approx 0.837$
* Total:

  $$
  \text{Gain} \approx \frac{1}{2}(0.973 + 0.837) = \frac{1}{2}(1.81) = 0.905
  $$

This split gives Gain ≈ **0.905**

---

## Step 2.3: Compute Leaf Weights

Using the best split found (for example, `credit_score < 700`), we now calculate the **leaf weights** for the two resulting leaves.

The formula for each leaf $j$ is:

$$
w_j = -\frac{G_j}{H_j + \lambda}
$$

Recall:

* $\lambda = 1$
* Left child (IDs 2, 4):
  $G_L = -1.2$,
  $H_L = 0.48$
* Right child (IDs 1, 3, 5):
  $G_R = 1.2$,
  $H_R = 0.72$

Calculate weights:

* Left leaf:

  $$
  w_L = -\frac{-1.2}{0.48 + 1} = \frac{1.2}{1.48} \approx 0.81
  $$
* Right leaf:

  $$
  w_R = -\frac{1.2}{0.72 + 1} = -\frac{1.2}{1.72} \approx -0.70
  $$

---

These weights represent how much the raw predictions $\hat{y}_i$ will be adjusted for samples in each leaf during the update step.

---

## Step 2.4: Update Predictions

For each sample, update the raw prediction score by adding the weighted leaf value multiplied by the learning rate $\eta$.

Assume:

$$
\eta = 0.1
$$

---

### Assign samples to leaves

| ID | credit\_score | Leaf       | Weight $w_j$ | Update $\eta \times w_j$   | Previous $\hat{y}_i^{(0)}$ | New $\hat{y}_i^{(1)}$     |
| -- | ------------- | ---------- | ------------ | -------------------------- | -------------------------- | ------------------------- |
| 1  | 720           | Right leaf | -0.70        | $0.1 \times -0.70 = -0.07$ | -0.405                     | $-0.405 - 0.07 = -0.475$  |
| 2  | 680           | Left leaf  | 0.81         | $0.1 \times 0.81 = 0.081$  | -0.405                     | $-0.405 + 0.081 = -0.324$ |
| 3  | 710           | Right leaf | -0.70        | -0.07                      | -0.405                     | -0.475                    |
| 4  | 690           | Left leaf  | 0.81         | 0.081                      | -0.405                     | -0.324                    |
| 5  | 730           | Right leaf | -0.70        | -0.07                      | -0.405                     | -0.475                    |

---

These are the updated raw scores after the first tree.

---

## Step 3: Second Boosting Round — Compute Gradients and Hessians with Updated Predictions

Use the updated raw predictions $\hat{y}_i^{(1)}$ to calculate the new predicted probabilities $\hat{p}_i^{(1)}$, gradients $g_i^{(1)}$, and Hessians $h_i^{(1)}$.

---

### Step 3.1: Calculate predicted probabilities

$$
\hat{p}_i^{(1)} = \sigma(\hat{y}_i^{(1)}) = \frac{1}{1 + e^{-\hat{y}_i^{(1)}}}
$$

Calculate for each sample:

| ID | $\hat{y}_i^{(1)}$ | $\hat{p}_i^{(1)} = \sigma(\hat{y}_i^{(1)})$ (approx) |
| -- | ----------------- | ---------------------------------------------------- |
| 1  | -0.475            | $\frac{1}{1 + e^{0.475}} \approx 0.384$              |
| 2  | -0.324            | $\frac{1}{1 + e^{0.324}} \approx 0.420$              |
| 3  | -0.475            | 0.384                                                |
| 4  | -0.324            | 0.420                                                |
| 5  | -0.475            | 0.384                                                |

---

### Step 3.2: Calculate gradients and Hessians

$$
g_i^{(1)} = \hat{p}_i^{(1)} - y_i
$$

$$
h_i^{(1)} = \hat{p}_i^{(1)} \times (1 - \hat{p}_i^{(1)})
$$

Calculate for each sample:

| ID | $y_i$ | $g_i^{(1)} = \hat{p}_i^{(1)} - y_i$ | $h_i^{(1)} = \hat{p}_i^{(1)} (1 - \hat{p}_i^{(1)})$ |
| -- | ----- | ----------------------------------- | --------------------------------------------------- |
| 1  | 0     | 0.384                               | 0.384 × 0.616 = 0.237                               |
| 2  | 1     | 0.420 - 1 = -0.58                   | 0.420 × 0.580 = 0.244                               |
| 3  | 0     | 0.384                               | 0.237                                               |
| 4  | 1     | -0.58                               | 0.244                                               |
| 5  | 0     | 0.384                               | 0.237                                               |

---

## Step 4: Build the Second Regression Tree Using Updated Gradients and Hessians

We repeat the same splitting process, now using the new gradients $g_i^{(1)}$ and Hessians $h_i^{(1)}$.

---

### Try splitting again on `credit_score < 700`

**Left Child:** IDs 2, 4

* $G_L = -0.58 + (-0.58) = -1.16$
* $H_L = 0.244 + 0.244 = 0.488$

**Right Child:** IDs 1, 3, 5

* $G_R = 0.384 + 0.384 + 0.384 = 1.152$
* $H_R = 0.237 \times 3 = 0.711$

---

### Calculate Gain with $\lambda = 1, \gamma = 0$:

$$
\text{Gain} = \frac{1}{2} \left( \frac{(-1.16)^2}{0.488 + 1} + \frac{(1.152)^2}{0.711 + 1} - \frac{( -1.16 + 1.152 )^2}{0.488 + 0.711 + 1} \right)
$$

Calculate each term:

* Left:

  $$
  \frac{1.3456}{1.488} \approx 0.905
  $$

* Right:

  $$
  \frac{1.327}{1.711} \approx 0.776
  $$

* Parent sum $G_P = -1.16 + 1.152 = -0.008$,
  $H_P = 0.488 + 0.711 = 1.199$

* Parent term:

  $$
  \frac{(-0.008)^2}{1.199 + 1} = \frac{6.4 \times 10^{-5}}{2.199} \approx 2.91 \times 10^{-5}
  $$

---

### Gain:

$$
\frac{1}{2} (0.905 + 0.776 - 0.0000291) \approx \frac{1}{2} (1.681) = 0.841
$$

---

This split still produces positive gain (\~0.841). Similar process can be done for other splits.

## Step 4.2: Calculate Leaf Weights for Second Tree

Using the chosen split `credit_score < 700` and the updated gradients and Hessians:

* Left child (IDs 2, 4):

  $$
  G_L = -1.16, \quad H_L = 0.488
  $$
* Right child (IDs 1, 3, 5):

  $$
  G_R = 1.152, \quad H_R = 0.711
  $$
* Regularization parameter:

  $$
  \lambda = 1
  $$

Calculate leaf weights:

$$
w_j = -\frac{G_j}{H_j + \lambda}
$$

* Left leaf:

  $$
  w_L = -\frac{-1.16}{0.488 + 1} = \frac{1.16}{1.488} \approx 0.78
  $$
* Right leaf:

  $$
  w_R = -\frac{1.152}{0.711 + 1} = -\frac{1.152}{1.711} \approx -0.67
  $$

---

## Step 4.3: Update Predictions After Second Tree

Using learning rate $\eta = 0.1$, update raw predictions:

$$
\hat{y}_i^{(2)} = \hat{y}_i^{(1)} + \eta \times w_j
$$

Assign samples to leaves:

| ID | credit\_score | Leaf  | Leaf weight $w_j$ | Update $\eta \times w_j$ | Previous $\hat{y}_i^{(1)}$ | New $\hat{y}_i^{(2)}$   |
| -- | ------------- | ----- | ----------------- | ------------------------ | -------------------------- | ----------------------- |
| 1  | 720           | Right | -0.67             | -0.067                   | -0.475                     | -0.475 - 0.067 = -0.542 |
| 2  | 680           | Left  | 0.78              | 0.078                    | -0.324                     | -0.324 + 0.078 = -0.246 |
| 3  | 710           | Right | -0.67             | -0.067                   | -0.475                     | -0.542                  |
| 4  | 690           | Left  | 0.78              | 0.078                    | -0.324                     | -0.246                  |
| 5  | 730           | Right | -0.67             | -0.067                   | -0.475                     | -0.542                  |

---

## Step 5: Third Boosting Round — Compute Updated Gradients and Hessians

Use the updated raw predictions $\hat{y}_i^{(2)}$ to calculate predicted probabilities $\hat{p}_i^{(2)}$, gradients $g_i^{(2)}$, and Hessians $h_i^{(2)}$.

---

### Step 5.1: Calculate predicted probabilities

$$
\hat{p}_i^{(2)} = \sigma(\hat{y}_i^{(2)}) = \frac{1}{1 + e^{-\hat{y}_i^{(2)}}}
$$

Calculate approximately:

| ID | $\hat{y}_i^{(2)}$ | $\hat{p}_i^{(2)}$                       |
| -- | ----------------- | --------------------------------------- |
| 1  | -0.542            | $\frac{1}{1 + e^{0.542}} \approx 0.367$ |
| 2  | -0.246            | $\frac{1}{1 + e^{0.246}} \approx 0.438$ |
| 3  | -0.542            | 0.367                                   |
| 4  | -0.246            | 0.438                                   |
| 5  | -0.542            | 0.367                                   |

---

### Step 5.2: Calculate gradients and Hessians

$$
g_i^{(2)} = \hat{p}_i^{(2)} - y_i
$$

$$
h_i^{(2)} = \hat{p}_i^{(2)} (1 - \hat{p}_i^{(2)})
$$

Calculate:

| ID | $y_i$ | $g_i^{(2)}$ | $h_i^{(2)}$                  |
| -- | ----- | ----------- | ---------------------------- |
| 1  | 0     | 0.367       | $0.367 \times 0.633 = 0.232$ |
| 2  | 1     | -0.562      | $0.438 \times 0.562 = 0.246$ |
| 3  | 0     | 0.367       | 0.232                        |
| 4  | 1     | -0.562      | 0.246                        |
| 5  | 0     | 0.367       | 0.232                        |

---

## Summary

#### 1. Feature Importance and Split Selection in XGBoost

* **Split candidates**: For each node, XGBoost tries all possible splits on each feature by sorting unique feature values.
* **Gain calculation**: Uses gradients $G$ and Hessians $H$ sums to calculate the Gain formula (as shown before):

  $$
  \text{Gain} = \frac{1}{2}\left(\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right) - \gamma
  $$
* **Best split**: The split with the highest positive Gain is chosen.
* **Feature importance metrics**:

  * **Gain**: Sum of gain contributed by splits on each feature.
  * **Frequency**: How often a feature is used to split.
  * **Cover**: Sum of Hessians for samples split by the feature (measures coverage).

---

#### 2. Regularization Parameters and Their Effects

* **$\lambda$ (L2 regularization)**:

  * Penalizes large leaf weights $w_j$.
  * Higher $\lambda$ shrinks leaf weights, preventing overfitting.
* **$\gamma$ (minimum loss reduction for splits)**:

  * Minimum Gain required to make a split.
  * Larger $\gamma$ results in simpler trees (fewer splits).
* **Early stopping**: Stops training if validation loss doesn’t improve for N rounds.

---

#### 3. Combining Multiple Trees

* Each tree outputs raw scores $f_t(x)$.
* Final raw prediction is the sum of all trees’ outputs, scaled by learning rate $\eta$:

  $$
  \hat{y} = \sum_{t=1}^T \eta \cdot f_t(x)
  $$
* For classification, apply sigmoid (binary) or softmax (multiclass) on the sum.
* Trees are added sequentially to correct previous errors (boosting).

---

#### 4. Early Stopping, Pruning, Learning Rate

* **Early stopping**: Monitor validation error; stop training if no improvement after certain rounds.
* **Pruning**: Remove splits with negative or zero Gain after tree construction (post-pruning).
* **Learning rate $\eta$**:

  * Controls contribution of each tree.
  * Smaller $\eta$ means slower but more accurate learning, requires more trees.

---

#### 5. Handling Missing Values and Sparsity

* XGBoost treats missing values as a separate branch in splits.
* During training, missing values are assigned to whichever split direction yields better loss reduction.
* Efficient sparse matrix optimizations speed up computations on sparse data.

---

#### 6. Objective Function Optimization Using Gradient and Hessian

* At each iteration, XGBoost approximates the loss using a **second-order Taylor expansion** around current predictions:

  $$
  \mathcal{L} \approx \sum_{i} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)
  $$
* This approximation allows **closed-form optimization** of leaf weights $w_j$.
* Using gradients and Hessians improves convergence speed and accuracy over first-order methods.

---

