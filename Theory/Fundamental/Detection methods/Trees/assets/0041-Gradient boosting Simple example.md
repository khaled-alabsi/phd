 What `Updating the prediction using the new tree's output — this means the new tree predicts the residual (error), and its output is *added to the previous prediction*.`  means?

---

## Goal:

We want to **predict house prices** based on a single feature: **house size (in m²)**.

We use **gradient boosting** with:

* 3 boosting rounds
* Learning rate `η = 0.1`
* Target variable: `y` = actual house price
* One feature: `x` = house size

We’ll simulate it on **one example datapoint**:

* `x = 100` (m²)
* `y = 300,000` (true price)

---

## Step 0: Initialization

### Prediction:

We start by making a **constant prediction**, usually the **mean** of all `y`.

Let’s assume:

$$
\hat{y}_0 = 250{,}000
$$

This is the starting prediction **before** any trees are trained.

### Residual:

$$
r_1 = y - \hat{y}_0 = 300{,}000 - 250{,}000 = 50{,}000
$$

This residual is the **error** made by the initial guess.

---

## Step 1: Train First Tree on Residuals

We train a decision tree `T₁(x)` to predict `r₁ = 50,000` given input `x = 100`.

Assume the tree predicts:

$$
T_1(x) = 48{,}000
$$

Then we **update the prediction**:

$$
\hat{y}_1 = \hat{y}_0 + \eta \cdot T_1(x)
$$

$$
\hat{y}_1 = 250{,}000 + 0.1 \cdot 48{,}000 = 250{,}000 + 4{,}800 = 254{,}800
$$

### New residual:

$$
r_2 = y - \hat{y}_1 = 300{,}000 - 254{,}800 = 45{,}200
$$

We're closer to the true value.

---

## Step 2: Train Second Tree on Residuals

Train a second tree `T₂(x)` on the new residual `45,200`.

Assume this tree learns:

$$
T_2(x) = 42{,}000
$$

Update the prediction:

$$
\hat{y}_2 = \hat{y}_1 + \eta \cdot T_2(x) = 254{,}800 + 0.1 \cdot 42{,}000 = 254{,}800 + 4{,}200 = 259{,}000
$$

### New residual:

$$
r_3 = y - \hat{y}_2 = 300{,}000 - 259{,}000 = 41{,}000
$$

Still getting closer.

---

## 🌳 Step 3: Train Third Tree on Residuals

Train a third tree `T₃(x)` on the latest residual `41,000`.

Assume the tree predicts:

$$
T_3(x) = 37{,}000
$$

Update prediction:

$$
\hat{y}_3 = \hat{y}_2 + \eta \cdot T_3(x) = 259{,}000 + 0.1 \cdot 37{,}000 = 259{,}000 + 3{,}700 = 262{,}700
$$

### New residual:

$$
r_4 = y - \hat{y}_3 = 300{,}000 - 262{,}700 = 37{,}300
$$

---

## General Pattern

Each time:

* We **train a tree** on the current residual.
* We **scale its output** (learning rate).
* We **add it to the current prediction**.

Over many iterations:

* The prediction gets closer to `y`.
* The residual gets smaller.

---

## Recap of All Steps:

| Step   | Prediction (`ŷ`) | Residual (`y - ŷ`) | Tree Output | Update Amount |
| ------ | ---------------- | ------------------ | ----------- | ------------- |
| Init   | 250,000          | 50,000             | -           | -             |
| Tree 1 | 254,800          | 45,200             | 48,000      | 4,800         |
| Tree 2 | 259,000          | 41,000             | 42,000      | 4,200         |
| Tree 3 | 262,700          | 37,300             | 37,000      | 3,700         |

---

This is how each tree contributes a **small correction** to reduce the model’s overall error. The key idea: **new trees learn to correct the mistakes made so far**.
