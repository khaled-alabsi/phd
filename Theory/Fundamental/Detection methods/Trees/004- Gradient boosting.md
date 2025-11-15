# Gradient boosting

**High-Level Introduction to Gradient Boosted Trees**

Gradient Boosted Trees is a machine learning algorithm that builds a *sequence of decision trees*, where each tree *tries to correct the errors* of the previous ones. It does this by:

* Starting with a simple prediction (like the average of the target values)
* Calculating the error (residual) between the current prediction and the actual target
* Training a tree to predict this residual
* Updating the prediction using the new tree's output — this means the new tree predicts the residual (error), and its output is *added to the previous prediction*. [What updating means?](assets/0041-Gradient%20boosting%20Simple%20example.md)  By doing this iteratively, the model gradually removes the error from the original estimate (average), improving accuracy over time
* Repeating this process multiple times to refine the prediction further with each new tree

It's like a team of trees, where each new member helps fix the mistakes of the last one.

Understood! I’ll restructure and rewrite your notes to make them:

- **Consistent**: Remove the “two voices” style (no back-and-forth conversation).
- **Cohesive**: Use a single, clear example throughout.
- **Logical**: Fix numbering and structure — starting from **Step 1**, not Step 2.
- **Clean but complete**: Remove redundancy without omitting meaningful content.

I’ll keep all technical details intact and only remove repeated explanations or unnecessary repetition of steps. The goal is **better readability**, not brevity.

---

# Gradient Boosted Trees: A Structured and Clean Walkthrough

## Overview

**Gradient Boosted Trees (GBTs)** are an ensemble learning method that builds decision trees sequentially to correct errors made by previous trees. Each new tree predicts the residuals (errors) of the current model, and its output is added to the predictions with a learning rate to gradually improve accuracy.

We'll walk through GBT step-by-step using a small dataset predicting **Systolic Blood Pressure (SBP)** based on:
- `Age` (years)
- `Weight` (kg)
- `Exercise` (minutes/day)

### Dataset

| Person | Age | Weight | Exercise | SBP |
|--------|-----|--------|----------|-----|
| A      | 25  | 70     | 30       | 120 |
| B      | 45  | 85     | 10       | 142 |
| C      | 35  | 78     | 20       | 130 |
| D      | 50  | 90     | 5        | 150 |
| E      | 23  | 65     | 60       | 115 |

---

## Step 1: Initial Prediction (Model Initialization)

In regression tasks, gradient boosting typically starts with the **mean** of the target variable as the initial prediction for all samples.

$$
\text{Mean SBP} = \frac{120 + 142 + 130 + 150 + 115}{5} = \frac{657}{5} = 131.4
$$

So we initialize our model:

$$
\hat{y}^{(0)} = 131.4 \quad \text{(for every person)}
$$

Now compute the **residuals** (true value minus predicted value):

| Person | True SBP | Predicted SBP | Residual |
|--------|----------|---------------|----------|
| A      | 120      | 131.4         | -11.4    |
| B      | 142      | 131.4         | 10.6     |
| C      | 130      | 131.4         | -1.4     |
| D      | 150      | 131.4         | 18.6     |
| E      | 115      | 131.4         | -16.4    |

These residuals will be used to train the first tree.

---

## Step 2: Fit First Tree to Residuals

We now build a regression tree to predict these residuals.

To find the best split, we consider each feature and possible thresholds between sorted values.

### Feature: Age

Sorted values: 23, 25, 35, 45, 50  
Midpoints (possible thresholds): 24.0, 30.0, 40.0, 47.5

We calculate total squared error (SE) for each split:

#### Split: Age ≤ 40.0
- Left (A, C, E): residuals = -11.4, -1.4, -16.4 → mean = -9.73 → SE = 116.77
- Right (B, D): residuals = 10.6, 18.6 → mean = 14.6 → SE = 32.0
- **Total SE = 148.77**

This is the best split among all candidates.

---

### Tree Structure After Step 2

```
           Age ≤ 40.0
          /          \
       yes            no
    (-9.73)         (+14.6)
```

Each leaf contains the average residual of the samples in that region.

---

## Step 3: Update Predictions Using Learning Rate

Let’s use a **learning rate** $\eta = 0.1$. This means we add only 10% of each tree's prediction to the current estimate.

Update rule:

$$
\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot \gamma_t
$$

Apply this to each sample:

| Person | Leaf | γ     | New Prediction             | New Residual |
|--------|------|-------|----------------------------|--------------|
| A      | L1   | -9.73 | 131.4 − 0.973 = 130.427     | -10.427      |
| B      | L2   | 14.6  | 131.4 + 1.46 = 132.860      | 9.140        |
| C      | L1   | -9.73 | 131.4 − 0.973 = 130.427     | -0.427       |
| D      | L2   | 14.6  | 131.4 + 1.46 = 132.860      | 17.140       |
| E      | L1   | -9.73 | 131.4 − 0.973 = 130.427     | -15.427      |

These updated residuals will be used to train the next tree.

---

## Step 4: Fit Second Tree to Updated Residuals

We now fit a second tree to the residuals from Step 3:

| Person | Actual SBP | Prediction | Residual |
|--------|------------|------------|----------|
| A      | 120        | 130.427    | -10.427  |
| B      | 142        | 132.860    | 9.140    |
| C      | 130        | 130.427    | -0.427   |
| D      | 150        | 132.860    | 17.140   |
| E      | 115        | 130.427    | -15.427  |

After evaluating possible splits, the best one is:

### Split: Exercise ≤ 15

- Left (B, D): residuals = 9.140, 17.140 → mean = 13.14 → SE = ?
- Right (A, C, E): residuals = -10.427, -0.427, -15.427 → mean = -8.76 → SE = ?

Leaf outputs:
- L1 (B, D): 13.14
- L2 (A, C, E): -8.76

---

### Tree Structure After Step 4

```
           Exercise ≤ 15
              /     \
           yes       no
         (13.14)  (-8.76)
```

---

## Step 5: Update Predictions Again

Use the same update rule:

$$
\hat{y}^{(2)} = \hat{y}^{(1)} + \eta \cdot \gamma_2
$$

| Person | Leaf | γ     | Update | New Prediction | New Residual |
|--------|------|-------|--------|----------------|--------------|
| A      | L2   | -8.76 | -0.876 | 130.427 − 0.876 = 129.551 | -9.551 |
| B      | L1   | 13.14 | 1.314  | 132.860 + 1.314 = 134.174 | 7.826  |
| C      | L2   | -8.76 | -0.876 | 130.427 − 0.876 = 129.551 | 0.449  |
| D      | L1   | 13.14 | 1.314  | 132.860 + 1.314 = 134.174 | 15.826 |
| E      | L2   | -8.76 | -0.876 | 130.427 − 0.876 = 129.551 | -14.551|

---

## Step 6: General Pattern of Updates

At each step $t$, the process follows this pattern:

1. Compute residuals from current predictions:
   $$
   r_i^{(t-1)} = y_i - \hat{y}_i^{(t-1)}
   $$
2. Train a new tree on these residuals.
3. For each sample, get the leaf prediction $\gamma_t(\mathbf{x})$ from the new tree.
4. Update predictions:
   $$
   \hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot \gamma_t(\mathbf{x}_i)
   $$

Repeat until desired number of trees or convergence.

---

## Final Prediction Formula

After $T$ trees, the final prediction is:

$$
\hat{y}^{(T)}(\mathbf{x}) = \hat{y}^{(0)} + \eta \sum_{t=1}^T \gamma_t(\mathbf{x})
$$

Where:
- $\hat{y}^{(0)}$: initial prediction (mean of target)
- $\gamma_t(\mathbf{x})$: leaf output from tree $t$ for input $\mathbf{x}$
- $\eta$: learning rate

---

## Important Notes

### Can Trees Have More Than Two Leaves?

Yes. The number of leaves depends on the **depth** of the tree:
- Depth = 1 → 1 split → 2 leaves
- Depth = 2 → up to 4 leaves
- Depth = 3 → up to 8 leaves

More leaves allow more granular corrections but increase risk of overfitting.

You can control complexity using:
- `max_depth`: max depth of each tree
- `min_samples_leaf`: minimum number of samples per leaf
- `min_impurity_decrease`: minimum improvement required for a split

### How Are Splits Chosen?

All possible midpoints between sorted feature values are considered:
- For `Weight` values [65, 70, 78, 85, 90], midpoints are:
  $$
  67.5,\ 74,\ 81.5,\ 87.5
  $$
- Evaluate each threshold and choose the one that minimizes **total squared error** in resulting leaves.

### Why Use Learning Rate?

Learning rate ($\eta$) controls how much each tree contributes to the final prediction:
- $\eta = 1.0$: full correction (can lead to overfitting)
- $\eta = 0.1$: smaller updates → smoother convergence
- Lower learning rates generally result in better generalization, especially when using many trees

---

## Summary Table of Predictions and Residuals Over Trees

| Person | Actual SBP | $\hat{y}^{(0)}$ | $r^{(0)}$ | $\hat{y}^{(1)}$ | $r^{(1)}$ | $\hat{y}^{(2)}$ | $r^{(2)}$ |
|--------|------------|------------------|-----------|------------------|-----------|------------------|-----------|
| A      | 120        | 131.4            | -11.4     | 130.427          | -10.427   | 129.551          | -9.551    |
| B      | 142        | 131.4            | 10.6      | 132.860          | 9.140     | 134.174          | 7.826     |
| C      | 130        | 131.4            | -1.4      | 130.427          | -0.427    | 129.551          | 0.449     |
| D      | 150        | 131.4            | 18.6      | 132.860          | 17.140    | 134.174          | 15.826    |
| E      | 115        | 131.4            | -16.4     | 130.427          | -15.427   | 129.551          | -14.551   |

---

## Final Thoughts

Gradient boosting works like a team of specialists:
- Each tree focuses on correcting what's left unexplained by earlier trees.
- Small trees with shallow depth and low learning rates help avoid overfitting.
- Residuals evolve at each step, guiding the next tree toward reducing overall error.

The key formula remains consistent:
$$
F(\mathbf{x}) = \hat{y}^{(0)} + \eta \sum_{t=1}^T \gamma_t(\mathbf{x})
$$

Each tree adds a small, targeted correction to the cumulative prediction.

---

**key difference between Gradient Boosted Trees and other ensemble methods like Random Forests**.

Let’s clarify:

---

## ✅ Short Answer:
> **No, gradient boosting does not use bootstrapping or random subsampling of the data at each step (by default).**

Instead, **the same full dataset is used at every boosting iteration**, and each new tree is trained to predict the residuals (errors) made by the current model on that same dataset.

---

## 🧠 Why This Matters

You're likely thinking of **Random Forests**, where:
- Each tree is trained on a **random bootstrap sample** of the data
- Features are randomly sampled at each split

But in **Gradient Boosting**, the process is different:
- Every tree sees **all the training samples**
- But it focuses on **where the model currently makes errors** (i.e., large residuals)

So instead of reducing variance via bagging/bootstrapping (like Random Forest), gradient boosting reduces **bias** iteratively by focusing on correcting errors.

---

## 🔍 What Actually Happens During Training?

Here's how each boosting iteration works:

1. **Start with an initial prediction** (e.g., mean of target)
2. **Compute residuals** for all samples:  
   $r_i^{(t)} = y_i - \hat{y}_i^{(t)}$
3. **Train a new tree to predict these residuals** using the **full dataset**
4. **Update predictions** using learning rate
5. **Repeat** until convergence or max trees reached

Each tree sees the **same features and same rows**, just with updated residuals.

---

## ⚙️ Optional: Subsampling in Gradient Boosting

While **default behavior is to use the full dataset**, some implementations (like `scikit-learn`'s `GradientBoostingRegressor`) do allow optional subsampling:

### Parameters You Might See:
- `subsample`: float between 0 and 1  
  > Fraction of samples to use for fitting each base learner (tree).  
  > Example: `subsample=0.8` → each tree is trained on 80% of the data (chosen randomly).

This is called **Stochastic Gradient Boosting**, and it introduces randomness to reduce overfitting — similar to bagging.

However, this is **optional**, not standard behavior.

---

## 📌 Summary

| Feature | Gradient Boosting | Random Forest |
|--------|-------------------|---------------|
| **Data per Tree** | Same full dataset | Bootstrap sample |
| **Used for** | Reducing bias | Reducing variance |
| **Subsampling?** | Optional (`subsample < 1`) | Default behavior |
| **Feature Sampling?** | Optional (`max_features`) | Default behavior |

---

## 🔄 Optional Extension: Stochastic Gradient Boosting

In some implementations, each tree is trained on a **random subset of the data**:
```python
GradientBoostingRegressor(subsample=0.8)
```
- At each iteration, only 80% of the data is used
- Introduces diversity among trees
- Helps prevent overfitting
- Slower convergence due to less data per tree

This is useful when you have many trees and want to improve generalization.

