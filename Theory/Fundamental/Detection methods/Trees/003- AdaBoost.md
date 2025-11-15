**High-Level Introduction of AdaBoost with Decision Trees**

AdaBoost (Adaptive Boosting) builds a *weighted ensemble* of weak learners—typically **shallow decision trees (stumps)**. It does this by:

1. Starting with **uniform weights** on all samples.
2. Training a weak learner to minimize **weighted classification error**.
3. Increasing the weights on **misclassified samples**, decreasing on correctly classified.
4. Repeating for **T rounds**, each time focusing more on the “hard” samples.
5. Final prediction is a **weighted vote** of all weak learners.

The key concept: each tree learns from the **errors of the previous ones**, and trees are combined **non-uniformly**.

---

**Step 2: Dataset for Realistic Learning and Tree Construction**

We now use a **real-world-inspired dataset** that:

* Has **3 features**
* Is **not linearly separable**
* Cannot be classified perfectly by a single stump
* Creates a **non-trivial error** in each boosting step

We'll use the following synthetic dataset, inspired by medical diagnosis (e.g., predicting diabetes from symptoms):

| ID | Glucose (Glu) | BMI (Bmi) | Age | Diabetes |
| -- | ------------- | --------- | --- | -------- |
| 1  | 148           | 33.6      | 50  | 1        |
| 2  | 85            | 26.6      | 31  | 0        |
| 3  | 183           | 23.3      | 32  | 1        |
| 4  | 89            | 28.1      | 21  | 0        |
| 5  | 137           | 43.1      | 33  | 1        |
| 6  | 116           | 25.6      | 30  | 0        |
| 7  | 78            | 27.6      | 27  | 0        |
| 8  | 115           | 35.3      | 29  | 1        |
| 9  | 197           | 30.5      | 53  | 1        |
| 10 | 125           | 32.3      | 31  | 1        |

**Feature Descriptions:**

* **Glucose (Glu):** Blood glucose level.
* **BMI (Bmi):** Body Mass Index.
* **Age:** Patient's age.
* **Target (Diabetes):** 1 = diabetic, 0 = non-diabetic.

**Reason for Selection:**

* There’s **overlap** in values between classes (e.g., Glucose = 115 appears in both classes).
* A single feature threshold (stump) will misclassify some samples.
* Enables observing **error reweighting**, **alpha update**, and **decision boundary evolution** step-by-step.

**Step 3: Initialization and First Tree Training**

---

### **3.1 Initialize Sample Weights**

We have **N = 10** samples. Initially, AdaBoost sets **equal weights**:

$$
w_i = \frac{1}{N} = \frac{1}{10} = 0.1 \quad \text{for all } i
$$

---

### **3.2 Train First Decision Stump**

Each stump is a one-level decision tree: it picks a **single feature** and a **threshold** to split the data, minimizing **weighted classification error**:

$$
\epsilon = \sum_{i=1}^{N} w_i \cdot \mathbb{1}(h(x_i) \neq y_i)
$$


| Symbol                        | Meaning                                                           |
| ----------------------------- | ----------------------------------------------------------------- |
| $N$                           | Total number of training examples                                 |
| $w_i$                         | Weight of the $i$-th example (assigned by AdaBoost)               |
| $h(x_i)$                      | Prediction of the stump for input $x_i$                           |
| $y_i$                         | True label for input $x_i$                                        |
| $\mathbb{1}(h(x_i) \neq y_i)$ | Indicator function: equals 1 if prediction is wrong, 0 if correct |

---

### What It Does:

* The indicator function $\mathbb{1}(h(x_i) \neq y_i)$ identifies **misclassified** points.
* Each misclassified point contributes its **weight** $w_i$ to the total error.
* The final error $\epsilon$ is the **sum of the weights of the misclassified samples**.

So, this tells us **how much the current stump is "wrong," considering the importance (weights) of each sample**.



### Goal of the **indicator function**

Understand how this function behaves for **correct** and **incorrect** predictions.



$$
\mathbb{1}(h(x_i) \neq y_i) = 
\begin{cases}
1 & \text{if } h(x_i) \neq y_i \quad \text{(wrong prediction)} \\
0 & \text{if } h(x_i) = y_i \quad \text{(correct prediction)}
\end{cases}
$$


### Example:

Suppose we have 4 training examples, with the following:

| $i$ | $x_i$ | $y_i$  | $h(x_i)$ | $w_i$ | $\mathbb{1}(h(x_i) \neq y_i)$ | $w_i \cdot \mathbb{1}$ |
| --- | ----- | ------ | -------- | ----- | ----------------------------- | ---------------------- |
| 1   | —     | **+1** | **+1**   | 0.25  | 0 (correct)                   | 0                      |
| 2   | —     | **+1** | **–1**   | 0.25  | 1 (wrong)                     | 0.25                   |
| 3   | —     | **–1** | **–1**   | 0.25  | 0 (correct)                   | 0                      |
| 4   | —     | **–1** | **+1**   | 0.25  | 1 (wrong)                     | 0.25                   |


Now plug into the formula:

$$
\epsilon = \sum_{i=1}^{4} w_i \cdot \mathbb{1}(h(x_i) \neq y_i) = 0 + 0.25 + 0 + 0.25 = \boxed{0.5}
$$

So the **weighted error** of the stump is **0.5** — it got half the total weight **wrong**.

--- 

**Continue with the example:**
 We now try **every possible split** on each feature, compute the weighted error, and pick the best.



#### Try: `Glucose < 120`

We split samples based on:

* Left: Glucose < 120 → predict `0`
* Right: Glucose ≥ 120 → predict `1`

Let's go through the predictions:

| ID | Glu | True Label | Pred | Correct? |
| -- | --- | ---------- | ---- | -------- |
| 1  | 148 | 1          | 1    | ✔️       |
| 2  | 85  | 0          | 0    | ✔️       |
| 3  | 183 | 1          | 1    | ✔️       |
| 4  | 89  | 0          | 0    | ✔️       |
| 5  | 137 | 1          | 1    | ✔️       |
| 6  | 116 | 0          | 0    | ✔️       |
| 7  | 78  | 0          | 0    | ✔️       |
| 8  | 115 | 1          | 0    | ❌        |
| 9  | 197 | 1          | 1    | ✔️       |
| 10 | 125 | 1          | 1    | ✔️       |

Only **sample 8** is misclassified.

* Total error: $\epsilon = 0.1$

---

#### Try: `BMI < 30`

Predict 0 on left, 1 on right.

| ID | Bmi  | Label | Pred | Correct? |
| -- | ---- | ----- | ---- | -------- |
| 1  | 33.6 | 1     | 1    | ✔️       |
| 2  | 26.6 | 0     | 0    | ✔️       |
| 3  | 23.3 | 1     | 0    | ❌        |
| 4  | 28.1 | 0     | 0    | ✔️       |
| 5  | 43.1 | 1     | 1    | ✔️       |
| 6  | 25.6 | 0     | 0    | ✔️       |
| 7  | 27.6 | 0     | 0    | ✔️       |
| 8  | 35.3 | 1     | 1    | ✔️       |
| 9  | 30.5 | 1     | 1    | ✔️       |
| 10 | 32.3 | 1     | 1    | ✔️       |

Only **sample 3** is misclassified.

* Total error: $\epsilon = 0.1$

---

#### Try: `Age < 40`

Predict 0 on left, 1 on right.

| ID | Age | Label | Pred | Correct? |
| -- | --- | ----- | ---- | -------- |
| 1  | 50  | 1     | 1    | ✔️       |
| 2  | 31  | 0     | 0    | ✔️       |
| 3  | 32  | 1     | 0    | ❌        |
| 4  | 21  | 0     | 0    | ✔️       |
| 5  | 33  | 1     | 0    | ❌        |
| 6  | 30  | 0     | 0    | ✔️       |
| 7  | 27  | 0     | 0    | ✔️       |
| 8  | 29  | 1     | 0    | ❌        |
| 9  | 53  | 1     | 1    | ✔️       |
| 10 | 31  | 1     | 0    | ❌        |

Misclassified: IDs 3, 5, 8, 10 → 4 errors → $\epsilon = 0.4$

---

### **3.3 Best Split**

Both `Glu < 120` and `BMI < 30` yield the **lowest error = 0.1**. We can pick either; let's pick:

* **Stump 1:** `Glucose < 120 → 0`, else `1`

---

### **3.4 Compute Stump Weight (α)**

$$
\alpha_1 = \frac{1}{2} \ln\left(\frac{1 - \epsilon_1}{\epsilon_1}\right) = \frac{1}{2} \ln\left(\frac{0.9}{0.1}\right) ≈ 1.0986
$$

---

### **3.5 Update Sample Weights**

Weights are updated as:

$$
w_i \leftarrow w_i \cdot \exp\left(-\alpha_1 \cdot y_i \cdot h_1(x_i)\right)
$$

Where:

* $y_i \in \{-1, +1\}$ → we map class 0 to -1, class 1 to +1
* $h_1(x_i) \in \{-1, +1\}$ → predictions also in {-1, +1}

Let’s now convert labels and predictions and update weights.

**Step 4: Numerical Update of Weights After First Stump**

We use the rule:

$$
w_i \leftarrow w_i \cdot \exp(-\alpha_1 \cdot y_i \cdot h_1(x_i))
$$

Then normalize so that $\sum w_i = 1$

---

### **4.1 Map Labels and Predictions to ±1**

| ID | Label (y) | Glucose | Pred $h_1(x)$  |
| -- | --------- | ------- | -------------- |
| 1  | 1         | 148     | 1              |
| 2  | 0 → -1    | 85      | -1             |
| 3  | 1         | 183     | 1              |
| 4  | 0 → -1    | 89      | -1             |
| 5  | 1         | 137     | 1              |
| 6  | 0 → -1    | 116     | -1             |
| 7  | 0 → -1    | 78      | -1             |
| 8  | 1         | 115     | **-1** (error) |
| 9  | 1         | 197     | 1              |
| 10 | 1         | 125     | 1              |

**Correct predictions**: all except ID 8
$\alpha_1 ≈ 1.0986$

---

### **4.2 Apply Update Formula**

For correctly classified:

$$
w_i' = w_i \cdot e^{-\alpha_1 \cdot (y_i \cdot h_1(x_i))} = 0.1 \cdot e^{-1.0986 \cdot (1)} = 0.1 \cdot e^{-1.0986} ≈ 0.1 \cdot 0.333 ≈ 0.0333
$$

For misclassified (only ID 8):

$$
w_8' = 0.1 \cdot e^{1.0986} ≈ 0.1 \cdot 3 = 0.3
$$

---

### **4.3 Unnormalized Weights**

| ID | Correct? | Unnormalized $w_i'$ |
| -- | -------- | ------------------- |
| 1  | ✔        | 0.0333              |
| 2  | ✔        | 0.0333              |
| 3  | ✔        | 0.0333              |
| 4  | ✔        | 0.0333              |
| 5  | ✔        | 0.0333              |
| 6  | ✔        | 0.0333              |
| 7  | ✔        | 0.0333              |
| 8  | ✘        | 0.3                 |
| 9  | ✔        | 0.0333              |
| 10 | ✔        | 0.0333              |

Sum = $9 \times 0.0333 + 0.3 = 0.5997$

---

### **4.4 Normalize Weights**

Each new weight:

$$
w_i^{(new)} = \frac{w_i'}{\sum_j w_j'} = \frac{w_i'}{0.5997}
$$

Final weights:

| ID | $w_i^{(new)}$ |
| -- | ------------- |
| 1  | ≈ 0.0556      |
| 2  | ≈ 0.0556      |
| 3  | ≈ 0.0556      |
| 4  | ≈ 0.0556      |
| 5  | ≈ 0.0556      |
| 6  | ≈ 0.0556      |
| 7  | ≈ 0.0556      |
| 8  | ≈ 0.5         |
| 9  | ≈ 0.0556      |
| 10 | ≈ 0.0556      |

---

Now the second tree will **focus more on ID 8**, since it now has half the total weight.

**Step 5: Training the Second Stump Using Updated Sample Weights**

---

### **5.1 Use Updated Weights**

The weights after the first stump are:

| ID | Weight $w_i$ |
| -- | ------------ |
| 1  | 0.0556       |
| 2  | 0.0556       |
| 3  | 0.0556       |
| 4  | 0.0556       |
| 5  | 0.0556       |
| 6  | 0.0556       |
| 7  | 0.0556       |
| 8  | 0.5          |
| 9  | 0.0556       |
| 10 | 0.0556       |

---

### **5.2 Goal for Second Stump**

We try to find the decision stump that **minimizes the weighted error**:

$$
\epsilon = \sum_{i=1}^{N} w_i \cdot \mathbb{1}(h(x_i) \neq y_i)
$$

where

* $w_i$ now emphasize misclassified samples (ID 8 weight = 0.5, others ≈ 0.0556)
* $h(x_i)$ is stump prediction

---

### **5.3 Evaluate Candidate Splits**

We check again candidate thresholds on features, but this time **weighted errors count more where weights are higher**.

---

#### Candidate 1: `Glucose < 120`

Prediction as before:

* Predict 0 if Glucose < 120, else 1.

Misclassified sample is again ID 8.

Weighted error:

$$
\epsilon = w_8 = 0.5
$$

This is worse than before, because ID 8’s weight increased.

---

#### Candidate 2: `BMI < 30`

Misclassified sample previously was ID 3, weight now 0.0556.

Weighted error:

$$
\epsilon = w_3 = 0.0556
$$

Better than candidate 1.

---

#### Candidate 3: `Age < 40`

Misclassified: IDs 3, 5, 8, 10
Sum of weights:

$$
w_3 + w_5 + w_8 + w_{10} = 0.0556 + 0.0556 + 0.5 + 0.0556 = 0.6668
$$

Worst option.

---

#### Candidate 4: Try new threshold on `Glucose < 130`

Predict 0 if Glucose < 130, else 1.

Predictions and misclassifications:

| ID | Glu | Label | Pred | Correct? |
| -- | --- | ----- | ---- | -------- |
| 1  | 148 | 1     | 1    | ✔️       |
| 2  | 85  | 0     | 0    | ✔️       |
| 3  | 183 | 1     | 1    | ✔️       |
| 4  | 89  | 0     | 0    | ✔️       |
| 5  | 137 | 1     | 1    | ✔️       |
| 6  | 116 | 0     | 0    | ✔️       |
| 7  | 78  | 0     | 0    | ✔️       |
| 8  | 115 | 1     | 0    | ❌        |
| 9  | 197 | 1     | 1    | ✔️       |
| 10 | 125 | 1     | 1    | ✔️       |

Same misclassification: ID 8 with weight 0.5 → error 0.5

---

#### Candidate 5: Try `BMI < 35`

Predict 0 if BMI < 35 else 1.

| ID | BMI  | Label | Pred | Correct? |
| -- | ---- | ----- | ---- | -------- |
| 1  | 33.6 | 1     | 0    | ❌        |
| 2  | 26.6 | 0     | 0    | ✔️       |
| 3  | 23.3 | 1     | 0    | ❌        |
| 4  | 28.1 | 0     | 0    | ✔️       |
| 5  | 43.1 | 1     | 1    | ✔️       |
| 6  | 25.6 | 0     | 0    | ✔️       |
| 7  | 27.6 | 0     | 0    | ✔️       |
| 8  | 35.3 | 1     | 1    | ✔️       |
| 9  | 30.5 | 1     | 0    | ❌        |
| 10 | 32.3 | 1     | 0    | ❌        |

Misclassified: IDs 1, 3, 9, 10

Sum weights:

$$
0.0556 + 0.0556 + 0.0556 + 0.0556 = 0.2224
$$

Less than 0.5 → better than candidate 1 but worse than candidate 2.

---

### **5.4 Best Candidate**

The best stump is:

* `BMI < 30` with weighted error $\epsilon = 0.0556$

---

### **5.5 Compute Second Stump Weight**

$$
\alpha_2 = \frac{1}{2} \ln\left(\frac{1 - \epsilon_2}{\epsilon_2}\right) = \frac{1}{2} \ln\left(\frac{1 - 0.0556}{0.0556}\right) ≈ 1.43
$$

---

### **Summary of How Updated Weights Are Used**

* After first stump, samples misclassified get **higher weights**.
* These weights affect **error calculation** for second stump.
* Stumps that correctly classify misclassified samples get **lower weighted error**.
* This guides AdaBoost to focus on **hard-to-classify samples**.

---

**Step 6: Weight Update After Second Stump**

---

### 6.1 Labels and Predictions for Second Stump (`BMI < 30`)

| ID | BMI  | Label $y_i$ | Pred $h_2(x_i)$ | Correct? |
| -- | ---- | ----------- | --------------- | -------- |
| 1  | 33.6 | 1 (+1)      | 1 (BMI ≥ 30)    | ✔️       |
| 2  | 26.6 | 0 (-1)      | 0 (BMI < 30)    | ✔️       |
| 3  | 23.3 | 1 (+1)      | 0 (BMI < 30)    | ✘        |
| 4  | 28.1 | 0 (-1)      | 0 (BMI < 30)    | ✔️       |
| 5  | 43.1 | 1 (+1)      | 1 (BMI ≥ 30)    | ✔️       |
| 6  | 25.6 | 0 (-1)      | 0 (BMI < 30)    | ✔️       |
| 7  | 27.6 | 0 (-1)      | 0 (BMI < 30)    | ✔️       |
| 8  | 35.3 | 1 (+1)      | 1 (BMI ≥ 30)    | ✔️       |
| 9  | 30.5 | 1 (+1)      | 1 (BMI ≥ 30)    | ✔️       |
| 10 | 32.3 | 1 (+1)      | 1 (BMI ≥ 30)    | ✔️       |

Misclassified: ID 3 only.

---

### 6.2 Apply Weight Update Formula

$$
w_i^{new} = w_i \cdot \exp(-\alpha_2 \cdot y_i \cdot h_2(x_i))
$$

For correctly classified (i ≠ 3):

$$
w_i^{new} = w_i \cdot e^{-\alpha_2 \times 1} = w_i \cdot e^{-1.43} \approx w_i \times 0.239
$$

For misclassified (i = 3):

$$
w_3^{new} = w_3 \cdot e^{\alpha_2 \times 1} = w_3 \times e^{1.43} \approx w_3 \times 4.18
$$

---

### 6.3 Calculate New Weights

| ID | Old Weight $w_i$ | Correct? | Calculation    | New Weight $w_i^{new}$ |
| -- | ---------------- | -------- | -------------- | ---------------------- |
| 1  | 0.0556           | ✔️       | 0.0556 × 0.239 | 0.0133                 |
| 2  | 0.0556           | ✔️       | 0.0556 × 0.239 | 0.0133                 |
| 3  | 0.0556           | ✘        | 0.0556 × 4.18  | 0.2325                 |
| 4  | 0.0556           | ✔️       | 0.0556 × 0.239 | 0.0133                 |
| 5  | 0.0556           | ✔️       | 0.0556 × 0.239 | 0.0133                 |
| 6  | 0.0556           | ✔️       | 0.0556 × 0.239 | 0.0133                 |
| 7  | 0.0556           | ✔️       | 0.0556 × 0.239 | 0.0133                 |
| 8  | 0.5              | ✔️       | 0.5 × 0.239    | 0.1195                 |
| 9  | 0.0556           | ✔️       | 0.0556 × 0.239 | 0.0133                 |
| 10 | 0.0556           | ✔️       | 0.0556 × 0.239 | 0.0133                 |

---

### 6.4 Sum and Normalize

Sum all new weights:

$$
S = 0.2325 + 0.1195 + 8 \times 0.0133 = 0.2325 + 0.1195 + 0.1064 = 0.4584
$$

Normalized weights:

$$
w_i^{norm} = \frac{w_i^{new}}{S}
$$

| ID | Normalized Weight       |
| -- | ----------------------- |
| 1  | 0.0133 / 0.4584 ≈ 0.029 |
| 2  | 0.0133 / 0.4584 ≈ 0.029 |
| 3  | 0.2325 / 0.4584 ≈ 0.507 |
| 4  | 0.0133 / 0.4584 ≈ 0.029 |
| 5  | 0.0133 / 0.4584 ≈ 0.029 |
| 6  | 0.0133 / 0.4584 ≈ 0.029 |
| 7  | 0.0133 / 0.4584 ≈ 0.029 |
| 8  | 0.1195 / 0.4584 ≈ 0.261 |
| 9  | 0.0133 / 0.4584 ≈ 0.029 |
| 10 | 0.0133 / 0.4584 ≈ 0.029 |

---

### 6.5 Interpretation

* The misclassified sample **3** now has over **50% of the total weight**.
* Sample 8 weight decreased, because it was classified correctly.
* The algorithm will focus on sample 3 in the next stump.

---

**Step 7: Training Third Stump Using Updated Weights**

---

### 7.1 Current Weights

| ID | Weight $w_i$ |
| -- | ------------ |
| 1  | 0.029        |
| 2  | 0.029        |
| 3  | 0.507        |
| 4  | 0.029        |
| 5  | 0.029        |
| 6  | 0.029        |
| 7  | 0.029        |
| 8  | 0.261        |
| 9  | 0.029        |
| 10 | 0.029        |

---

### 7.2 Evaluate Candidate Stumps

Focus is now on correctly classifying ID 3 and 8 (weights 0.507, 0.261).

---

#### Candidate: `Glucose < 130`

Prediction:

* If Glucose < 130 → 0, else 1.

Check misclassifications:

| ID | Glu  | Label | Pred | Correct? |
| -- | ---- | ----- | ---- | -------- |
| 1  | 148  | 1     | 1    | ✔️       |
| 2  | 85   | 0     | 0    | ✔️       |
| 3  | 183  | 1     | 1    | ✔️       |
| 4  | 89   | 0     | 0    | ✔️       |
| 5  | 137  | 1     | 1    | ✔️       |
| 6  | 116  | 0     | 0    | ✔️       |
| 7  | 78   | 0     | 0    | ✔️       |
| 8  | 35.3 | 1     | 0    | ❌        |
| 9  | 197  | 1     | 1    | ✔️       |
| 10 | 125  | 1     | 0    | ❌        |

Misclassified: IDs 8 and 10.

Weighted error:

$$
\epsilon = w_8 + w_{10} = 0.261 + 0.029 = 0.29
$$

---

#### Candidate: `Age < 35`

Predict 0 if Age < 35 else 1.

Check misclassifications:

| ID | Age | Label | Pred | Correct? |
| -- | --- | ----- | ---- | -------- |
| 1  | 50  | 1     | 1    | ✔️       |
| 2  | 31  | 0     | 0    | ✔️       |
| 3  | 32  | 1     | 0    | ❌        |
| 4  | 21  | 0     | 0    | ✔️       |
| 5  | 33  | 1     | 0    | ❌        |
| 6  | 30  | 0     | 0    | ✔️       |
| 7  | 27  | 0     | 0    | ✔️       |
| 8  | 29  | 1     | 0    | ❌        |
| 9  | 53  | 1     | 1    | ✔️       |
| 10 | 31  | 1     | 0    | ❌        |

Misclassified: IDs 3, 5, 8, 10.

Weighted error:

$$
0.507 + 0.029 + 0.261 + 0.029 = 0.826
$$

Worse than previous candidate.

---

### 7.3 Best Candidate and Alpha

Best stump: `Glucose < 130`, error $\epsilon_3 = 0.29$.

Calculate alpha:

$$
\alpha_3 = \frac{1}{2} \ln \frac{1 - \epsilon_3}{\epsilon_3} = \frac{1}{2} \ln \frac{0.71}{0.29} \approx 0.45
$$

---

### 7.4 Final AdaBoost Model After 3 Stumps

Final prediction is weighted majority vote:

$$
H(x) = \text{sign}\left(\sum_{t=1}^3 \alpha_t h_t(x)\right)
$$

Where

* $h_1:$ `Glucose < 120`
* $h_2:$ `BMI < 30`
* $h_3:$ `Glucose < 130`

Weights:

* $\alpha_1 ≈ 1.10$
* $\alpha_2 ≈ 1.43$
* $\alpha_3 ≈ 0.45$

---

### Explanation Summary

* Initially, all samples weighted equally.
* After each stump, misclassified samples’ weights increase.
* Next stump tries to reduce error focusing on hard samples.
* Alpha weights measure stump confidence; higher alpha means better stump.
* Final prediction is weighted vote of all stumps.
* The ensemble reduces bias by combining weak but focused classifiers.

AdaBoost **boosts weak learners by adapting weights iteratively**, improving overall accuracy step-by-step.

This completes the detailed step-by-step AdaBoost explanation with decision trees on a real dataset.
