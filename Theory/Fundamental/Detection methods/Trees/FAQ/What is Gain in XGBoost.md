# XGBoost gain formula

### **1. Objective and Context**

* Purpose of the gain formula
* Where it fits in tree construction
* When and why it's computed

---

### **2. Intuition**

* What “gain” measures conceptually
* Analogy with “if/else” in programming
* Reframing as “similarity score improvement”

---

### **3. Mathematical Formula**

* Full gain formula
* Definitions of each term: $G_L, H_L, \lambda, \gamma$, etc.

---

### **4. Derivation from Loss Function**

* Taylor expansion of loss
* Optimal leaf weight
* Leaf score = potential loss reduction

---

### **5. Interpretation of Terms**

* Parent node as a virtual leaf
* Children nodes as independent leaves
* Role of regularization $\lambda$, split penalty $\gamma$

---

### **6. Residual-Based View (Squared Error Case)**

* Mapping gradients to residuals
* When $h_i = 1$, and gain becomes residual-based

---

### **7. Example: No Gain vs Positive Gain**

* Step-by-step numerical example (as done earlier)
* One case with 0 gain
* One case with high gain

---

### **8. Gain as Similarity Score Change**

* Interpreting $\frac{G^2}{H + \lambda}$ as a similarity score
* Gain = improvement in similarity after split
* Summary of this viewpoint

---

### **9. Role in Tree Construction**

* How gain is used to select best split
* Why it's preferred over classical impurity measures

---

Here is the first section explained:

---

### **1. Objective and Context**

The **gain formula** in XGBoost quantifies how much a candidate split improves the model. It is used during **tree construction** to decide:

> “Should we split this node? If yes, on which feature and threshold?”

Every internal node in the tree is evaluated as a potential split point. The model compares the **loss reduction** achieved by:

* Keeping the node as a **leaf** (assigning one weight)
* Splitting it into two **child leaves** (assigning two different weights)

This is a **greedy process**:
At each node, XGBoost chooses the split with the **highest gain**, growing the tree step by step.

The formula computes this **gain from splitting**, using the **gradient** and **hessian** of the loss function for the samples in the node. These quantities reflect how much the model's predictions could be improved (gradient) and how certain it is about that direction (hessian).

This gain acts as a **criterion** to:

* Select the best split across all features
* Stop splitting when gain falls below a threshold (regularization $\gamma$)

In summary, the gain formula is the core tool that drives how XGBoost grows trees by greedily choosing the most beneficial splits, measured by potential loss reduction.


---

Here is the second section explained:

---

### **2. Intuition**

The gain formula answers a simple but fundamental question during training:

> “If I split this group of samples, will I be able to make better predictions?”

To understand this, imagine every node in the tree as an **if/else decision block**:

```python
if condition:
    predict with weight w_left
else:
    predict with weight w_right
```

Before the split, there is **only one group** — all samples are assigned a single prediction. After the split, each child can **learn its own weight**, meaning the model can **correct errors more precisely**.

#### Gain Measures the Improvement:

* How well can the parent node predict if left **unsplit**?
* How much better can the model do if it **splits** and learns two separate predictions?

---

#### Similarity Score Analogy:

We can think of each node's "quality" using a **similarity score**:

$$
\text{Similarity} = \frac{G^2}{H + \lambda}
$$

This measures how focused and aligned the gradient signals (or residuals) are. Higher score means the model sees a **clear signal** and can apply a strong corrective update.

Then:

* **Gain** = similarity of children − similarity of parent
* It reflects how much better the model can **specialize its prediction** after the split.

This perspective treats gain as an improvement in the **internal agreement of samples**, not just error reduction — which helps connect statistical reasoning with intuitive programming logic.

Let’s walk through exactly how, under **squared error loss**, the Hessian $H$ becomes just the **number of samples**, and how that turns the gain formula into a sum-of-errors-like expression.

---

### **1. Squared Error Loss**

$$
\ell = \frac{1}{2}(y_i - \hat{y}_i)^2
$$

Then:

* Gradient:

  $$
  g_i = \frac{\partial \ell}{\partial \hat{y}_i} = \hat{y}_i - y_i = -r_i
  $$
* Hessian:

  $$
  h_i = \frac{\partial^2 \ell}{\partial \hat{y}_i^2} = 1
  $$

So for all samples $i$, we have:

* $h_i = 1$
* Therefore:

  $$
  H = \sum_i h_i = \sum_i 1 = n
  $$

---

### **2. Plug into the Gain Formula**

The score of a node (before or after split) is:

$$
\text{Score} = \frac{G^2}{H + \lambda}
$$

Now:

* $G = \sum g_i = \sum (\hat{y}_i - y_i) = -\sum r_i$
* $H = n$

Then:

$$
\text{Score} = \frac{(\sum g_i)^2}{n + \lambda} = \frac{\left( \sum -r_i \right)^2}{n + \lambda}
= \frac{(\sum r_i)^2}{n + \lambda}
$$

---

### **Interpretation**

* You are **summing the residuals** first: $\sum r_i$
* Then **squaring** that total: $(\sum r_i)^2$
* Then dividing by $n + \lambda$

This is **not** the same as $\sum r_i^2$, which would be the **total squared error**.

Instead, this measures **how strongly aligned the errors are**. If all residuals have the **same sign**, then $\sum r_i$ is large, so this score is large → high gain.

If residuals **cancel each other** (e.g., some +r, some -r), then $\sum r_i \approx 0$, so gain is small → not worth splitting.

---

### **Conclusion**

Yes:
Under squared error loss, the gain formula reduces to:

$$
\frac{(\sum \text{residuals})^2}{n + \lambda}
$$

This is a kind of **node purity** or **residual alignment** measure. It’s not the variance of errors, but rather a signal of **how confidently we can apply one correction to the group**.

---

### **3. Mathematical Formula**

The gain from a split in XGBoost is given by:

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

---

#### **Definition of Terms:**

* $G_L$: Sum of gradients in the left child
  $G_L = \sum_{i \in \text{left}} g_i$

* $G_R$: Sum of gradients in the right child
  $G_R = \sum_{i \in \text{right}} g_i$

* $H_L$: Sum of hessians in the left child
  $H_L = \sum_{i \in \text{left}} h_i$

* $H_R$: Sum of hessians in the right child
  $H_R = \sum_{i \in \text{right}} h_i$

* $\lambda$: L2 regularization term on the leaf weights
  Prevents overfitting by penalizing large corrections

* $\gamma$: Minimum required gain to make a split
  Controls complexity by requiring a gain threshold

---

#### **Structure of the Formula:**

* First term: Gain if we split into **left** and **right** leaves
  $\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda}$

* Second term: Score if we keep **parent node as a leaf**
  $\frac{(G_L + G_R)^2}{H_L + H_R + \lambda}$

* Final: Subtract **parent score** from sum of **children scores**, scaled by ½
  Then subtract $\gamma$ to enforce regularization



This formula is computed for **each candidate split** of each feature. The split with the highest **positive gain** is selected, or none is made if all gains are negative or below $\gamma$.

---

 Here is the next section:

---

### **4. Derivation from Loss Function**

The gain formula is derived by approximating the **total training loss** using a **second-order Taylor expansion**. This allows XGBoost to evaluate how much the loss would decrease if a node is split.

---

#### **Step 1: Taylor Approximation of Loss**

For a given prediction $\hat{y}_i$, and loss function $\ell(y_i, \hat{y}_i)$, we expand around the current prediction:

$$
\ell(y_i, \hat{y}_i + w) \approx \ell(y_i, \hat{y}_i) + g_i w + \frac{1}{2} h_i w^2
$$

Where:

* $g_i = \frac{\partial \ell}{\partial \hat{y}_i}$ (gradient)
* $h_i = \frac{\partial^2 \ell}{\partial \hat{y}_i^2}$ (hessian)
* $w$: the leaf weight (correction)

---

#### **Step 2: Summing Over Samples in Node**

Let the node contain samples $i = 1, \dots, n$. Then:

$$
\sum_i \ell(y_i, \hat{y}_i + w) \approx \sum_i \left[ g_i w + \frac{1}{2} h_i w^2 \right] + \frac{1}{2} \lambda w^2
= G w + \frac{1}{2}(H + \lambda) w^2
$$

Where:

* $G = \sum_i g_i$
* $H = \sum_i h_i$

---

#### **Step 3: Optimal Leaf Weight**

To minimize this expression, take derivative w\.r.t. $w$:

$$
\frac{d}{dw} \left( G w + \frac{1}{2}(H + \lambda) w^2 \right) = G + (H + \lambda) w = 0
\Rightarrow w^* = -\frac{G}{H + \lambda}
$$

---

#### **Step 4: Minimum Loss for This Leaf**

Plug $w^*$ back into the loss:

$$
\mathcal{L}_{\text{min}} = G w^* + \frac{1}{2}(H + \lambda)(w^*)^2
= -\frac{G^2}{H + \lambda} + \frac{1}{2} \cdot \frac{G^2}{H + \lambda} = -\frac{1}{2} \cdot \frac{G^2}{H + \lambda}
$$

This is the **score** of a leaf — the higher $G^2/(H + \lambda)$, the more effective the correction.

---

#### **Step 5: Gain From Splitting**

Apply this score to:

* Parent node (before split)
* Left and right child nodes (after split)

Then the gain is:

$$
\text{Gain} = \left(-\frac{1}{2} \cdot \frac{G_L^2}{H_L + \lambda} - \frac{1}{2} \cdot \frac{G_R^2}{H_R + \lambda} \right) - \left(-\frac{1}{2} \cdot \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right)
= \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right]
$$

Subtract $\gamma$ for regularization, and you have the full gain formula.

This connects the formula directly to the objective function being optimized.


---

Here is the next section:

---

### **4. Derivation from Loss Function**

The gain formula is derived by approximating the **total training loss** using a **second-order Taylor expansion**. This allows XGBoost to evaluate how much the loss would decrease if a node is split.

---

#### **Step 1: Taylor Approximation of Loss**

For a given prediction $\hat{y}_i$, and loss function $\ell(y_i, \hat{y}_i)$, we expand around the current prediction:

$$
\ell(y_i, \hat{y}_i + w) \approx \ell(y_i, \hat{y}_i) + g_i w + \frac{1}{2} h_i w^2
$$

Where:

* $g_i = \frac{\partial \ell}{\partial \hat{y}_i}$ (gradient)
* $h_i = \frac{\partial^2 \ell}{\partial \hat{y}_i^2}$ (hessian)
* $w$: the leaf weight (correction)

---

#### **Step 2: Summing Over Samples in Node**

Let the node contain samples $i = 1, \dots, n$. Then:

$$
\sum_i \ell(y_i, \hat{y}_i + w) \approx \sum_i \left[ g_i w + \frac{1}{2} h_i w^2 \right] + \frac{1}{2} \lambda w^2
= G w + \frac{1}{2}(H + \lambda) w^2
$$

Where:

* $G = \sum_i g_i$
* $H = \sum_i h_i$

---

#### **Step 3: Optimal Leaf Weight**

To minimize this expression, take derivative w\.r.t. $w$:

$$
\frac{d}{dw} \left( G w + \frac{1}{2}(H + \lambda) w^2 \right) = G + (H + \lambda) w = 0
\Rightarrow w^* = -\frac{G}{H + \lambda}
$$

---

#### **Step 4: Minimum Loss for This Leaf**

Plug $w^*$ back into the loss:

$$
\mathcal{L}_{\text{min}} = G w^* + \frac{1}{2}(H + \lambda)(w^*)^2
= -\frac{G^2}{H + \lambda} + \frac{1}{2} \cdot \frac{G^2}{H + \lambda} = -\frac{1}{2} \cdot \frac{G^2}{H + \lambda}
$$

This is the **score** of a leaf — the higher $G^2/(H + \lambda)$, the more effective the correction.

---

#### **Step 5: Gain From Splitting**

Apply this score to:

* Parent node (before split)
* Left and right child nodes (after split)

Then the gain is:

$$
\text{Gain} = \left(-\frac{1}{2} \cdot \frac{G_L^2}{H_L + \lambda} - \frac{1}{2} \cdot \frac{G_R^2}{H_R + \lambda} \right) - \left(-\frac{1}{2} \cdot \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right)
= \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right]
$$

Subtract $\gamma$ for regularization, and you have the full gain formula.

This connects the formula directly to the objective function being optimized.

---

Here are the remaining sections for your summary note:

---

### **6. Residual-Based View (Squared Error Case)**

For squared error loss:

$$
\ell = \frac{1}{2} (y_i - \hat{y}_i)^2
$$

* Gradient: $g_i = \hat{y}_i - y_i = -r_i$ (negative residual)
* Hessian: $h_i = 1$

Therefore:

$$
G = \sum g_i = -\sum r_i, \quad H = \sum h_i = n
$$

The node score becomes:

$$
\frac{G^2}{H + \lambda} = \frac{\left( \sum r_i \right)^2}{n + \lambda}
$$

This measures the **alignment of residuals** (sum of residuals squared), not their variance. If residuals have the same sign and large magnitude, this score is large, indicating a strong corrective signal.

---

### **7. Example: No Gain vs Positive Gain**

**No Gain Case (Homogeneous Node):**
All residuals equal, e.g. $r_i = -4$ for all samples.

* Parent node similarity score = high value.
* Children similarity scores sum to same value.
* Gain = 0 → splitting doesn’t improve model.

**Positive Gain Case (Heterogeneous Node):**
Residuals differ between groups, e.g., left residuals $-4$, right residuals $+1$.

* Parent similarity score is smaller.
* Children scores are larger when calculated separately.
* Gain > 0 → splitting improves predictions.

---

### **8. Gain as Similarity Score Change**

Interpret the node score

$$
\frac{G^2}{H + \lambda}
$$

as a **similarity score** reflecting how well samples agree in their gradient direction.

* The gain formula computes the **difference in similarity** before and after splitting.
* Higher gain means the children nodes are more homogeneous and easier to predict separately.
* Selecting splits maximizes gain, i.e., maximizes improvement in “internal agreement.”

---

### **9. Role in Tree Construction**

* Gain is calculated for every possible split candidate.
* The split with the highest positive gain (above $\gamma$) is selected.
* If no positive gain exceeds $\gamma$, the node is not split further.
* This process greedily grows the tree to minimize training loss.
* The gain formula unifies gradient and second-order info with regularization for robust splitting.

---