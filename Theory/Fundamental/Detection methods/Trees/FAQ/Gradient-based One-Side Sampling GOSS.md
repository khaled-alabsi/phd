# **Gradient-based One-Side Sampling (GOSS) in LightGBM**

#### **Its like smart sampling**
---

## **1. Problem It Solves**

In large datasets, training gradient boosting models becomes computationally expensive. Especially in the early trees of boosting, many training samples are **easy to predict** (i.e., have small gradients) and contribute **little** to model improvement. Processing all samples in every iteration wastes computation without improving performance.

---

## **2. Core Idea**

Prioritize **hard-to-predict samples** — those with **large gradients** — because they carry more learning signal. Discarding some easy (low-gradient) samples saves time, but to avoid **biasing gradient statistics**, GOSS applies a **scaling correction**.

---

## **3. Algorithm Overview**

Let:

* $N$: total number of training examples
* $a \in (0, 1)$: fraction of samples with **large gradients** to **keep entirely**
* $b \in (0, 1)$: fraction of samples with **small gradients** to **randomly sample**

**Steps:**

1. Compute gradient $g_i = \frac{\partial \mathcal{L}}{\partial \hat{y}_i}$ for each training sample.
2. Sort all samples by $|g_i|$ in descending order.
3. Keep top $aN$ samples (large gradients) unconditionally.
4. Randomly sample $bN$ from the remaining $(1 - a)N$ (small gradients).
5. To correct for under-representation, **scale the gradients and Hessians** of the small-gradient samples by:

$$
\text{scaling factor} = \frac{1 - a}{b}
$$

This ensures the overall distribution remains unbiased in split calculations.

---

## **4. Mathematical Details**

In gradient boosting trees, the **split gain** is calculated using the sum of gradients $G$ and Hessians $H$ as:

$$
\text{Gain} = \frac{1}{2} \left( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right)
$$

If low-gradient samples are downsampled, the sums $G$ and $H$ become biased. GOSS avoids this by **scaling them up**, maintaining an **unbiased estimate** of split quality.

---

## **5. Numerical Example**

Assume:

* $N = 100{,}000$ samples
* $a = 20\%$, $b = 10\%$

Then:

* Keep $20{,}000$ samples with the largest gradients.
* Randomly select $8{,}000$ samples from the remaining $80{,}000$.
* For the 8,000 small-gradient samples, scale gradients and Hessians by:

$$
\frac{1 - 0.2}{0.1} = 8
$$

This preserves the statistical contribution of those samples.

---

## **6. Why It Works**

* **Large gradients** correspond to **high loss** → samples not yet well-learned.
* **Small gradients** signal low loss → samples already well-modeled.
* Keeping all large-gradient samples ensures model focuses on "difficult" examples.
* Sampling and scaling small-gradient examples keeps computation low while preserving information balance.

---

## **7. Benefits**

* **Faster training**: smaller sample used per split.
* **Memory efficiency**: less data held in memory.
* **Accuracy preserved**: hard examples fully retained.
* **No statistical bias**: scaling maintains unbiased gradient estimation.

---

## **8. Reference**

The GOSS technique is introduced in:

**LightGBM: A Highly Efficient Gradient Boosting Decision Tree**, Section 3.3
[Link to paper](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

