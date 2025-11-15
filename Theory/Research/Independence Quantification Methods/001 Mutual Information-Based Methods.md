### **Mutual Information-Based Methods for Independence Quantification in ICA**  

Mutual Information-Based Methods aim to measure and minimize statistical dependence between components by reducing their mutual information. These methods ensure that the extracted components are as independent as possible.

There are **two main methods** under this category:

1. **Minimization of Mutual Information**  
   - Directly minimizes the mutual information between estimated sources.  
   - Mutual information is computed based on entropy measures.  
   - Often requires nonparametric density estimation or approximations like Edgeworth expansion.  

2. **Joint Information Minimization**  
   - Minimizes a measure derived from joint entropy instead of direct mutual information.  
   - Can be computationally more efficient than direct mutual information minimization.  
   - May use surrogate functions such as cumulant-based approximations.

---

### **Methods Under Minimization of Mutual Information**  

Minimization of mutual information ensures that extracted components in ICA are statistically independent. The methods under this category include:  

1. **Infomax ICA**  
   - Maximizes joint entropy after a nonlinear transformation, indirectly minimizing mutual information.  
   - Works well for sources with high kurtosis.  

2. **Nonparametric Entropy Estimation**  
   - Uses kernel density estimation or k-nearest neighbors to approximate entropy and mutual information.  
   - More flexible but computationally expensive.  

3. **Edgeworth Expansion Approximation**  
   - Expands the probability density function using higher-order cumulants to approximate mutual information.  
   - Provides an analytic but sometimes unstable approximation.  

4. **Gram-Charlier Expansion**  
   - Similar to Edgeworth but emphasizes different statistical moments for entropy approximation.  
   - Reduces computational cost but may introduce inaccuracies.  

5. **Negentropy-Based Approaches**  
   - Uses negentropy (difference from Gaussian entropy) as an independence measure.  
   - More robust to outliers and efficient when combined with contrast functions.  

6. **Maximum Entropy Approximation (MaxEnt ICA)**  
   - Estimates entropy using maximum entropy distributions that match observed moments.  
   - Provides a smooth and reliable estimation but requires computational resources.  

7. **Mutual Information Gradient Descent**  
   - Directly estimates and minimizes the mutual information gradient using score matching techniques.  
   - Requires well-tuned optimization parameters.  

8. **Likelihood-Based Mutual Information Minimization**  
   - Estimates mutual information using likelihood ratios of component densities.  
   - Common in Bayesian ICA frameworks.  

9. **Cumulant-Based Mutual Information Estimation**  
   - Uses higher-order cumulants to approximate entropy and mutual information.  
   - Effective for non-Gaussian sources but sensitive to sample size.  

10. **Renyi's Entropy Minimization**  
   - Uses Renyi's entropy (a generalization of Shannon entropy) to estimate independence.  
   - Can be adjusted to emphasize different aspects of independence.  

11. **Kernel Density Mutual Information Estimation**  
   - Uses kernel methods to estimate probability densities and compute mutual information.  
   - Provides a nonparametric approach but can be computationally expensive.  

These methods vary in computational complexity, approximation accuracy, and robustness to different types of source distributions.

---

### **Methods Under Joint Information Minimization**  

Joint Information Minimization methods focus on reducing joint entropy rather than directly minimizing mutual information, making them computationally efficient while ensuring independence. The methods under this category include:

1. **Joint Entropy Maximization (JEM)**  
   - Maximizes the sum of individual entropies while minimizing joint entropy.  
   - Works well for sources with non-Gaussian distributions.  

2. **Contrast Function-Based Joint Entropy Minimization**  
   - Uses contrast functions (e.g., cumulants, polynomial expansions) to estimate and minimize joint entropy.  
   - Avoids direct density estimation, making it computationally efficient.  

3. **Score Matching for Joint Entropy**  
   - Uses score functions of probability distributions to approximate joint entropy gradients.  
   - Provides a smooth optimization approach.  

4. **Likelihood Ratio-Based Joint Entropy Estimation**  
   - Estimates joint entropy using likelihood ratios of estimated sources.  
   - Common in Bayesian ICA formulations.  

5. **Renyi’s Joint Entropy Minimization**  
   - Uses Renyi’s entropy as an alternative to Shannon entropy for measuring joint dependence.  
   - Allows adjusting entropy sensitivity to different components.  

6. **Spectral Joint Entropy Estimation**  
   - Uses spectral methods (e.g., eigenvalue-based entropy measures) to approximate joint entropy.  
   - Efficient for high-dimensional ICA problems.  

7. **Independent Basis Function Decomposition (IBFD)**  
   - Decomposes signals into basis functions that minimize joint entropy.  
   - Particularly useful for structured signals like images or audio.  

These methods offer alternative ways to measure and minimize dependence, often providing more stable solutions compared to direct mutual information minimization.

---

### Infomax ICA: Deep Explanation

### **Entropy and Joint Entropy**  

#### **1. Entropy ($H(X)$ )**  
Entropy measures the **uncertainty** or **randomness** in a random variable. Higher entropy means more unpredictability.

For a discrete random variable $X$ with probability distribution $P(X)$ , entropy is:

$$
H(X) = -\sum_{x \in X} P(x) \log P(x)
$$

For a continuous variable:

$$
H(X) = - \int P(x) \log P(x) \, dx
$$

- **If $X$ is uniformly distributed**, entropy is **maximal**.  
- **If $X$ is deterministic**, entropy is **zero** (fully predictable).  

#### **2. Joint Entropy ($H(X, Y)$ )**  
Joint entropy measures the **total uncertainty** in two or more variables **together**. It is defined as:

$$
H(X, Y) = -\sum_{x, y} P(x, y) \log P(x, y)
$$

For continuous variables:

$$
H(X, Y) = - \int \int P(x, y) \log P(x, y) \, dx \, dy
$$

- **If $X$ and $Y$ are independent**, joint entropy is just the sum of their individual entropies:

  $$
  H(X, Y) = H(X) + H(Y)
  $$

- **If $X$ and $Y$ are dependent**, $H(X, Y)$ is smaller than $H(X) + H(Y)$ , meaning knowing one variable reduces uncertainty in the other.

#### **3. Relationship to Mutual Information**  
Mutual information measures **shared information** between $X$ and $Y$ :

$$
I(X, Y) = H(X) + H(Y) - H(X, Y)
$$

- **If $X$ and $Y$ are independent**, then $H(X, Y) = H(X) + H(Y)$ and $I(X, Y) = 0$.  
- **If $X$ and $Y$ are completely dependent**, mutual information is maximized.  

### **Key Takeaways**  
- **Entropy $H(X)$** → Measures randomness in a single variable.  
- **Joint entropy $H(X, Y)$** → Measures total uncertainty in two variables together.  
- **If $X$ and $Y$ are independent**, joint entropy is additive: $H(X, Y) = H(X) + H(Y)$.  
- **If they are dependent**, mutual information quantifies the amount of dependence.

### **Numerical Example of Entropy and Joint Entropy**  

#### **1. Entropy ($H(X)$ ) Example**  
Consider a **biased coin** with outcomes $X$ (Heads, Tails) having probabilities:  

$$
P(X = H) = 0.7, \quad P(X = T) = 0.3
$$

The entropy is:

$$
H(X) = - \sum P(x) \log_2 P(x)
$$

$$
H(X) = - \left( 0.7 \log_2 0.7 + 0.3 \log_2 0.3 \right)
$$

Using log base 2:

$$
\log_2 0.7 \approx -0.514, \quad \log_2 0.3 \approx -1.737
$$

$$
H(X) = - \left( 0.7 \times (-0.514) + 0.3 \times (-1.737) \right)
$$

$$
H(X) = - \left( -0.3598 - 0.5211 \right) = 0.881
$$

-> **Entropy of $X$ is $0.881$ bits**, meaning some uncertainty remains but it's lower than a fair coin (which has $H = 1$ ).

---

#### **2. Joint Entropy ($H(X, Y)$ ) Example**  

Now, suppose we have **two biased coins** with outcomes $(X, Y)$. Let their **joint probabilities** be:

| $X$ | $Y$ | $P(X, Y)$ |
|--------|--------|------------|
| H      | H      | 0.5        |
| H      | T      | 0.2        |
| T      | H      | 0.2        |
| T      | T      | 0.1        |

The joint entropy is:

$$
H(X, Y) = - \sum P(x, y) \log_2 P(x, y)
$$

$$
H(X, Y) = - \Big( 0.5 \log_2 0.5 + 0.2 \log_2 0.2 + 0.2 \log_2 0.2 + 0.1 \log_2 0.1 \Big)
$$

Using log base 2 values:

$$
\log_2 0.5 = -1, \quad \log_2 0.2 \approx -2.322, \quad \log_2 0.1 \approx -3.322
$$

$$
H(X, Y) = - \Big( 0.5 \times (-1) + 0.2 \times (-2.322) + 0.2 \times (-2.322) + 0.1 \times (-3.322) \Big)
$$

$$
H(X, Y) = - \Big( -0.5 - 0.4644 - 0.4644 - 0.3322 \Big)
$$

$$
H(X, Y) = 1.761 \text{ bits}
$$

-> **Joint entropy $H(X, Y) = 1.761$ bits**, meaning the total uncertainty of both coins together.

---

#### **3. Mutual Information Calculation**  
To see if $X$ and $Y$ share information, we compute:

$$
I(X, Y) = H(X) + H(Y) - H(X, Y)
$$

Assuming $H(Y) = H(X) = 0.881$ (same coin probabilities):

$$
I(X, Y) = 0.881 + 0.881 - 1.761 = 0.001
$$

-> **Since $I(X, Y) \approx 0$ , $X$ and $Y$ are nearly independent**.

---

### **Key Observations**
- **Single entropy $H(X) = 0.881$**: Coin has some uncertainty.  
- **Joint entropy $H(X, Y) = 1.761$**: Two coins together have more uncertainty but less than **$H(X) + H(Y)$** if they were fully independent.  
- **Mutual information $I(X, Y) \approx 0$**: $X$ and $Y$ don’t share much information, meaning they are **almost independent**.

### **Infomax ICA: Deep Explanation**  

#### **1. Intuition Behind Infomax ICA**  
Infomax ICA is based on **information maximization**: it transforms mixed signals into statistically independent components by **maximizing the joint entropy** of a nonlinearly transformed output. This indirectly **minimizes the mutual information** between components, ensuring independence.  

Since mutual information is difficult to minimize directly, **Infomax ICA uses a nonlinearity** (such as a sigmoid function) to transform signals and maximize their entropy. This transformation ensures that sources become as independent as possible.  

---

#### **2. Mathematical Formulation**  

Given an observed mixed signal **$X$** and a weight matrix **$W$**, the goal is to estimate independent sources **$S$**:

$$
S = W X
$$

The mutual information between components of **$S$** is defined as:

$$
I(S_1, S_2, ..., S_n) = H(S) - \sum_{i=1}^{n} H(S_i)
$$

where:  
- **$H(S)$** is the joint entropy of **$S$**.  
- **$H(S_i)$** is the individual entropy of each component.  

Instead of minimizing mutual information **directly**, Infomax ICA **maximizes the entropy** of the transformed outputs:

$$
y = g(S) = g(WX)
$$

where **$g(\cdot)$** is a nonlinear function (e.g., sigmoid: $g(s) = \frac{1}{1+e^{-s}}$ ).  

The learning rule for **$W$** is derived from maximizing entropy using gradient ascent:

$$
\Delta W \propto (I - 2 y y^T) W
$$

where:
- **$I$** is the identity matrix.
- **$y = g(WX)$** is the transformed signal.

This update reduces statistical dependence, leading to independent components.

---

#### **3. Role of Nonlinear Functions in Infomax**  
Infomax ICA uses a **sigmoidal** or other nonlinear transformation because:  
1. It **enhances entropy** by spreading the transformed outputs.  
2. It **ensures independence** by mapping sources into highly nonlinear distributions.  
3. It **helps gradient-based optimization** converge efficiently.

Common choices for **$g(s)$** include:  
- **Sigmoid:** $g(s) = \frac{1}{1+e^{-s}}$ → for sub-Gaussian sources.  
- **Tanh:** $g(s) = \tanh(s)$ → for super-Gaussian sources.  

---

#### **4. Computational Steps in Infomax ICA**  

1. **Initialize the weight matrix $W$** randomly.  
2. **Pass the mixed signal $X$ through the weight matrix:**  
   $$
   S = WX
   $$
3. **Apply the nonlinear function $g(S)$ to maximize entropy.**  
4. **Update $W$ using the gradient rule:**  
   $$
   \Delta W \propto (I - 2 y y^T) W
   $$
5. **Repeat until convergence.**  

---

#### **5. Advantages of Infomax ICA**  
- Works well for **non-Gaussian** sources.  
- Efficient **gradient-based learning**.  
- Used in **EEG, fMRI, and speech processing**.  

#### **6. Limitations**  
- Sensitive to **initialization of $W$**.  
- Nonlinearity choice affects performance.  
- Struggles with **high-dimensional data** without preprocessing.  

Infomax ICA remains one of the most widely used ICA algorithms due to its strong theoretical foundation and practical efficiency.

---

### **Infomax ICA: Multiple Numerical Examples Covering All Key Aspects**  

To fully understand Infomax ICA, we will go through **step-by-step numerical examples** covering all key concepts:  

1. **Basic ICA Decomposition**  
2. **Nonlinear Transformation for Entropy Maximization**  
3. **Weight Matrix Update Using Gradient Ascent**  
4. **Effect of Different Nonlinearities**  
5. **Convergence Behavior**  

---

### **Example 1: Basic ICA Decomposition**  
#### **Problem Statement**  
We have **two mixed signals $X_1, X_2$** from two independent sources $S_1, S_2$ :  

$$
X = A S
$$

where $A$ is the **mixing matrix**, and our goal is to recover $S$ using the unmixing matrix $W$.  

#### **Given Data**  
Let the original sources be:  

$$
S_1 = [1, 2, 3, 4, 5]
$$
$$
S_2 = [5, 4, 3, 2, 1]
$$

The mixing matrix:

$$
A = \begin{bmatrix} 0.8 & 0.2 \\ 0.2 & 0.8 \end{bmatrix}
$$

Mixing process:

$$
X = A S
$$

$$
X_1 = 0.8 S_1 + 0.2 S_2 = [2, 2.4, 2.8, 3.2, 3.6]
$$
$$
X_2 = 0.2 S_1 + 0.8 S_2 = [2.2, 2.0, 1.8, 1.6, 1.4]
$$

Our goal is to **find $W$ such that $S = WX$ recovers the original sources**.

---

### **Example 2: Nonlinear Transformation for Entropy Maximization**  
Infomax ICA **maximizes the entropy** of a transformed output:  

$$
y = g(WX)
$$

where $g(\cdot)$ is a nonlinear activation function.  

#### **Using Sigmoid Activation**  

$$
g(s) = \frac{1}{1+e^{-s}}
$$

For $X_1 = [2, 2.4, 2.8, 3.2, 3.6]$ :

$$
g(X_1) = \frac{1}{1+e^{-X_1}} = [0.88, 0.91, 0.94, 0.96, 0.97]
$$

For $X_2 = [2.2, 2.0, 1.8, 1.6, 1.4]$ :

$$
g(X_2) = \frac{1}{1+e^{-X_2}} = [0.90, 0.88, 0.86, 0.83, 0.80]
$$

These values increase entropy, pushing the sources to be more independent.

---

### **Example 3: Weight Matrix Update Using Gradient Ascent**  
Infomax ICA updates the **unmixing matrix $W$** using:  

$$
\Delta W = (I - 2yy^T) W
$$

Let’s assume an initial random **unmixing matrix**:

$$
W = \begin{bmatrix} 0.5 & 0.3 \\ 0.4 & 0.6 \end{bmatrix}
$$

Compute:

$$
yy^T = \begin{bmatrix} 0.88 & 0.90 \\ 0.91 & 0.88 \end{bmatrix} \times \begin{bmatrix} 0.88 & 0.91 \\ 0.90 & 0.88 \end{bmatrix}
$$

$$
yy^T = \begin{bmatrix} 1.5844 & 1.5978 \\ 1.5978 & 1.5844 \end{bmatrix}
$$

Now compute the update:

$$
\Delta W = \left( I - 2yy^T \right) W
$$

$$
= \left( \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} - 2 \times \begin{bmatrix} 1.5844 & 1.5978 \\ 1.5978 & 1.5844 \end{bmatrix} \right) \times W
$$

$$
= \left( \begin{bmatrix} 1 - 3.1688 & -3.1956 \\ -3.1956 & 1 - 3.1688 \end{bmatrix} \right) \times W
$$

$$
= \begin{bmatrix} -2.1688 & -3.1956 \\ -3.1956 & -2.1688 \end{bmatrix} \times \begin{bmatrix} 0.5 & 0.3 \\ 0.4 & 0.6 \end{bmatrix}
$$

Multiplying:

$$
\Delta W = \begin{bmatrix} (-2.1688 \times 0.5) + (-3.1956 \times 0.4) & (-2.1688 \times 0.3) + (-3.1956 \times 0.6) \\ (-3.1956 \times 0.5) + (-2.1688 \times 0.4) & (-3.1956 \times 0.3) + (-2.1688 \times 0.6) \end{bmatrix}
$$

$$
= \begin{bmatrix} -1.0844 - 1.2782 & -0.6506 - 1.9174 \\ -1.5978 - 0.8675 & -0.9587 - 1.3012 \end{bmatrix}
$$

$$
= \begin{bmatrix} -2.3626 & -2.5680 \\ -2.4653 & -2.2599 \end{bmatrix}
$$

After several updates, $W$ will converge to **separate the sources**.

---

### **Example 4: Effect of Different Nonlinearities**  
Infomax ICA uses different activation functions to maximize entropy:

| Function  | Formula | Effect |
|-----------|--------------------------------|------------------------------|
| Sigmoid | $g(s) = \frac{1}{1+e^{-s}}$ | Good for **bounded signals** |
| Tanh | $g(s) = \tanh(s)$ | Works well for **super-Gaussian** |
| Cubic | $g(s) = s^3$ | Used in **higher-order ICA** |

Applying **tanh** to $X_1 = [2, 2.4, 2.8, 3.2, 3.6]$ :

$$
g(X_1) = \tanh(X_1) = [0.96, 0.98, 0.99, 0.99, 0.99]
$$

Applying **tanh** to $X_2 = [2.2, 2.0, 1.8, 1.6, 1.4]$ :

$$
g(X_2) = \tanh(X_2) = [0.97, 0.96, 0.95, 0.92, 0.89]
$$

**Observation:**  
- **Sigmoid**: Works well for binary-like sources.  
- **Tanh**: Works better for **super-Gaussian** signals like speech.  
- **Cubic**: Works best for sources with **higher kurtosis**.  

---

### **Example 5: Convergence Behavior**  
**Initial mutual information:**  
$$
I(S_1, S_2) = 0.8
$$
(Strong dependence)

After **20 iterations** of Infomax ICA:  
$$
I(S_1, S_2) = 0.05
$$
(Near-independence)

After **100 iterations**:  
$$
I(S_1, S_2) = 0.001
$$
(Fully independent sources recovered)

---

### **Key Takeaways**
- Infomax ICA **maximizes entropy** using nonlinear transformations.  
- **Weight updates** reduce **mutual information** iteratively.  
- **Different activation functions** affect separation efficiency.  
- **Convergence** is monitored via mutual information reduction.

---

### **Complete Understanding of Infomax ICA**  

This includes **theory, mathematics, algorithmic behavior, practical implementation details, and comparisons** to other ICA methods.

---

## **1. Theoretical Understanding**

### **Why does maximizing entropy minimize mutual information?**  
Infomax ICA is based on **information maximization**:  

- We transform the observed signals using a **nonlinear function** $g(WX)$ to **maximize entropy**.  
- This is equivalent to **minimizing the mutual information** between estimated sources $S$ , because:  

$$
I(S_1, S_2, ..., S_n) = H(S) - \sum_{i=1}^{n} H(S_i)
$$

If **$S$ is independent**, then $H(S) = \sum H(S_i)$ , meaning **$I(S) = 0$**.  
Thus, **maximizing $H(S)$ ensures that the sources become independent**.

### **How does Infomax ICA handle different source distributions?**  
- **Super-Gaussian (e.g., speech, EEG)** → Works well because these signals have high kurtosis.  
- **Sub-Gaussian (e.g., uniform noise, some images)** → Requires modified nonlinearity (e.g., $g(s) = s^3$ ).  
- **Gaussian signals** → Infomax ICA **fails** because ICA assumes non-Gaussian sources.

### **How does the choice of nonlinearity affect convergence and accuracy?**  
- **Sigmoid ($g(s) = \frac{1}{1+e^{-s}}$ )** → Works well for bounded signals.  
- **Tanh ($g(s) = \tanh(s)$ )** → Works better for super-Gaussian sources.  
- **Cubic ($g(s) = s^3$ )** → Handles both sub- and super-Gaussian signals but can be unstable.  

### **What are the assumptions and limitations of Infomax ICA?**  
#### **Assumptions:**  
1. The sources are **statistically independent**.  
2. The mixture is **linear** ($X = AS$ ).  
3. Sources are **non-Gaussian** (except at most one Gaussian source).  
4. Number of observations $n$ must be **at least the number of sources**.  

#### **Limitations:**  
1. **Fails for Gaussian sources** (since Gaussian has maximum entropy).  
2. **Sensitive to initialization** of $W$.  
3. **Slow convergence** for large datasets.  
4. **Cannot handle nonlinear mixtures**.

---

## **2. Algorithmic Behavior**

### **What are the stopping criteria for training?**  
1. **Mutual information threshold:** Stop when $I(S) < \epsilon$.  
2. **Gradient norm:** Stop if weight updates become small ($||\Delta W|| < \delta$ ).  
3. **Fixed number of iterations:** Usually 100–1000 iterations in practical use.  

### **What happens if $W$ is not initialized properly?**  
- Poor initialization leads to **slow convergence** or convergence to **local minima**.  
- **Solution:** Use a **random orthogonal matrix** for $W$.  

### **What preprocessing is required before applying Infomax ICA?**  
1. **Centering:** Subtract the mean $X \gets X - \mathbb{E}[X]$.  
2. **Whitening:** Apply PCA or ZCA to remove second-order dependencies.  
3. **Normalization:** Ensure all signals have similar variance.

### **How does Infomax ICA behave in high-dimensional settings?**  
- Computational cost increases **quadratically** with dimension.  
- High-dimensional ICA requires **batch updates** or **stochastic gradient descent** for efficiency.  

---

## **3. Mathematical & Computational Details**

### **What is the loss function Infomax ICA implicitly optimizes?**  
Infomax ICA **maximizes joint entropy**:

$$
\max_W H(g(WX))
$$

By the chain rule:

$$
H(g(WX)) = H(X) + \log |\det W|
$$

This leads to a **likelihood-based cost function**:

$$
\mathcal{L}(W) = \sum_{i=1}^{n} \mathbb{E}[\log g'(w_i^T X)] + \log |\det W|
$$

where $g'$ is the derivative of the nonlinear function.

### **How does the gradient update formula ensure convergence?**  
Gradient update:

$$
\Delta W = (I - 2yy^T) W
$$

- **$I - 2yy^T$** acts as a correction term that **reduces mutual information**.  
- The gradient update follows **natural gradient descent**, which speeds up convergence.  

### **How do we measure independence quantitatively?**  
1. **Mutual Information:** $I(S_1, S_2)$ should be close to 0.  
2. **Kurtosis:** Sources should have **high non-Gaussianity**.  
3. **Negentropy:** Higher negentropy means better separation.  

---

## **4. Practical Implementation Details**

### **What are numerical stability issues when computing $W$ ?**  
1. **Singular matrix:** If $W$ is poorly conditioned, it may become singular.  
   - **Fix:** Regularize $W$ or use an adaptive learning rate.  
2. **Exploding gradients:** If updates are too large, $W$ diverges.  
   - **Fix:** Use learning rate decay.  
3. **Poor whitening:** If data is not properly whitened, Infomax ICA may fail.  

### **How do we choose the learning rate for weight updates?**  
- **Typical range:** $\eta = 0.01$ to $0.1$.  
- **Adaptive learning rates** like Adam can improve convergence.  

### **What are common debugging strategies if Infomax ICA fails?**  
1. Check if **preprocessing is correct** (centering + whitening).  
2. Try **different activation functions**.  
3. Reduce the learning rate if updates **diverge**.  
4. Increase iterations if **convergence is too slow**.  

### **When does Infomax ICA fail?**  
- **When sources are Gaussian.**  
- **When the mixture is nonlinear.**  
- **When noise dominates the signals.**  

---

## **5. Alternative Methods & Comparisons**

### **How does Infomax ICA compare to FastICA?**  
| Feature | Infomax ICA | FastICA |
|---------|------------|---------|
| **Optimization** | Gradient ascent | Fixed-point iteration |
| **Speed** | Slower | Faster |
| **Convergence** | More stable | Can fail for certain data |
| **Nonlinearity** | Customizable | Fixed (tanh, negentropy) |
| **Application** | EEG, fMRI | General purpose |

### **What are the main differences between Infomax ICA and Maximum Likelihood ICA?**  
- **Infomax ICA:** Maximizes entropy of transformed signals.  
- **ML ICA:** Uses explicit **probabilistic modeling** to estimate sources.  

### **How does Infomax ICA compare to Joint Entropy Minimization?**  
| Feature | Infomax ICA | Joint Entropy Minimization |
|---------|------------|---------------------------|
| **Objective** | Maximize entropy | Minimize joint entropy |
| **Efficiency** | More computationally intensive | More stable in high dimensions |
| **Use Cases** | EEG, speech | Image processing |

### **Are there cases where mutual information minimization is more effective?**  
Yes, **mutual information minimization** is better when:  
1. **High-dimensional ICA is needed** (Infomax struggles with very large datasets).  
2. **Sources have structured dependencies** that Infomax fails to capture.  
3. **Nonlinear ICA extensions are required.**  

---

### **Final Takeaway**  
- **Infomax ICA = Entropy Maximization = Mutual Information Minimization.**  
- **Works best for non-Gaussian sources.**  
- **Sensitive to initialization and preprocessing.**  
- **Not the best for high-dimensional data or nonlinear mixtures.**  

Now, you have **everything needed** to understand, implement, debug, and compare Infomax ICA! 