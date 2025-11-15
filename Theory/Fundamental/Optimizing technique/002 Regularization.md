### **Regularization** is a technique in machine learning to prevent **overfitting** by adding a penalty term to the loss function of a model. It discourages overly complex models that fit the training data too well but fail to generalize to new data.  

### Key Concepts:
1. **Overfitting**: 
   - Happens when a model learns noise or small fluctuations in the training data instead of the true underlying patterns.
   - Results in low training error but high test error.

2. **Loss Function**: 
   - Regularization modifies the loss function by adding a penalty term that is based on the complexity of the model. 
   - General form:  
     $$
     \text{Regularized Loss} = \text{Original Loss} + \lambda \cdot \text{Penalty Term}
     $$  
     - $\lambda$ : Regularization strength (hyperparameter).

---

### Types of Regularization:
#### 1. **L1 Regularization (Lasso)**:  
   - Penalty: Sum of absolute values of model parameters ($\sum |w_i|$ ).
   - Encourages sparsity by driving some weights to **exactly zero**, making it useful for feature selection.

#### 2. **L2 Regularization (Ridge)**:  
   - Penalty: Sum of squared values of model parameters ($\sum w_i^2$ ).
   - Shrinks all weights closer to zero but does not set them exactly to zero, maintaining all features.

#### 3. **Elastic Net**:  
   - Combines L1 and L2 regularization:
     $$
     \text{Penalty Term} = \alpha \cdot \sum |w_i| + (1 - \alpha) \cdot \sum w_i^2
     $$
   - Balances sparsity and smoothness.

---

### Effect on Models:
- **Linear Models**: Prevent overfitting by constraining weights.
- **Neural Networks**:
  - Use techniques like **weight decay** (similar to L2 regularization).
  - Apply **Dropout**: Randomly disable neurons during training to reduce dependency on specific neurons.

### Choosing $\lambda$ :  
- Determined through **cross-validation** to balance bias and variance.

### Regularization with Numerical Example

Let’s illustrate **L2 Regularization** (Ridge) and **L1 Regularization** (Lasso) using a simple regression example.

#### Problem Setup:
We have the following data:
- Feature matrix $X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 2 & 2 \\ 2 & 3 \end{bmatrix}$
- Target $y = \begin{bmatrix} 1 \\ 2 \\ 2 \\ 3 \end{bmatrix}$

The objective is to minimize the **regularized loss function**:
$$
J(w) = \frac{1}{2n} \| Xw - y \|^2 + \lambda \cdot \text{Penalty}(w)
$$
where:
- $w = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}$ (weights)
- $n = 4$ (number of samples)
- $\lambda$ = 0.1 (regularization strength)

---

### Step 1: Unregularized Loss
The unregularized loss function is:
$$
J(w) = \frac{1}{2n} \| Xw - y \|^2
$$

Compute $Xw - y$ :
$$
Xw = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 2 & 2 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} w_1 \\ w_2 \end{bmatrix} = \begin{bmatrix} w_1 + w_2 \\ w_1 + 2w_2 \\ 2w_1 + 2w_2 \\ 2w_1 + 3w_2 \end{bmatrix}
$$

$$
Xw - y = \begin{bmatrix} w_1 + w_2 - 1 \\ w_1 + 2w_2 - 2 \\ 2w_1 + 2w_2 - 2 \\ 2w_1 + 3w_2 - 3 \end{bmatrix}
$$

Square the residuals:
$$
\| Xw - y \|^2 = (w_1 + w_2 - 1)^2 + (w_1 + 2w_2 - 2)^2 + (2w_1 + 2w_2 - 2)^2 + (2w_1 + 3w_2 - 3)^2
$$

Take the average:
$$
\frac{1}{2n} \| Xw - y \|^2 = \frac{1}{8} [(w_1 + w_2 - 1)^2 + (w_1 + 2w_2 - 2)^2 + (2w_1 + 2w_2 - 2)^2 + (2w_1 + 3w_2 - 3)^2]
$$

---

### Step 2: Adding L2 Regularization
The regularized loss for **Ridge Regression** is:
$$
J(w) = \frac{1}{2n} \| Xw - y \|^2 + \lambda \cdot \|w\|^2
$$

Substitute the penalty term:
$$
\|w\|^2 = w_1^2 + w_2^2
$$

The new objective becomes:
$$
J(w) = \frac{1}{8} [(w_1 + w_2 - 1)^2 + (w_1 + 2w_2 - 2)^2 + (2w_1 + 2w_2 - 2)^2 + (2w_1 + 3w_2 - 3)^2] + 0.1 (w_1^2 + w_2^2)
$$

**Step-by-Step Evaluation**:
1. Compute the squared residuals and add them.
2. Add $0.1 \cdot (w_1^2 + w_2^2)$ to penalize large weights.

---

### Step 3: Adding L1 Regularization
The regularized loss for **Lasso Regression** is:
$$
J(w) = \frac{1}{2n} \| Xw - y \|^2 + \lambda \cdot \|w\|_1
$$

Substitute the penalty term:
$$
\|w\|_1 = |w_1| + |w_2|
$$

The new objective becomes:
$$
J(w) = \frac{1}{8} [(w_1 + w_2 - 1)^2 + (w_1 + 2w_2 - 2)^2 + (2w_1 + 2w_2 - 2)^2 + (2w_1 + 3w_2 - 3)^2] + 0.1 (|w_1| + |w_2|)
$$

---

### Comparing Ridge and Lasso Effects:
1. **Ridge**:
   - Shrinks weights continuously but does not force them to zero.
   - Ensures all features contribute (no sparsity).

2. **Lasso**:
   - Encourages sparsity by driving some weights to exactly zero.
   - Useful for feature selection.

### Sparse PCA with Regularization: Numerical Example

Sparse PCA (Principal Component Analysis) extends standard PCA by adding an **L1 regularization** term to enforce sparsity in the principal components. This ensures that some component loadings are set to zero, leading to more interpretable components.

#### Problem Setup:
We have a data matrix $X$ (centered, assumed):
$$
X = \begin{bmatrix} 
2 & 4 & 1 \\ 
3 & 6 & 2 \\ 
4 & 8 & 3 
\end{bmatrix}
$$
- $X$ has 3 observations and 3 features.
- Goal: Extract sparse principal components.

---

### Step 1: Compute the Covariance Matrix
The covariance matrix is:
$$
C = \frac{1}{n} X^T X
$$
where $n = 3$ (number of observations).

Compute $X^T X$ :
$$
X^T = \begin{bmatrix} 
2 & 3 & 4 \\ 
4 & 6 & 8 \\ 
1 & 2 & 3 
\end{bmatrix}, \quad 
X^T X = \begin{bmatrix}
29 & 58 & 20 \\
58 & 116 & 40 \\
20 & 40 & 14
\end{bmatrix}.
$$

Normalize by $n = 3$ :
$$
C = \frac{1}{3} \begin{bmatrix}
29 & 58 & 20 \\
58 & 116 & 40 \\
20 & 40 & 14
\end{bmatrix} = 
\begin{bmatrix}
9.67 & 19.33 & 6.67 \\
19.33 & 38.67 & 13.33 \\
6.67 & 13.33 & 4.67
\end{bmatrix}.
$$

---

### Step 2: Sparse PCA Objective
Sparse PCA solves:
$$
\max_w \quad w^T C w \quad \text{subject to} \quad \|w\|_2 \leq 1, \; \|w\|_1 \leq \lambda
$$
- $w$ : Loadings (principal component direction).
- $\|w\|_2$ : L2-norm constraint (standard PCA normalization).
- $\|w\|_1$ : L1-norm sparsity constraint.

---

### Step 3: Adding Regularization
1. Without regularization ($\lambda = \infty$ ):
   - Solve the standard PCA eigenvalue problem $Cw = \lambda w$.
   - First eigenvector (principal component): $w = [0.707, 0.707, 0]$.

2. With sparsity ($\lambda = 1.5$ ):
   - Introduce L1 penalty $\|w\|_1 = |w_1| + |w_2| + |w_3| \leq 1.5$.
   - Enforce sparsity: Manually adjust $w$ to satisfy $\|w\|_1$ while maximizing $w^T C w$.

---

### Step 4: Numerical Approximation
Suppose we initialize $w = [0.5, 0.5, 0.5]$ :
1. Compute $w^T C w$ :
   $$
   w^T C w = \begin{bmatrix} 0.5 & 0.5 & 0.5 \end{bmatrix}
   \begin{bmatrix}
   9.67 & 19.33 & 6.67 \\
   19.33 & 38.67 & 13.33 \\
   6.67 & 13.33 & 4.67
   \end{bmatrix}
   \begin{bmatrix} 0.5 \\ 0.5 \\ 0.5 \end{bmatrix}.
   $$

2. Apply L1 constraint by zeroing smallest loadings:
   - Set $w_3 = 0$ (smallest loading).
   - Normalize remaining $w = [0.707, 0.707, 0]$.

---

### Final Sparse Principal Component
With $\lambda = 1.5$ , the sparse principal component is:
$$
w = [0.707, 0.707, 0].
$$
This zeroes out the third feature, resulting in a sparse principal component.

```python
import numpy as np
from sklearn.decomposition import SparsePCA

# Data matrix X (centered)
X = np.array([
    [2, 4, 1],
    [3, 6, 2],
    [4, 8, 3]
])

# Initialize Sparse PCA
sparse_pca = SparsePCA(n_components=2, alpha=1.0, random_state=42)

# Fit the Sparse PCA model
sparse_pca.fit(X)

# Extract sparse components and transform the data
components = sparse_pca.components_
transformed_data = sparse_pca.transform(X)

print("Sparse Components (Principal Directions):")
print(components)
print("\nTransformed Data (Projection onto Sparse Components):")
print(transformed_data)
```

The results of the Sparse PCA are as follows:

### Sparse Components (Principal Directions):
$$
\text{Component 1: } [0.2157, 0.9523, 0.2157] \\
\text{Component 2: } [0, 0, 0] \quad \text{(indicating sparsity)}
$$

### Transformed Data (Projection onto Sparse Components):
$$
\begin{bmatrix}
-2.313 & 0 \\
0 & 0 \\
2.313 & 0
\end{bmatrix}
$$

This shows the first principal component captures the variance in the data, while the second is entirely sparse.