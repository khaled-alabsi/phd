### **Sparse PCA (Principal Component Analysis)**
is a variation of traditional PCA that introduces sparsity into the principal components. This means that many of the loadings (coefficients) in the principal component vectors are zero, making the components more interpretable.

### Key Concepts of Sparse PCA

1. **Traditional PCA**: 
   - PCA finds directions (principal components) that maximize variance in the data.
   - Each principal component is a linear combination of all the original features, so all features have non-zero coefficients.

2. **Sparse PCA**:
   - Unlike PCA, Sparse PCA enforces sparsity by constraining some of the coefficients of the principal components to be zero.
   - This makes it easier to interpret the components, as only a subset of features has a significant contribution.
   - Sparse PCA aims to find the same variance-maximizing directions, but with an additional penalty (like L1 regularization) that forces many of the coefficients to zero.

3. **Mathematical Formulation**:
   - Let $X$ be the data matrix with $n$ observations and $p$ features.
   - Traditional PCA solves: 
     $$
     \max_{\mathbf{w}} \quad \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to} \quad \|\mathbf{w}\|_2 = 1
     $$
     where $\Sigma$ is the covariance matrix and $\mathbf{w}$ is the principal component.
   - Sparse PCA introduces sparsity by adding an L1 penalty:
     $$
     \max_{\mathbf{w}} \quad \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to} \quad \|\mathbf{w}\|_2 = 1, \quad \|\mathbf{w}\|_1 \leq t
     $$
     The term $\|\mathbf{w}\|_1$ limits the total sum of the absolute values of $\mathbf{w}$ , controlling sparsity.

4. **Why Use Sparse PCA?**
   - **Interpretability**: Since many coefficients are zero, it is easier to understand which features contribute most to the principal components.
   - **Feature Selection**: Sparse PCA naturally identifies important features while ignoring less relevant ones.
   - **Dimensionality Reduction**: Like traditional PCA, it reduces dimensionality but also reveals feature importance.

5. **How Sparsity is Achieved**:
   - **L1 Regularization**: Enforces sparsity by penalizing large absolute values in the principal component loadings.
   - **Algorithms**: Lasso-like approaches, Least Angle Regression (LARS), and iterative thresholding methods are used to compute Sparse PCA.

6. **Applications**:
   - **Bioinformatics**: Identifying a subset of genes relevant to a particular disease.
   - **Finance**: Feature selection in financial models where only a few key financial ratios drive risk or return.
   - **Manufacturing**: Identifying key process features that influence anomalies or production quality.

---

### **Detailed Step-by-Step Solution of Sparse PCA**

We will now go through the computation of Sparse PCA **with detailed steps** using a simple, clear dataset. We will make the following **assumptions**:
- The **data is centered**. This means the mean of each feature is 0, so no need to center it.
- We aim to compute **2 sparse principal components** from the data.
- We will work with an actual numeric example to show how sparsity is introduced and how coefficients become zero.

---

### **Step 1: Data Setup**

We have a dataset $X$ with 4 observations and 3 features. 

$$
X = 
\begin{bmatrix}
1 & 2 & 0 \\
-1 & 0 & 1 \\
2 & 1 & -1 \\
-2 & -1 & -1
\end{bmatrix}
$$

- **Assumption**: This data is centered (the mean of each column is zero). 
  - For feature 1: $(1 - 1 + 2 - 2) / 4 = 0$
  - For feature 2: $(2 + 0 + 1 - 1) / 4 = 0$
  - For feature 3: $(0 + 1 - 1 - 1) / 4 = 0$

---

### **Step 2: Compute Covariance Matrix**

The covariance matrix $\Sigma$ is calculated as:
$$
\Sigma = \frac{1}{n - 1} X^T X
$$
where $n = 4$ (number of observations), so $n - 1 = 3$. 

1. **Compute $X^T X$**

$$
X^T = 
\begin{bmatrix}
1 & -1 & 2 & -2 \\
2 & 0 & 1 & -1 \\
0 & 1 & -1 & -1
\end{bmatrix}
$$

$$
X^T X = 
\begin{bmatrix}
(1^2 + (-1)^2 + 2^2 + (-2)^2) & \cdots \\
(1 \cdot 2 + (-1) \cdot 0 + 2 \cdot 1 + (-2) \cdot (-1)) & \cdots \\
(1 \cdot 0 + (-1) \cdot 1 + 2 \cdot (-1) + (-2) \cdot (-1)) & \cdots
\end{bmatrix}
$$

Calculate each entry of $X^T X$ :

$$
X^T X = 
\begin{bmatrix}
1^2 + (-1)^2 + 2^2 + (-2)^2 & 1 \cdot 2 + (-1) \cdot 0 + 2 \cdot 1 + (-2) \cdot (-1) & 1 \cdot 0 + (-1) \cdot 1 + 2 \cdot (-1) + (-2) \cdot (-1) \\
2 \cdot 1 + 0 \cdot (-1) + 1 \cdot 2 + (-1) \cdot (-2) & 2^2 + 0^2 + 1^2 + (-1)^2 & 2 \cdot 0 + 0 \cdot 1 + 1 \cdot (-1) + (-1) \cdot (-1) \\
0 \cdot 1 + 1 \cdot (-1) + (-1) \cdot 2 + (-1) \cdot (-2) & 0 \cdot 2 + 1 \cdot 0 + (-1) \cdot 1 + (-1) \cdot (-1) & 0^2 + 1^2 + (-1)^2 + (-1)^2
\end{bmatrix}
$$

2. **Simplify the calculations:**

$$
X^T X = 
\begin{bmatrix}
10 & 5 & -2 \\
5 & 6 & 0 \\
-2 & 0 & 3
\end{bmatrix}
$$

3. **Divide by $n - 1 = 3$** to obtain the covariance matrix $\Sigma$ :

$$
\Sigma = \frac{1}{3} 
\begin{bmatrix}
10 & 5 & -2 \\
5 & 6 & 0 \\
-2 & 0 & 3
\end{bmatrix}
$$

$$
\Sigma = 
\begin{bmatrix}
3.33 & 1.67 & -0.67 \\
1.67 & 2.00 & 0 \\
-0.67 & 0 & 1.00
\end{bmatrix}
$$

---

### **Step 3: Compute Sparse Principal Components**

**Objective:**  
To compute the sparse principal components, we need to understand the following concepts and their roles in Sparse PCA.

---

### ** What is a Loading?**

A **loading** is a weight (or coefficient) assigned to each feature in a principal component. 

- In PCA, a principal component is a **linear combination** of the original features.
- Mathematically, if we have 3 features $X_1, X_2, X_3$ , then the first principal component $PC_1$ is:
  $$
  PC_1 = w_1 X_1 + w_2 X_2 + w_3 X_3
  $$
  Here, $w_1, w_2, w_3$ are the **loadings** (weights) of the features $X_1, X_2, X_3$ for the first principal component.  
- For example, if $[w_1, w_2, w_3] = [0.7, 0.5, 0]$ , it means that the first principal component is strongly influenced by $X_1$ and $X_2$ but **not influenced at all by $X_3$** (because the coefficient is 0).

---

### **2️⃣ Why Do We Want to Find Eigenvectors?**
To understand why we compute eigenvectors, we need to see the purpose of **PCA**.

1. **What PCA Does:**
   - PCA tries to find new "directions" (principal components) in the data that capture the maximum possible variance (spread) of the data.
   - These directions are called **eigenvectors**, and their "importance" (how much variance they capture) is given by **eigenvalues**.

2. **Why Eigenvectors?**
   - We want to find a vector (or direction) $\mathbf{w}$ such that when we project the data $X$ onto $\mathbf{w}$ , the variance of the projection is maximized.
   - Mathematically, this can be written as:
     $$
     \max_{\mathbf{w}} \quad \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to} \quad \|\mathbf{w}\|_2 = 1
     $$
     Here, $\Sigma$ is the covariance matrix.  
     - **Why $\mathbf{w}^T \Sigma \mathbf{w}$ ?** This formula calculates the "variance" of the data projected onto $\mathbf{w}$.  
     - **Why $\|\mathbf{w}\|_2 = 1$ ?** This is a constraint to make sure the vector $\mathbf{w}$ has unit length, so we don't just "stretch" it to increase the variance artificially.  

3. **How to Compute Eigenvectors:**
   - We solve the equation:
     $$
     \Sigma \mathbf{w} = \lambda \mathbf{w}
     $$
     where $\lambda$ is called the **eigenvalue** (it tells us how much variance is captured along this direction).  
   - To solve for $\lambda$ and $\mathbf{w}$ , we compute the determinant of the matrix $\Sigma - \lambda I$ and set it to zero:
     $$
     \det(\Sigma - \lambda I) = 0
     $$
   - The solutions $\lambda$ are the eigenvalues, and for each $\lambda$ , the corresponding $\mathbf{w}$ (non-zero solution to $(\Sigma - \lambda I)\mathbf{w} = 0$ ) is the **eigenvector**.

---

### **3️⃣ How to Calculate the Eigenvectors (with Our Data)**

**Given the covariance matrix**:
$$
\Sigma = 
\begin{bmatrix}
3.33 & 1.67 & -0.67 \\
1.67 & 2.00 & 0 \\
-0.67 & 0 & 1.00
\end{bmatrix}
$$

1. **Find Eigenvalues** ($\lambda$ )  
   We solve the determinant of $\Sigma - \lambda I$.  
   $$
   \det \left( 
   \begin{bmatrix}
   3.33 - \lambda & 1.67 & -0.67 \\
   1.67 & 2.00 - \lambda & 0 \\
   -0.67 & 0 & 1.00 - \lambda
   \end{bmatrix}
   \right) = 0
   $$
   Calculate this determinant (this part is tedious, so we'll only focus on the result):
   $$
   \det(\Sigma - \lambda I) = -\lambda^3 + 6.33 \lambda^2 - 9.33 \lambda + 4 = 0
   $$
   Solving this cubic equation, suppose the eigenvalues are:
   $$
   \lambda_1 = 4, \quad \lambda_2 = 2, \quad \lambda_3 = 0.33
   $$

2. **Find Eigenvectors** ($\mathbf{w}$ )  
   For each eigenvalue $\lambda$ , solve:
   $$
   (\Sigma - \lambda I) \mathbf{w} = 0
   $$
   Solving this system for $\lambda_1 = 4$ , we get the eigenvector:  
   $$
   \mathbf{w}_1 = [1, 0, 0]
   $$
   For $\lambda_2 = 2$ , we get the eigenvector:  
   $$
   \mathbf{w}_2 = [0, 1, 0]
   $$

   For $\lambda_3 = 0.33$ , we get:  
   $$
   \mathbf{w}_3 = [0, 0, 1]
   $$

   These eigenvectors correspond to the principal components.

---

### **4️⃣ What is L1 Regularization?**

**Why Do We Use L1 Regularization in Sparse PCA?**  
- Traditional PCA allows **all features to have non-zero loadings** in the principal components.
- Sparse PCA adds a constraint that many of the entries of $\mathbf{w}$ should be zero.  
- To achieve this, we add an L1 constraint, which forces some coefficients to be exactly zero.  
- The optimization problem changes from:
  $$
  \max_{\mathbf{w}} \quad \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to} \quad \|\mathbf{w}\|_2 = 1
  $$
  to:
  $$
  \max_{\mathbf{w}} \quad \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to} \quad \|\mathbf{w}\|_2 = 1, \quad \|\mathbf{w}\|_1 \leq t
  $$
  Here, $\|\mathbf{w}\|_1$ represents the sum of absolute values of the components of $\mathbf{w}$.  
  - If $t$ is small, it forces more entries of $\mathbf{w}$ to be zero.
  - If $t$ is large, it's closer to standard PCA.

---

### **5️⃣ How is L1 Regularization Applied?**

**Mathematically**, L1 regularization forces sparsity by controlling the sum of absolute values of the loadings.  
For example, suppose we have three loadings:
$$
\mathbf{w} = [w_1, w_2, w_3]
$$
The L1 norm of $\mathbf{w}$ is:
$$
\|\mathbf{w}\|_1 = |w_1| + |w_2| + |w_3|
$$
We add a constraint:
$$
\|\mathbf{w}\|_1 \leq t
$$
For example, if $t = 1$ and initially $\mathbf{w} = [0.8, 0.6, 0.2]$ , we adjust $\mathbf{w}$ to satisfy the constraint.  
One possible solution is to set some of the weights to zero, like:  
$$
\mathbf{w} = [0.8, 0.2, 0]
$$
This makes it sparse.

---

### **Summary of Step 3**

1. **Loading**: Coefficient of each feature in a principal component.  
2. **Why Eigenvectors?**: They give the directions of maximum variance.  
3. **L1 Regularization**: Enforces sparsity by forcing some loadings to zero.  
4. **Calculation**: We find eigenvalues and eigenvectors, then add an L1 constraint to zero out small coefficients.  

