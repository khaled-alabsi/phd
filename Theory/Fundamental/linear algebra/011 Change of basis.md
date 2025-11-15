Here’s an expanded and enhanced explanation of the **Change of Basis** concept from the PDF, adding more mathematical intuition, step-by-step derivations, and deeper insights into transformations.

---

# **Change of Basis: An In-Depth Exploration**

## **1. Understanding Basis and Coordinate Systems**
A **basis** in a vector space provides a reference framework for describing vectors. In two-dimensional space $\mathbb{R}^2$ , the standard basis consists of:

$$
\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

Any vector $\mathbf{v}$ can be expressed as:

$$
\mathbf{v} = x \mathbf{e}_1 + y \mathbf{e}_2
$$

where $x$ and $y$ are the **coordinates of $\mathbf{v}$** in this basis.

### **Alternative Basis**
Now, consider another basis formed by two new vectors:

$$
\mathbf{b}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad \mathbf{b}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

A vector $\mathbf{v}$ in this new basis is written as:

$$
\mathbf{v} = a \mathbf{b}_1 + b \mathbf{b}_2
$$

where $a$ and $b$ are the coordinates of $\mathbf{v}$ in the **new basis**.

### **Key Idea**
The vector itself **does not change**, only the way we describe it does. The change of basis allows us to **reinterpret the same vector** from different perspectives.

---

## **2. Change of Basis Matrix**
To transition from one coordinate system to another, we use a **change of basis matrix**.

### **From New Basis to Standard Basis**
If a vector is written in the new basis $\{\mathbf{b}_1, \mathbf{b}_2\}$ :

$$
\mathbf{v} = a \mathbf{b}_1 + b \mathbf{b}_2
$$

then in matrix form:

$$
\mathbf{v} = B \mathbf{c}
$$

where:
- $B$ is the **change of basis matrix**, whose columns are the new basis vectors:

  $$
  B = \begin{bmatrix} \mathbf{b}_1 & \mathbf{b}_2 \end{bmatrix} =
  \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}
  $$

- $\mathbf{c}$ is the coordinate vector in the new basis:

  $$
  \mathbf{c} = \begin{bmatrix} a \\ b \end{bmatrix}
  $$

Thus, the coordinates in the standard basis are given by:

$$
\mathbf{v}_{\text{standard}} = B \mathbf{c}
$$

### **From Standard Basis to New Basis**
To express a vector in the new basis, we solve for $\mathbf{c}$ :

$$
\mathbf{c} = B^{-1} \mathbf{v}_{\text{standard}}
$$

where $B^{-1}$ is the **inverse change of basis matrix**.

#### **Computing $B^{-1}$**
Using the formula for the inverse of a $2 \times 2$ matrix:

$$
B^{-1} = \frac{1}{\det(B)}
\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

For our basis:

$$
B = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}
$$

$$
\det(B) = (2)(1) - (-1)(1) = 3
$$

$$
B^{-1} = \frac{1}{3} \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}
$$

Now, to find the new basis coordinates $\mathbf{c}$ :

$$
\mathbf{c} = B^{-1} \mathbf{v}_{\text{standard}}
$$

For example, if $\mathbf{v} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$ , its coordinates in the new basis are:

$$
\mathbf{c} = \frac{1}{3} \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix}
$$

$$
= \frac{1}{3} \begin{bmatrix} 3 + 2 \\ -3 + 4 \end{bmatrix} = \frac{1}{3} \begin{bmatrix} 5 \\ 1 \end{bmatrix} = \begin{bmatrix} 5/3 \\ 1/3 \end{bmatrix}
$$

Thus, in Jennifer's coordinate system:

$$
\mathbf{v} = \frac{5}{3} \mathbf{b}_1 + \frac{1}{3} \mathbf{b}_2
$$

---

## **3. Change of Basis for Transformations**
Just as we describe vectors in different bases, we also describe **linear transformations** in different bases.

### **Transformation in Standard Basis**
Consider a transformation $A$ in the standard basis:

$$
A = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}
$$

which represents a **90-degree counterclockwise rotation**.

### **Transformation in the New Basis**
To express $A$ in the new basis:

1. **Convert coordinates from the new basis to the standard basis** (using $B$ ).
2. **Apply the transformation $A$ in the standard basis**.
3. **Convert back to the new basis** (using $B^{-1}$ ).

$$
A' = B^{-1} A B
$$

Using our matrices:

$$
A' = \frac{1}{3} \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}
\begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}
\begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}
$$

Carrying out the multiplication:

$$
A' = \frac{1}{3} \begin{bmatrix} 1(-1) + 1(0) & 1(0) + 1(1) \\ -1(-1) + 2(0) & -1(0) + 2(1) \end{bmatrix}
\begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}
$$

$$
= \frac{1}{3} \begin{bmatrix} -1 & 1 \\ 1 & 2 \end{bmatrix}
\begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}
$$

$$
= \frac{1}{3} \begin{bmatrix} -1(2) + 1(1) & -1(-1) + 1(1) \\ 1(2) + 2(1) & 1(-1) + 2(1) \end{bmatrix}
$$

$$
= \frac{1}{3} \begin{bmatrix} -2 + 1 & 1 + 1 \\ 2 + 2 & -1 + 2 \end{bmatrix}
$$

$$
= \frac{1}{3} \begin{bmatrix} -1 & 2 \\ 4 & 1 \end{bmatrix}
$$

Thus, in the new basis, the transformation matrix is:

$$
A' = \begin{bmatrix} -1/3 & 2/3 \\ 4/3 & 1/3 \end{bmatrix}
$$

---

## **4. Key Takeaways**
- A **change of basis matrix** transforms coordinates from one system to another.
- Its **inverse** reverses the transformation.
- Transformations themselves can be rewritten in different bases using $A' = B^{-1} A B$.
- In applications, change of basis simplifies **diagonalization, PCA, and eigenvector computations**.

---

### **Diagonalization via Change of Basis: Numerical Example**

#### **1. What is Diagonalization?**
Diagonalization is the process of finding a basis in which a given matrix $A$ is represented as a diagonal matrix $D$. This means we can rewrite $A$ as:

$$
A = P D P^{-1}
$$

where:
- $A$ is the original matrix.
- $D$ is a diagonal matrix containing eigenvalues of $A$.
- $P$ is the **change of basis matrix**, whose columns are eigenvectors of $A$.
- $P^{-1}$ is the inverse of $P$.

This is useful because diagonal matrices are easier to compute with (e.g., for exponentiation).

---

#### **2. Example: Finding the Change of Basis Matrix**
Consider the matrix:

$$
A = \begin{bmatrix} 4 & 1 \\ 6 & 3 \end{bmatrix}
$$

We want to diagonalize it, i.e., find a basis in which it is represented as a diagonal matrix.

---

#### **3. Step 1: Compute the Eigenvalues**
Eigenvalues satisfy the characteristic equation:

$$
\det(A - \lambda I) = 0
$$

$$
\begin{vmatrix} 4 - \lambda & 1 \\ 6 & 3 - \lambda \end{vmatrix} = 0
$$

Expanding the determinant:

$$
(4 - \lambda)(3 - \lambda) - (6 \cdot 1) = 0
$$

$$
12 - 4\lambda - 3\lambda + \lambda^2 - 6 = 0
$$

$$
\lambda^2 - 7\lambda + 6 = 0
$$

Factoring:

$$
(\lambda - 6)(\lambda - 1) = 0
$$

So the eigenvalues are:

$$
\lambda_1 = 6, \quad \lambda_2 = 1
$$

---

#### **4. Step 2: Compute Eigenvectors**
For each eigenvalue $\lambda$ , solve $(A - \lambda I) \mathbf{v} = 0$.

##### **Eigenvector for $\lambda_1 = 6$ :**
$$
(A - 6I) = \begin{bmatrix} 4 - 6 & 1 \\ 6 & 3 - 6 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\ 6 & -3 \end{bmatrix}
$$

Solving:

$$
\begin{bmatrix} -2 & 1 \\ 6 & -3 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

From the first row:

$$
-2x + y = 0 \quad \Rightarrow \quad y = 2x
$$

Choosing $x = 1$ , we get $y = 2$ , so the eigenvector is:

$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

##### **Eigenvector for $\lambda_2 = 1$ :**
$$
(A - 1I) = \begin{bmatrix} 4 - 1 & 1 \\ 6 & 3 - 1 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 6 & 2 \end{bmatrix}
$$

Solving:

$$
\begin{bmatrix} 3 & 1 \\ 6 & 2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

From the first row:

$$
3x + y = 0 \quad \Rightarrow \quad y = -3x
$$

Choosing $x = 1$ , we get $y = -3$ , so the eigenvector is:

$$
\mathbf{v}_2 = \begin{bmatrix} 1 \\ -3 \end{bmatrix}
$$

---

#### **5. Step 3: Construct the Change of Basis Matrix $P$**
The matrix $P$ consists of the eigenvectors as columns:

$$
P = \begin{bmatrix} 1 & 1 \\ 2 & -3 \end{bmatrix}
$$

Its inverse is:

$$
P^{-1} = \frac{1}{\det(P)} \begin{bmatrix} -3 & -1 \\ -2 & 1 \end{bmatrix}
$$

$$
\det(P) = (1)(-3) - (1)(2) = -3 - 2 = -5
$$

$$
P^{-1} = \frac{1}{-5} \begin{bmatrix} -3 & -1 \\ -2 & 1 \end{bmatrix} = \begin{bmatrix} 3/5 & 1/5 \\ 2/5 & -1/5 \end{bmatrix}
$$

---

#### **6. Step 4: Compute the Diagonal Matrix $D$**
$$
D = P^{-1} A P
$$

Computing:

$$
D = \begin{bmatrix} 3/5 & 1/5 \\ 2/5 & -1/5 \end{bmatrix}
\begin{bmatrix} 4 & 1 \\ 6 & 3 \end{bmatrix}
\begin{bmatrix} 1 & 1 \\ 2 & -3 \end{bmatrix}
$$

After computation, we get:

$$
D = \begin{bmatrix} 6 & 0 \\ 0 & 1 \end{bmatrix}
$$

Thus, $A$ is diagonalized as:

$$
A = P D P^{-1}
$$

---

### **7. Interpretation and Applications**
- The diagonal form $D$ makes exponentiation easy: $A^n = P D^n P^{-1}$.
- In many applications (e.g., **PCA, Markov Chains, Differential Equations**), diagonalization simplifies calculations.
- If a matrix is diagonalizable, we can interpret its **eigenvectors as a new coordinate system** where transformation effects are purely scaling.

---

### **Practical Application: Principal Component Analysis (PCA) via Change of Basis**

**Goal:**  
PCA is a statistical method used to reduce dimensionality and extract the most significant features from a dataset. It relies on **change of basis** to transform the data into a coordinate system aligned with the directions of maximum variance.

---

## **1. Data Representation and Covariance Matrix**
Consider a **dataset** of points in 2D space:

$$
X = \begin{bmatrix} 2 & 3 \\ 3 & 4 \\ 4 & 5 \\ 5 & 6 \end{bmatrix}
$$

where each row represents a data point $(x_1, x_2)$.  
To apply PCA, we **center the data** (subtract the mean of each column):

$$
X_{\text{centered}} = X - \text{mean}(X)
$$

The **covariance matrix** $C$ is computed as:

$$
C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}
$$

---

## **2. Compute Eigenvalues and Eigenvectors**
1. Solve for eigenvalues $\lambda$ using:

   $$
   \det(C - \lambda I) = 0
   $$

2. Find eigenvectors corresponding to each eigenvalue.

These eigenvectors form the **principal components**, defining a new **orthogonal basis**.

---

## **3. Change of Basis Using Eigenvectors**
Let $P$ be the matrix of eigenvectors:

$$
P = [\mathbf{v}_1, \mathbf{v}_2]
$$

To **transform** the data into the PCA coordinate system:

$$
X' = X_{\text{centered}} P
$$

where $X'$ represents the data in the **new basis**.

---

## **4. Dimensionality Reduction**
If we keep only the eigenvector with the largest eigenvalue, we **reduce dimensionality** from 2D to 1D:

$$
X'_{\text{reduced}} = X_{\text{centered}} \mathbf{v}_1
$$

This projects the data onto the direction of **maximum variance**.

---

## **5. Application in Machine Learning**
- PCA is used in **image compression**, **anomaly detection**, and **feature selection**.
- The **new coordinate system** (eigenvectors) allows efficient data representation.
- PCA is a **change of basis**, but instead of arbitrary vectors, we choose basis vectors that **maximize variance**.

---

```python
import numpy as np
import pandas as pd
import ace_tools as tools

# Sample dataset (each row is a data point in 2D space)
X = np.array([[2, 3], [3, 4], [4, 5], [5, 6]])

# Step 1: Centering the Data (subtracting the mean)
X_centered = X - np.mean(X, axis=0)

# Step 2: Compute the Covariance Matrix
C = np.cov(X_centered.T)

# Step 3: Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(C)

# Step 4: Sort Eigenvectors by Eigenvalues (Descending Order)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 5: Transform Data to Principal Component Basis
X_transformed = X_centered @ eigenvectors

# Step 6: Reduce Dimensionality to 1D (Using Only the First Principal Component)
X_reduced = X_transformed[:, 0].reshape(-1, 1)

# Display results
df_pca = pd.DataFrame({
    "Original X1": X[:, 0],
    "Original X2": X[:, 1],
    "PC1": X_transformed[:, 0],
    "PC2": X_transformed[:, 1],
    "Reduced Dim (PC1)": X_reduced.flatten()
})

tools.display_dataframe_to_user(name="PCA Results", dataframe=df_pca)


output:

Original X1  Original X2       PC1  PC2  Reduced Dim (PC1)
0            2            3 -2.121320  0.0          -2.121320
1            3            4 -0.707107  0.0          -0.707107
2            4            5  0.707107  0.0           0.707107
3            5            6  2.121320  0.0           2.121320

```