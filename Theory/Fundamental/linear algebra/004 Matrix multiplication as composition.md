### **Matrix Multiplication as Composition**
- Matrix multiplication represents the composition of two linear transformations.
- When two linear transformations are applied sequentially, their combined effect is captured by multiplying their corresponding matrices.

---

### **Linear Transformations and Basis Vectors**
- A linear transformation can be fully described by the way it maps basis vectors $\hat{i}$ and $\hat{j}$.
- For a 2D matrix:
  $$
  \begin{bmatrix}
  a & b \\
  c & d
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y
  \end{bmatrix}
  =
  x
  \begin{bmatrix}
  a \\
  c
  \end{bmatrix}
  +
  y
  \begin{bmatrix}
  b \\
  d
  \end{bmatrix}.
  $$
  This demonstrates how the columns of the matrix determine where the basis vectors land.

---

### **Composition of Transformations**
- Applying one transformation followed by another produces a new transformation described by a single matrix.
- Example:
  - Rotate the plane counterclockwise by $90^\circ$ (rotation matrix) and then shear it (shear matrix).
  - The resulting transformation is a matrix whose columns are the images of $\hat{i}$ and $\hat{j}$ after both transformations.

---

### **Order of Matrix Multiplication**
- The order of operations is critical because matrix multiplication is not commutative. 
- Transformation order:
  1. The matrix on the right applies its transformation first.
  2. The matrix on the left applies its transformation second.

---

### **General Computation of Matrix Multiplication**
- To multiply matrices $M_1$ and $M_2$ :
  - Take each column of $M_2$ , multiply it by $M_1$ , and use the resulting vectors as columns of the resulting matrix.
  - For $M_1 = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ and $M_2 = \begin{bmatrix} e & f \\ g & h \end{bmatrix}$ :
    $$
    M_1 M_2 =
    \begin{bmatrix}
    ae + bg & af + bh \\
    ce + dg & cf + dh
    \end{bmatrix}.
    $$

---

### **Associativity of Matrix Multiplication**
- Matrix multiplication is associative: $(AB)C = A(BC)$.
- Geometric intuition: The sequence of applying transformations remains unchanged regardless of grouping.

---

### **Noncommutativity of Matrix Multiplication**
- The order of matrices matters: $AB \neq BA$ in general.
- Example:
  - Shear matrix $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and rotation matrix $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ yield different results for $SR$ and $RS$.

---

### **Geometric Meaning**
- Matrix multiplication corresponds to applying one transformation after another.
- The product matrix captures the combined geometric effect.

---
