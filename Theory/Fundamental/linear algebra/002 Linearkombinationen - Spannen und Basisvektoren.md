### **Understanding the Span**

#### **Definition of Span**
The **span** of a set of vectors is the collection of all possible **linear combinations** of those vectors.

#### **Linear Combination**
A **linear combination** of vectors $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n$ is:
$$
c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_n \mathbf{v}_n, \quad c_i \in \mathbb{R}
$$
where $c_1, c_2, \dots, c_n$ are scalars (real numbers).

#### **Span**
The **span** of the vectors $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n$ is the set of all such linear combinations:
$$
\text{Span}(\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}) = \{ c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_n \mathbf{v}_n \ | \ c_i \in \mathbb{R} \}.
$$
**Geometric Meaning**:
- It is the collection of all points you can reach by **scaling** and **adding** these vectors.

---

### **Special Vectors in the xy-Coordinate System**

#### **Unit Vectors**
In the $xy$ -coordinate system, there are two special **unit vectors**:
1. $\mathbf{i}$ : Points to the **right** with length $1$ , commonly called:
   - "i-hat" ($\mathbf{i}$ ).
   - The **unit vector in the x-direction**.
   - Representation: $\mathbf{i} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

2. $\mathbf{j}$ : Points **straight up** with length $1$ , commonly called:
   - "j-hat" ($\mathbf{j}$ ).
   - The **unit vector in the y-direction**.
   - Representation: $\mathbf{j} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

#### **Building Other Vectors with $\mathbf{i}$ and $\mathbf{j}$**
Any vector $\mathbf{v}$ in the $xy$ -plane can be written as a linear combination of $\mathbf{i}$ and $\mathbf{j}$ :
$$
\mathbf{v} = c_1 \mathbf{i} + c_2 \mathbf{j} = \begin{bmatrix} c_1 \\ c_2 \end{bmatrix}
$$
Here:
- $c_1$ : The scalar multiplier for $\mathbf{i}$ , representing movement along the x-axis.
- $c_2$ : The scalar multiplier for $\mathbf{j}$ , representing movement along the y-axis.

#### **Geometric Meaning**
- $\mathbf{i}$ and $\mathbf{j}$ define the **basis** for the xy-plane.
- They span $\mathbb{R}^2$ , meaning any vector in the xy-plane can be represented using $\mathbf{i}$ and $\mathbf{j}$.

---

### **Key Geometric Intuitions**

1. **Span of a Single Vector**:
   - The span of a single vector $\mathbf{v}_1$ is a **line** passing through the origin and $\mathbf{v}_1$.
   - This represents a **1D subspace** of the larger space.

2. **Span of Two Vectors**:
   - If two vectors $\mathbf{v}_1$ and $\mathbf{v}_2$ are **not collinear** (i.e., not multiples of each other), their span forms a **plane** through the origin.
   - If they are collinear, their span remains a **line**.

3. **Span of Three or More Vectors**:
   - If three vectors $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ are **not coplanar** (i.e., not confined to the same plane), their span forms a **3D space**.
   - If one vector is a linear combination of the others, the span will remain **lower-dimensional**.

---

### **Linear Independence and Span**

#### **Linearly Independent Vectors**:
- Vectors are **linearly independent** if no vector in the set can be written as a linear combination of the others.
- **Geometric Meaning**:
  - Independent vectors form the **basis** of their span.

#### **Linearly Dependent Vectors**:
- If one vector is a combination of others, the vectors are **dependent**, and the span remains lower-dimensional.

---

### **Examples**

#### **1. Span of a Single Vector**
Let:
$$
\mathbf{v} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$
The span of $\mathbf{v}$ is the line:
$$
\text{Span}(\mathbf{v}) = \{ c \begin{bmatrix} 1 \\ 2 \end{bmatrix} \ | \ c \in \mathbb{R} \}.
$$

#### **2. Span of Two Vectors**
Let:
$$
\mathbf{v}_1 = \mathbf{i} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \mathbf{j} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$
The span is:
$$
\text{Span}(\{\mathbf{v}_1, \mathbf{v}_2\}) = \mathbb{R}^2.
$$

#### **3. Dependent Vectors**
Let:
$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 2 \\ 0 \end{bmatrix}
$$
The span is:
$$
\text{Span}(\{\mathbf{v}_1, \mathbf{v}_2\}) = \text{Span}(\mathbf{v}_1).
$$

---

### **Applications**

1. **Vector Spaces**:
   - Span helps define **vector subspaces**.
   - Basis vectors span the entire space.

2. **Linear Transformations**:
   - The range of a transformation is the span of its column vectors.

3. **Physics**:
   - Describes forces, velocities, or other vector quantities.

4. **Data Science**:
   - PCA (Principal Component Analysis) reduces data dimensions by analyzing spans.

