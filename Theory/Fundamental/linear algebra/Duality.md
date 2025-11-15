### **Duality in 2D, 3D, and Higher Dimensions**

Duality refers to the correspondence between vectors and certain objects (like scalars, vectors, or higher-order tensors) in spaces of various dimensions. This duality arises from the **wedge product** in exterior algebra and the properties of **antisymmetric tensors**.

---

### **1. Duality in 2D**
In 2D, vectors have a dual correspondence with **scalars**.

#### **Details:**
1. **Cross Product Analogy:**
   - In 2D, the cross product doesn't produce a vector perpendicular to a plane (since there's no third dimension). Instead, it produces a scalar representing the **signed area** of the parallelogram formed by two vectors.

   For two vectors $\mathbf{a} = \begin{bmatrix} a_x \\ a_y \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} b_x \\ b_y \end{bmatrix}$ :
   $$
   \mathbf{a} \times \mathbf{b} = a_x b_y - a_y b_x
   $$

   This scalar can be interpreted as the z-component of a 3D cross product with z-components of 0.

2. **Geometric Interpretation:**
   - The **signed scalar** indicates the orientation of $\mathbf{a}$ and $\mathbf{b}$ :
     - Positive: Counterclockwise rotation from $\mathbf{a}$ to $\mathbf{b}$.
     - Negative: Clockwise rotation.

3. **Dual Representation:**
   - Any vector $\mathbf{v} = \begin{bmatrix} v_x \\ v_y \end{bmatrix}$ in 2D has a dual scalar through:
     $$
     \text{Dual Scalar of } \mathbf{v} = v_x e_1 + v_y e_2
     $$
   - The scalar represents the area interaction in 2D.

---

### **2. Duality in 3D**
In 3D, vectors are dual to **antisymmetric matrices** or **pseudovectors** (depending on context).

#### **Details:**
1. **Cross Product:**
   - The cross product of two vectors $\mathbf{a}$ and $\mathbf{b}$ produces a vector perpendicular to both, encoding both **area** (magnitude) and **orientation** (direction):
     $$
     \mathbf{a} \times \mathbf{b} = \text{Dual Vector Corresponding to } (\mathbf{a} \wedge \mathbf{b})
     $$
   - The wedge product $(\mathbf{a} \wedge \mathbf{b})$ is inherently antisymmetric.

2. **Antisymmetric Tensor:**
   - A vector $\mathbf{v}$ can be represented as a **skew-symmetric matrix**:
     $$
     \mathbf{v} = \begin{bmatrix} v_x \\ v_y \\ v_z \end{bmatrix} 
     \leftrightarrow 
     \begin{bmatrix}
     0 & -v_z & v_y \\
     v_z & 0 & -v_x \\
     -v_y & v_x & 0
     \end{bmatrix}
     $$
   - This matrix encodes the plane's orientation and magnitude perpendicular to $\mathbf{v}$.

3. **Geometric Interpretation:**
   - Duality assigns the plane spanned by $\mathbf{a}$ and $\mathbf{b}$ a **normal vector** representing the perpendicularity in 3D.

---

### **3. Duality in Higher Dimensions (nD)**
In $n$ -dimensional spaces, vectors have dual correspondence with **multivectors** or **antisymmetric tensors of order $k$**.

#### **Details:**
1. **Wedge Product in $n$ -D:**
   - The wedge product generalizes the cross product, producing $k$ -forms that represent volumes spanned by $k$ vectors in $n$ -space.
   - For $k = n-1$ , the dual of a $k$ -form is a vector orthogonal to the subspace spanned by the $k$ vectors.

2. **No Unique Perpendicular Vector in $n > 3$ :**
   - In dimensions higher than 3, there are infinitely many vectors orthogonal to a given $k$ -plane, so the cross product as a single vector is undefined.
   - Instead, the duality produces a **higher-order object** (e.g., bivectors or pseudotensors).

3. **Hodge Duality:**
   - The Hodge dual maps $k$ -forms to $(n-k)$ -forms in $n$ -dimensional space.
   - Example: In 4D, the Hodge dual of a 2-form (e.g., the wedge product of two vectors) is another 2-form representing the orthogonal complement.

---

### **Summary of Duality in Dimensions**
| **Dimension** | **Dual of a Vector**                       | **Result**                   | **Cross Product Behavior** |
|---------------|-------------------------------------------|------------------------------|----------------------------|
| 2D            | Scalar (signed area)                     | Antisymmetric scalar         | Produces scalar            |
| 3D            | Vector or skew-symmetric matrix          | Normal vector               | Defined, unique direction  |
| 4D+           | $k$ -forms or higher-order tensors       | Multivectors                | Undefined as unique vector |

This shows how duality extends the concepts of orientation, area, and perpendicularity across different dimensions.

