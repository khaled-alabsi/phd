### Gradient

The **gradient** is a vector that points in the direction of the steepest increase of a function. For a function $f(x, y)$ , the gradient is represented as:

$$
\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}
$$

### Step-by-Step Explanation

1. **Partial Derivatives**:
   - The gradient contains the **partial derivatives** of the function with respect to each variable.  
     For example:
     $$
     \frac{\partial f}{\partial x} \text{ measures how } f \text{ changes as } x \text{ changes, while keeping } y \text{ constant.}
     $$
     Similarly, $\frac{\partial f}{\partial y}$ measures the change with respect to $y$.

2. **Direction**:
   - The gradient points in the direction where the function $f(x, y)$ increases the fastest.

3. **Magnitude**:
   - The magnitude of the gradient $\|\nabla f(x, y)\|$ indicates the rate of increase in that direction.

To understand the relationship between the gradient pointing in the direction of the fastest increase and the update formula in gradient descent, let's break it down step by step.

---

### Step 1: The Gradient's Direction
- The gradient $\nabla f(x, y)$ at any point $(x, y)$ points in the direction where the function $f(x, y)$ increases the fastest.
- For example, if $\nabla f(x, y) = [4, 2]$ , moving in the direction of $[4, 2]$ (e.g., by adding $[4, 2]$ to $(x, y)$ ) will lead to an increase in $f(x, y)$.

---

### Step 2: Minimization Requires Moving in the Opposite Direction
- Since we want to **minimize** $f(x, y)$ , we must move in the opposite direction of the gradient.
- The opposite direction is $-\nabla f(x, y)$ , which leads to a decrease in $f(x, y)$.

---

### Step 3: Updating the Values
- The update formula is:
  $$
  (x_{\text{new}}, y_{\text{new}}) = (x_{\text{old}}, y_{\text{old}}) - \eta \cdot \nabla f(x_{\text{old}}, y_{\text{old}})
  $$
  - $(x_{\text{old}}, y_{\text{old}})$ : Current values of $x$ and $y$.
  - $\nabla f(x_{\text{old}}, y_{\text{old}})$ : Gradient at the current point, which points toward the fastest increase.
  - $\eta$ : Learning rate, which controls how far we step in the opposite direction.

---

### Connection Between the Gradient and Updates
1. **Gradient Direction**:
   - The gradient indicates where the function increases fastest.
   - To minimize the function, we move in the exact opposite direction.

2. **Magnitude of the Step**:
   - The size of the step is proportional to $\|\nabla f(x, y)\|$ , meaning larger gradients lead to bigger updates.
   - The learning rate $\eta$ ensures the step size is controlled.

3. **Iterative Convergence**:
   - Each update reduces $f(x, y)$ , bringing $(x, y)$ closer to the minimum.
   - As the gradient becomes smaller (closer to zero), the updates naturally slow down.

---

The value of the gradient is **relevant** for moving because it determines:

1. **How steep the slope is** at the current point.
2. **How far we need to move** to effectively minimize the function.

Let’s break it down further:

---

### Step 1: What Does the Gradient Represent?
The gradient at a point $(x, y)$ , $\nabla f(x, y) = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]$ , has two components:
1. **Direction**: Points toward the steepest ascent.
2. **Magnitude**: Tells us how steep that ascent is.

If the gradient is large, the function $f(x, y)$ is changing rapidly in that direction. If the gradient is small, the function is flatter.

---

### Step 2: How Does This Affect Movement?
In gradient descent, we update the values as:
$$
(x_{\text{new}}, y_{\text{new}}) = (x_{\text{old}}, y_{\text{old}}) - \eta \cdot \nabla f(x_{\text{old}}, y_{\text{old}})
$$

1. **Gradient Magnitude Affects Step Size**:
   - When the gradient is large (e.g., steep slope), $\nabla f(x, y)$ has a large magnitude, so we take a larger step to move quickly toward the minimum.
   - When the gradient is small (e.g., near the flat minimum), $\nabla f(x, y)$ has a small magnitude, so we take smaller steps, avoiding overshooting the minimum.

2. **Direction Adjusts the Path**:
   - The components of $\nabla f(x, y)$ determine how much to adjust $x$ and $y$ individually.
   - For example:
     - A gradient of $[4, 2]$ means $x$ contributes twice as much to the slope as $y$ , so we adjust $x$ more significantly than $y$.

---

### Step 3: Why Can't We Ignore Gradient Magnitude?
If we ignore the gradient magnitude and move with a constant step size:
- In **steep regions**, we might overshoot the minimum because we’re moving too fast.
- In **flat regions**, we’d move too slowly, wasting iterations because the step size isn’t adapted to the terrain.

---

### Summary
The gradient magnitude determines how aggressively to move:
- **Large gradient**: Big steps for faster progress.
- **Small gradient**: Small steps to refine the solution near the minimum.

To understand how derivatives capture the **direction** and **value of increase**, we need to analyze the role of partial derivatives and their mathematical behavior.

---

### Step 1: What is a Partial Derivative?

For a function $f(x, y)$ , the **partial derivatives** are:

1. $\frac{\partial f}{\partial x}$ : Measures the rate of change of $f(x, y)$ with respect to $x$ , holding $y$ constant.
2. $\frac{\partial f}{\partial y}$ : Measures the rate of change of $f(x, y)$ with respect to $y$ , holding $x$ constant.

---

### Step 2: Direction of Increase

The **gradient** $\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}$ combines the partial derivatives into a vector:

1. **Direction**: 
   - The gradient vector points in the direction where the function $f(x, y)$ increases the fastest.
   - Why? Each partial derivative tells how much $f(x, y)$ increases as you move along $x$ or $y$. Combining them forms a direction where their contributions maximize the increase.

---

### Step 3: Value of Increase

The **magnitude** of the gradient vector, $\|\nabla f(x, y)\|$ , gives the **rate of increase**:

$$
\|\nabla f(x, y)\| = \sqrt{\left(\frac{\partial f}{\partial x}\right)^2 + \left(\frac{\partial f}{\partial y}\right)^2}
$$

1. If the gradient is large ($\|\nabla f(x, y)\| \gg 0$ ):
   - The function increases quickly in the gradient's direction.
2. If the gradient is small ($\|\nabla f(x, y)\| \approx 0$ ):
   - The function is nearly flat in the gradient's direction.

---

### Step 4: Example to Illustrate

Take $f(x, y) = x^2 + y^2$.

1. Compute partial derivatives:
   $$
   \frac{\partial f}{\partial x} = 2x, \quad \frac{\partial f}{\partial y} = 2y
   $$

2. Gradient at a point $(x, y)$ :
   $$
   \nabla f(x, y) = \begin{bmatrix} 2x \\ 2y \end{bmatrix}
   $$

3. **Direction**:
   - At $(1, 2)$ , the gradient is:
     $$
     \nabla f(1, 2) = \begin{bmatrix} 2 \\ 4 \end{bmatrix}
     $$
   - The gradient points toward the direction of the fastest increase of $f(x, y)$ , which is aligned with $[2, 4]$.

4. **Value**:
   - The gradient magnitude at $(1, 2)$ :
     $$
     \|\nabla f(1, 2)\| = \sqrt{2^2 + 4^2} = \sqrt{20} \approx 4.47
     $$
   - This tells us that if you move slightly in the direction of $[2, 4]$ , $f(x, y)$ will increase at a rate of approximately $4.47$.

---

### Summary

1. **Direction**:
   - Derivatives indicate the direction of increase in each variable (e.g., $x$ and $y$ ).
   - Combining them in the gradient gives the fastest ascent direction.

2. **Value**:
   - The magnitude of the derivatives indicates the steepness of the slope, showing how quickly $f(x, y)$ increases in that direction.

   Let’s compute the **rate of change** (i.e., slope) manually without using derivative rules by applying the definition of the derivative:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

We will approximate this by using small values for $h$.

---

### Example: $f(x) = x^2$

#### Step 1: Calculate the Rate of Change at $x = 2$
Let $f(x) = x^2$. We compute the rate of change using small increments ($h$ ).

1. For $h = 0.1$ :
   $$
   \frac{f(2 + 0.1) - f(2)}{0.1} = \frac{(2.1)^2 - (2)^2}{0.1}
   $$
   Calculate:
   $$
   (2.1)^2 = 4.41, \quad (2)^2 = 4
   $$
   $$
   \frac{4.41 - 4}{0.1} = \frac{0.41}{0.1} = 4.1
   $$

2. For $h = 0.01$ :
   $$
   \frac{f(2 + 0.01) - f(2)}{0.01} = \frac{(2.01)^2 - (2)^2}{0.01}
   $$
   Calculate:
   $$
   (2.01)^2 = 4.0401, \quad (2)^2 = 4
   $$
   $$
   \frac{4.0401 - 4}{0.01} = \frac{0.0401}{0.01} = 4.01
   $$

As $h$ becomes smaller, the rate of change approaches $4$.

---

#### Step 2: Calculate the Rate of Change at $x = -1$
1. For $h = 0.1$ :
   $$
   \frac{f(-1 + 0.1) - f(-1)}{0.1} = \frac{(-0.9)^2 - (-1)^2}{0.1}
   $$
   Calculate:
   $$
   (-0.9)^2 = 0.81, \quad (-1)^2 = 1
   $$
   $$
   \frac{0.81 - 1}{0.1} = \frac{-0.19}{0.1} = -1.9
   $$

2. For $h = 0.01$ :
   $$
   \frac{f(-1 + 0.01) - f(-1)}{0.01} = \frac{(-0.99)^2 - (-1)^2}{0.01}
   $$
   Calculate:
   $$
   (-0.99)^2 = 0.9801, \quad (-1)^2 = 1
   $$
   $$
   \frac{0.9801 - 1}{0.01} = \frac{-0.0199}{0.01} = -1.99
   $$

As $h$ becomes smaller, the rate of change approaches $-2$.

---

### Interpretation

1. At $x = 2$ , the slope (rate of change) is **positive** and approaches $4$. This means $f(x)$ is increasing.
2. At $x = -1$ , the slope is **negative** and approaches $-2$. This means $f(x)$ is decreasing.

This calculation uses only the definition of the derivative, showing how the rate of change depends on $f(x+h) - f(x)$ over small intervals $h$.

Derivative rules work by generalizing the calculation of the **rate of change** (slope) as $h \to 0$. Let’s explain how they emerge naturally by looking at the structure of $f(x + h)$ and simplifying step by step.

---

### Step 1: Start With the Definition of the Derivative

The derivative is defined as:
$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

The rules emerge by simplifying $f(x + h) - f(x)$ when $h$ is very small, using algebra.

---

### Step 2: Example: $f(x) = x^2$

Using the definition:
$$
f'(x) = \lim_{h \to 0} \frac{(x + h)^2 - x^2}{h}
$$

#### Expand $f(x + h)$ :
$$
(x + h)^2 = x^2 + 2xh + h^2
$$

#### Subtract $f(x)$ :
$$
(x + h)^2 - x^2 = x^2 + 2xh + h^2 - x^2 = 2xh + h^2
$$

#### Divide by $h$ :
$$
\frac{f(x + h) - f(x)}{h} = \frac{2xh + h^2}{h} = 2x + h
$$

#### Take the limit as $h \to 0$ :
$$
f'(x) = \lim_{h \to 0} (2x + h) = 2x
$$

---

### Step 3: Why Does This Work?

1. **Simplifying $f(x + h) - f(x)$**:
   - The term $h^2$ becomes negligible as $h \to 0$. In the final step, we drop any terms that contain $h$ because they contribute almost nothing when $h$ is very small.

2. **General Pattern**:
   - The derivative rules, like $\frac{d}{dx} x^n = n x^{n-1}$ , come from repeatedly applying the same logic:
     - Expand $f(x + h)$ for powers of $x$ ,
     - Subtract $f(x)$ ,
     - Divide by $h$ ,
     - Drop higher-order terms (like $h^2, h^3, \dots$ ).

---

### Step 4: How Does This Apply to Other Functions?

#### For $f(x) = x^n$ :
Using the same approach:
$$
f'(x) = \lim_{h \to 0} \frac{(x + h)^n - x^n}{h}
$$

1. Expand $(x + h)^n$ using the binomial theorem:
   $$
   (x + h)^n = x^n + n x^{n-1}h + \frac{n(n-1)}{2} x^{n-2} h^2 + \dots
   $$

2. Subtract $x^n$ :
   $$
   (x + h)^n - x^n = n x^{n-1} h + \frac{n(n-1)}{2} x^{n-2} h^2 + \dots
   $$

3. Divide by $h$ :
   $$
   \frac{(x + h)^n - x^n}{h} = n x^{n-1} + \frac{n(n-1)}{2} x^{n-2} h + \dots
   $$

4. Take the limit as $h \to 0$ :
   - Only the first term $n x^{n-1}$ survives:
   $$
   f'(x) = n x^{n-1}
   $$

---

### Summary of Derivative Rules
- Rules like $\frac{d}{dx} x^n = n x^{n-1}$ come from expanding $f(x + h)$ and keeping only the **dominant terms** as $h$ becomes very small.
- The higher-order terms in $h$ vanish because their contribution becomes negligible when $h \to 0$.



### Example :: Problem: Minimize $f(x, y) = x^2 + y^2$

1. **Function**: $f(x, y) = x^2 + y^2$ is a paraboloid centered at the origin, and the minimum value is at $(x, y) = (0, 0)$.
2. **Gradient**:  
   Compute $\nabla f(x, y)$ :
   $$
   \nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x \\ 2y \end{bmatrix}
   $$
3. **Optimization Method**: Gradient Descent.  
   Start at an initial guess $(x_0, y_0)$ and iteratively move in the opposite direction of the gradient:
   $$
   (x_{\text{new}}, y_{\text{new}}) = (x_{\text{old}}, y_{\text{old}}) - \eta \cdot \nabla f(x_{\text{old}}, y_{\text{old}})
   $$
   where $\eta$ is the learning rate (step size).

---

### Step 1: Initialization

- Initial point: $(x_0, y_0) = (2, 1)$
- Learning rate: $\eta = 0.1$

---

### Step 2: Iteration 1

1. Compute the gradient at $(x_0, y_0) = (2, 1)$ :
   $$
   \nabla f(2, 1) = \begin{bmatrix} 2(2) \\ 2(1) \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}
   $$

2. Update $(x, y)$ :
   $$
   (x_{\text{new}}, y_{\text{new}}) = (2, 1) - 0.1 \cdot \begin{bmatrix} 4 \\ 2 \end{bmatrix}
   $$
   $$
   (x_{\text{new}}, y_{\text{new}}) = \begin{bmatrix} 2 - 0.4 \\ 1 - 0.2 \end{bmatrix} = \begin{bmatrix} 1.6 \\ 0.8 \end{bmatrix}
   $$

- New point: $(1.6, 0.8)$
- Function value:  
  $$
  f(1.6, 0.8) = (1.6)^2 + (0.8)^2 = 2.56 + 0.64 = 3.2
  $$

---

### Step 3: Iteration 2

1. Compute the gradient at $(1.6, 0.8)$ :
   $$
   \nabla f(1.6, 0.8) = \begin{bmatrix} 2(1.6) \\ 2(0.8) \end{bmatrix} = \begin{bmatrix} 3.2 \\ 1.6 \end{bmatrix}
   $$

2. Update $(x, y)$ :
   $$
   (x_{\text{new}}, y_{\text{new}}) = (1.6, 0.8) - 0.1 \cdot \begin{bmatrix} 3.2 \\ 1.6 \end{bmatrix}
   $$
   $$
   (x_{\text{new}}, y_{\text{new}}) = \begin{bmatrix} 1.6 - 0.32 \\ 0.8 - 0.16 \end{bmatrix} = \begin{bmatrix} 1.28 \\ 0.64 \end{bmatrix}
   $$

- New point: $(1.28, 0.64)$
- Function value:  
  $$
  f(1.28, 0.64) = (1.28)^2 + (0.64)^2 = 1.6384 + 0.4096 = 2.048
  $$

---

### Step 4: Iteration 3

1. Compute the gradient at $(1.28, 0.64)$ :
   $$
   \nabla f(1.28, 0.64) = \begin{bmatrix} 2(1.28) \\ 2(0.64) \end{bmatrix} = \begin{bmatrix} 2.56 \\ 1.28 \end{bmatrix}
   $$

2. Update $(x, y)$ :
   $$
   (x_{\text{new}}, y_{\text{new}}) = (1.28, 0.64) - 0.1 \cdot \begin{bmatrix} 2.56 \\ 1.28 \end{bmatrix}
   $$
   $$
   (x_{\text{new}}, y_{\text{new}}) = \begin{bmatrix} 1.28 - 0.256 \\ 0.64 - 0.128 \end{bmatrix} = \begin{bmatrix} 1.024 \\ 0.512 \end{bmatrix}
   $$

- New point: $(1.024, 0.512)$
- Function value:  
  $$
  f(1.024, 0.512) = (1.024)^2 + (0.512)^2 = 1.048576 + 0.262144 = 1.31072
  $$

---

### Observing the Pattern

With each iteration:
1. The gradient gets smaller.
2. The new point $(x, y)$ moves closer to $(0, 0)$ , the minimum.
3. The function value $f(x, y)$ decreases steadily.

---

### Step 5: Stopping Criteria

We continue this process until:
1. The gradient magnitude $\|\nabla f(x, y)\|$ becomes very small, or
2. The change in function value $f(x, y)$ between iterations becomes negligible.

```python
import numpy as np

def gradient_descent(learning_rate: float, iterations: int, initial_point: tuple[float, float]):
    # Define the function and its gradient
    def f(x: float, y: float) -> float:
        return x**2 + y**2

    def grad_f(x: float, y: float) -> tuple[float, float]:
        return 2 * x, 2 * y

    # Initialize variables
    x, y = initial_point
    points = [(x, y)]  # Store points for tracking progress

    for i in range(iterations):
        # Compute gradient
        grad_x, grad_y = grad_f(x, y)

        # Update x and y
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y

        # Save the updated point
        points.append((x, y))

        # Compute and print function value at the new point
        func_value = f(x, y)
        print(f"Iteration {i+1}: x = {x:.4f}, y = {y:.4f}, f(x, y) = {func_value:.4f}")

    return points

# Parameters
learning_rate = 0.1
iterations = 10
initial_point = (2.0, 1.0)

# Run gradient descent
gradient_descent(learning_rate, iterations, initial_point)

```

#### output

```
Iteration 1: x = 1.6000, y = 0.8000, f(x, y) = 3.2000
Iteration 2: x = 1.2800, y = 0.6400, f(x, y) = 2.0480
Iteration 3: x = 1.0240, y = 0.5120, f(x, y) = 1.3107
Iteration 4: x = 0.8192, y = 0.4096, f(x, y) = 0.8389
Iteration 5: x = 0.6554, y = 0.3277, f(x, y) = 0.5369
Iteration 6: x = 0.5243, y = 0.2621, f(x, y) = 0.3436
Iteration 7: x = 0.4194, y = 0.2097, f(x, y) = 0.2199
Iteration 8: x = 0.3355, y = 0.1678, f(x, y) = 0.1407
Iteration 9: x = 0.2684, y = 0.1342, f(x, y) = 0.0901
Iteration 10: x = 0.2147, y = 0.1074, f(x, y) = 0.0576

```

In machine learning, **gradient descent** is often used to minimize the **error function** (also called the **loss function**) that measures the difference between predicted and actual values. Let me show you how this works step by step with a **numerical example** in the context of a linear regression model.

---

### Problem: Minimize the Mean Squared Error (MSE)

We have:
- A **linear model**: $y = w \cdot x + b$ , where $w$ is the weight and $b$ is the bias.
- A dataset: $\{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$ , where $x$ is the input and $y$ is the actual output.
- **Loss function**: The Mean Squared Error (MSE):
  $$
  \text{MSE}(w, b) = \frac{1}{n} \sum_{i=1}^n \left( y_i - (w \cdot x_i + b) \right)^2
  $$

---

### Goal

We want to adjust $w$ and $b$ to minimize the MSE using gradient descent.

---

### Step 1: Compute Gradients

The gradients of the MSE with respect to $w$ and $b$ are:
$$
\frac{\partial \text{MSE}}{\partial w} = -\frac{2}{n} \sum_{i=1}^n x_i \cdot \left( y_i - (w \cdot x_i + b) \right)
$$
$$
\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum_{i=1}^n \left( y_i - (w \cdot x_i + b) \right)
$$

---

### Step 2: Update Rules

Gradient descent updates $w$ and $b$ iteratively:
$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{MSE}}{\partial w}
$$
$$
b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial \text{MSE}}{\partial b}
$$
where $\eta$ is the learning rate.

---

### Step 3: Numerical Example

#### Dataset
$$
\{(x_1, y_1), (x_2, y_2)\} = \{(1, 2), (2, 3)\}
$$

#### Initial values
$$
w = 0, \, b = 0, \, \eta = 0.1
$$

#### Iteration 1:
1. Predicted values:
   $$
   \hat{y}_1 = w \cdot x_1 + b = 0 \cdot 1 + 0 = 0
   $$
   $$
   \hat{y}_2 = w \cdot x_2 + b = 0 \cdot 2 + 0 = 0
   $$

2. Errors:
   $$
   e_1 = y_1 - \hat{y}_1 = 2 - 0 = 2
   $$
   $$
   e_2 = y_2 - \hat{y}_2 = 3 - 0 = 3
   $$

3. Gradients:
   $$
   \frac{\partial \text{MSE}}{\partial w} = -\frac{2}{2} \cdot \left( 1 \cdot 2 + 2 \cdot 3 \right) = -\frac{1}{1} \cdot (2 + 6) = -8
   $$
   $$
   \frac{\partial \text{MSE}}{\partial b} = -\frac{2}{2} \cdot \left( 2 + 3 \right) = -5
   $$

4. Update $w$ and $b$ :
   $$
   w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{MSE}}{\partial w} = 0 - 0.1 \cdot (-8) = 0.8
   $$
   $$
   b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial \text{MSE}}{\partial b} = 0 - 0.1 \cdot (-5) = 0.5
   $$

#### Updated values after Iteration 1:
$$
w = 0.8, \, b = 0.5
$$

---

### Iteration 2:
1. Predicted values:
   $$
   \hat{y}_1 = w \cdot x_1 + b = 0.8 \cdot 1 + 0.5 = 1.3
   $$
   $$
   \hat{y}_2 = w \cdot x_2 + b = 0.8 \cdot 2 + 0.5 = 2.1
   $$

2. Errors:
   $$
   e_1 = y_1 - \hat{y}_1 = 2 - 1.3 = 0.7
   $$
   $$
   e_2 = y_2 - \hat{y}_2 = 3 - 2.1 = 0.9
   $$

3. Gradients:
   $$
   \frac{\partial \text{MSE}}{\partial w} = -\frac{2}{2} \cdot \left( 1 \cdot 0.7 + 2 \cdot 0.9 \right) = -\frac{1}{1} \cdot (0.7 + 1.8) = -2.5
   $$
   $$
   \frac{\partial \text{MSE}}{\partial b} = -\frac{2}{2} \cdot \left( 0.7 + 0.9 \right) = -0.8
   $$

4. Update $w$ and $b$ :
   $$
   w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{MSE}}{\partial w} = 0.8 - 0.1 \cdot (-2.5) = 1.05
   $$
   $$
   b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{\partial \text{MSE}}{\partial b} = 0.5 - 0.1 \cdot (-0.8) = 0.58
   $$

#### Updated values after Iteration 2:
$$
w = 1.05, \, b = 0.58
$$


```python
import numpy as np

def gradient_descent_mse(learning_rate: float, iterations: int, initial_w: float, initial_b: float, x: np.ndarray, y: np.ndarray):
    n = len(x)  # Number of data points

    # Initialize parameters
    w, b = initial_w, initial_b

    for i in range(iterations):
        # Compute predictions
        y_pred = w * x + b

        # Compute errors
        errors = y - y_pred

        # Compute gradients
        grad_w = -(2/n) * np.sum(x * errors)
        grad_b = -(2/n) * np.sum(errors)

        # Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        # Compute MSE
        mse = np.mean(errors**2)

        # Print iteration details
        print(f"Iteration {i+1}: w = {w:.4f}, b = {b:.4f}, MSE = {mse:.4f}")

    return w, b

# Dataset
x = np.array([1, 2])
y = np.array([2, 3])

# Parameters
learning_rate = 0.1
iterations = 10
initial_w = 0.0
initial_b = 0.0

# Run gradient descent
gradient_descent_mse(learning_rate, iterations, initial_w, initial_b, x, y)

```
#### output

```
Iteration 1: w = 0.8000, b = 0.5000, MSE = 6.5000
Iteration 2: w = 1.0500, b = 0.6600, MSE = 0.6500
Iteration 3: w = 1.1270, b = 0.7130, MSE = 0.0708
Iteration 4: w = 1.1496, b = 0.7323, MSE = 0.0133
Iteration 5: w = 1.1551, b = 0.7410, MSE = 0.0075
Iteration 6: w = 1.1553, b = 0.7462, MSE = 0.0067
Iteration 7: w = 1.1538, b = 0.7504, MSE = 0.0065
Iteration 8: w = 1.1518, b = 0.7542, MSE = 0.0063
Iteration 9: w = 1.1496, b = 0.7578, MSE = 0.0061
Iteration 10: w = 1.1475, b = 0.7614, MSE = 0.0059

```
