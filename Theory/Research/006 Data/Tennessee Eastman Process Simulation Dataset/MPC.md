# Model Predictive Control (MPC): Technical Tutorial

## 1. **What is MPC?**

Model Predictive Control (MPC) is an **advanced control strategy** that uses an **explicit process model** to predict future system behavior over a finite horizon and optimize control actions accordingly. Unlike PID, which reacts to errors, MPC **optimizes future control inputs** to minimize a cost function subject to **process and actuator constraints**.

---

## 2. **Key Components of MPC**

### 2.1 Model

Assume discrete-time linear dynamics:

$$
x_{k+1} = A x_k + B u_k \\
y_k = C x_k
$$

Where:

* $x_k \in \mathbb{R}^n$: system state
* $u_k \in \mathbb{R}^m$: control input
* $y_k \in \mathbb{R}^p$: output

In TEP, nonlinear or linearized models (via system ID or physics) are used.

---

### 2.2 Prediction Horizon

* Predict outputs over $N_p$ steps: $y_{k+1}, \ldots, y_{k+N_p}$
* Optimize control moves over $N_c \leq N_p$

---

### 2.3 Cost Function

Typical quadratic objective:

$$
J = \sum_{i=1}^{N_p} \| y_{k+i} - r_{k+i} \|_Q^2 + \sum_{i=0}^{N_c - 1} \| \Delta u_{k+i} \|_R^2
$$

Where:

* $\Delta u_k = u_k - u_{k-1}$
* $Q$: output tracking weight matrix
* $R$: input penalty matrix

---

### 2.4 Constraints

MPC handles **state and input constraints** explicitly:

$$
u_{\min} \leq u_k \leq u_{\max} \\
y_{\min} \leq y_k \leq y_{\max}
$$

---

## 3. **How MPC Works Step-by-Step**

1. **Measure** current state/output: $x_k, y_k$
2. **Predict** future trajectories using model and previous control
3. **Solve** optimization problem to compute optimal control sequence $u_{k}, u_{k+1}, \ldots$
4. **Apply only first control input** $u_k$
5. **Repeat** at next timestep (receding horizon)

---

## 4. **Numerical Example: 1D Linear System**

### System

$$
x_{k+1} = 0.9 x_k + 0.1 u_k
$$

### Objective

$$
\min_{u_0, u_1} \sum_{i=1}^{2} (x_{k+i} - r)^2 + \lambda \sum_{i=0}^{1} \Delta u_{k+i}^2
$$

Parameters:

* $x_k = 2.0$, $r = 0$, $\lambda = 0.1$, $u_{k-1} = 0$
* Constraints: $-1 \leq u_k \leq 1$

### Prediction

$$
x_{k+1} = 0.9x_k + 0.1u_k = 1.8 + 0.1u_0 \\
x_{k+2} = 0.9x_{k+1} + 0.1u_1 = 1.62 + 0.09u_0 + 0.1u_1
$$

### Cost

$$
J = (x_{k+1} - 0)^2 + (x_{k+2} - 0)^2 + 0.1(u_0^2 + (u_1 - u_0)^2)
$$

Substitute and solve the QP problem.

---

## 5. **Application in TEP**

| Use Case                  | MPC Role                                 |
| ------------------------- | ---------------------------------------- |
| Reactor temperature       | Multivariable control with constraints   |
| Product rate optimization | Tracking while limiting recycle pressure |
| Feed ratio coordination   | Prioritizing economic setpoints          |
| Fault compensation        | Switching to alternate operating regions |

TEP systems can be **linearized for MPC** around steady-states, or handled via **nonlinear MPC (NMPC)**.

---

## 6. **Advantages Over PID**

| Feature                | PID      | MPC             |
| ---------------------- | -------- | --------------- |
| Multivariable handling | ✗        | ✓               |
| Constraint handling    | Indirect | Explicit        |
| Prediction             | ✗        | ✓ (model-based) |
| Setpoint tracking      | Reactive | Optimized       |

---

## 7. **Python Skeleton Example (with `cvxpy`)**

```python
import cvxpy as cp
import numpy as np

def mpc_optimize(x0: float, Np: int, Nc: int, r: float, lambda_: float) -> float:
    A: float = 0.9
    B: float = 0.1
    u = cp.Variable(Nc)
    x = [x0]

    cost = 0
    for i in range(Np):
        if i < Nc:
            x_next = A * x[i] + B * u[i]
        else:
            x_next = A * x[i]  # Assume zero control after Nc
        x.append(x_next)
        cost += cp.square(x_next - r)
        if i < Nc:
            if i == 0:
                delta_u = u[i]
            else:
                delta_u = u[i] - u[i - 1]
            cost += lambda_ * cp.square(delta_u)

    constraints = [u >= -1, u <= 1]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    return u.value[0]
```

---

You're right — the definitions of variables like `A` and `B` must be **explicitly and formally defined**, especially when used in control-theoretic equations. Below is the corrected and **self-contained explanation** with all matrix terms defined:

---

# **Model Predictive Control (MPC): System Model Details**

## 1. **State-Space Representation**

MPC typically assumes a **discrete-time linear time-invariant system** of the form:

$$
x_{k+1} = A x_k + B u_k \\
y_k = C x_k + D u_k
$$

Where:

| Symbol                          | Dimension                                             | Meaning |
| ------------------------------- | ----------------------------------------------------- | ------- |
| $x_k \in \mathbb{R}^n$          | State vector at time step $k$                         |         |
| $u_k \in \mathbb{R}^m$          | Control input vector                                  |         |
| $y_k \in \mathbb{R}^p$          | Output vector                                         |         |
| $A \in \mathbb{R}^{n \times n}$ | State transition matrix                               |         |
| $B \in \mathbb{R}^{n \times m}$ | Input matrix (maps control to state dynamics)         |         |
| $C \in \mathbb{R}^{p \times n}$ | Output matrix (maps state to output)                  |         |
| $D \in \mathbb{R}^{p \times m}$ | Direct input-output matrix (often zero in TEP models) |         |

---

## 2. **Example: 2-State, 1-Input System**

Assume a temperature-pressure control subsystem linearized from TEP:

$$
x_k = \begin{bmatrix} \text{reactor temperature} \\ \text{reactor pressure} \end{bmatrix},\quad
u_k = \begin{bmatrix} \text{coolant flow} \end{bmatrix}
$$

System matrices might be:

$$
A = \begin{bmatrix} 0.95 & 0.02 \\ 0.01 & 0.90 \end{bmatrix},\quad
B = \begin{bmatrix} 0.05 \\ 0.03 \end{bmatrix},\quad
C = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\quad
D = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

So,

$$
x_{k+1} = A x_k + B u_k = 
\begin{bmatrix} 
0.95 & 0.02 \\ 
0.01 & 0.90 
\end{bmatrix} 
x_k + 
\begin{bmatrix} 
0.05 \\ 
0.03 
\end{bmatrix} u_k
$$

and

$$
y_k = C x_k = x_k
$$

---

## 3. **Use in MPC**

This model predicts future states $x_{k+1}, x_{k+2}, \ldots, x_{k+N}$ as a function of current state $x_k$ and future control inputs $u_k, u_{k+1}, \ldots$. The optimization problem is then constructed to:

* Track a desired output trajectory $r_k$
* Penalize control effort and deviations
* Respect constraints $u_{\min} \leq u_k \leq u_{\max}$, etc.

---

If needed, I can also show how to derive $A$ and $B$ from first principles or from TEP simulation data using system identification (e.g., via subspace methods or regression).

---

**Model Predictive Control (MPC) – Car Driving Example with Constraints**

Use case: Autonomous car following a lane.

---

### Sample Time

* **Definition**: Interval between control updates.
* **Example**: Car's controller updates every 0.1 seconds.

---

### Prediction Horizon

* **Definition**: Future time window over which predictions are made.
* **Example**: Predict car behavior over the next 2 seconds (20 steps if sample time = 0.1s).

---

### Control Horizon

* **Definition**: Future time window over which control inputs are optimized.
* **Example**: Optimize only the next 0.5 seconds (5 steps); after that, assume constant input.

---

### Constraints

* **Definition**: Physical or safety limits on states or control inputs.
* **Examples**:

  * **Speed**: Limited between 0 and 120 km/h.
  * **Steering angle**: Limited between -30° and +30°.
  * **Steering rate**: Limit how quickly the steering wheel turns (e.g., max ±10°/s).
  * **Acceleration**: Bounded between -3 m/s² (braking) and +2 m/s² (acceleration).
  * **Lane boundaries**: Lateral position must remain within road edges.

---

### Weights

* **Definition**: Penalize deviations or aggressive control actions in cost function.
* **Examples**:

  * **Tracking error weight**: High to keep the car in the center of the lane.
  * **Steering effort weight**: Moderate to avoid overly aggressive steering.
  * **Acceleration weight**: Higher to ensure smooth driving.
  * **Slack variable weight**: Penalizes constraint violations (used for soft constraints).

---

### Example MPC Control Loop (at every timestep)

1. Predict car’s position, speed, and heading for next 2 seconds.
2. Optimize steering and acceleration over the first 0.5 seconds to minimize lane deviation and control effort.
3. Enforce constraints (speed, steering angle, etc.).
4. Apply the first control input.
5. Move forward one sample time and repeat.

This ensures the car stays in lane, respects speed and comfort limits, and reacts optimally to the road ahead.