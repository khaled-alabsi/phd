# Question: Closed-Loop vs Open-Loop in Process Control

#### 1. **What is Open-Loop?**

An *open-loop* system applies control actions **without feedback**. The system acts purely based on input settings (setpoints), without observing the outcome.

* **Example**: Setting the flow rate of a feed pump manually without checking if the product concentration is correct.
* **In the TEP**: You can simulate the TEP in open-loop mode, where all control loops are disabled, and variables evolve based only on inputs and disturbances.

#### 2. **What is Closed-Loop?**

A *closed-loop* system continuously adjusts its control actions **based on feedback** from the process outputs.

* **Example**: A temperature controller adjusts the cooling valve to maintain the reactor temperature at a target level.
* **In the TEP**: The default simulation setup includes many **PID controllers** acting on variables like flow, pressure, level, and temperature. This makes the TEP model **closed-loop** by default.

#### 3. **Why is Closed-Loop Mentioned in the Paper?**

The paper uses the **closed-loop** version of the Tennessee Eastman Process. The authors specifically mention "monitoring of closed-loop processes" because:

* Faults may be **masked by control loops**.
* The effect of a fault may take longer to be visible due to **controller action correcting deviations**.
* Detection performance (e.g. ARLₒ.c) depends on **how the controllers respond** to a disturbance.

### What Can Be Tuned?

The **PID controller parameters** can be tuned to **change how aggressively** the controller reacts to a disturbance or fault. The typical tunable parameters in each PID loop:

| Parameter                | Effect                                              |
|  |  |
| `Kp` (Proportional gain) | Increases response speed and amplitude to error     |
| `Ti` (Integral time)     | Eliminates steady-state error (longer Ti = slower)  |
| `Td` (Derivative time)   | Reacts to rate of error change (predictive control) |

#### In the Paper:

For fault IDV(15) (valve stiction in XMV$$
11]):

* Tuning the PID controlling **cooling water flow** (XMV$$
11]) changes how much and how fast the manipulated variable responds.
* **Higher `Kp`**: Faster detection (lower ARLₒ.c), but more aggressive actuator movement (higher variance in control variable).
* **Lower `Kp`**: Smoother control, but **slower detection** (higher ARLₒ.c), because the controller "hides" the fault.

### Summary of the Trade-off

* **More aggressive tuning** (e.g., higher `Kp`) → **faster fault detection**, but **risk of increased process variability** and actuator wear.
* **More conservative tuning** → **slower detection**, but **better process stability**.

This is called the **control-detection trade-off**.

---

# Question: What pid means?

**PID** stands for **Proportional–Integral–Derivative** controller. It's the most widely used control algorithm in industrial process control systems.

### 1. **What PID Does**

A PID controller adjusts a **manipulated variable** (e.g., valve position) to keep a **process variable** (e.g., temperature, pressure) close to a **setpoint**.

It does this by computing the control signal from three components:

$$
u(t) = K_p \cdot e(t) + K_i \cdot \int e(t) dt + K_d \cdot \frac{de(t)}{dt}
$$

Where:

* $e(t) = \text{setpoint} - \text{process value}$
* $u(t)$ = control output (e.g., signal to a valve)
* $K_p, K_i, K_d$ = tuning parameters

### 2. **The Three Terms**

| Term                                    | Role                                                             |
|  | - |
| **P**roportional $(K_p e(t))$           | Reacts to the current error (larger error → larger reaction)     |
| **I**ntegral $(K_i \int e(t))$          | Reacts to accumulated past errors (removes steady-state error)   |
| **D**erivative $(K_d \frac{de(t)}{dt})$ | Reacts to the rate of change of error (predicts future behavior) |

### 3. **Why PID is Used**

* Simple, robust, and easy to implement
* Works well for **SISO** (single-input-single-output) loops
* Can handle **disturbances** and maintain control in noisy industrial systems

### 4. **In Tennessee Eastman Process**

* Over 20 control loops are implemented as **PID controllers**
* Example: Reactor pressure is controlled by manipulating purge valve opening
* Tuning these PIDs influences how faults propagate and how detectable they are

---

# Question: So the author is saying that, tuning the  manipulated variables can enhance ARL oc?

Not exactly. The **author is saying that tuning the *PID controller parameters*** (not directly the manipulated variables) affects how faults propagate and how quickly they can be detected — which influences **ARLₒ.c (Average Run Length under out-of-control condition)**.

### Clarified Interpretation:

* **Manipulated variables (MVs)** are adjusted by **controllers**, not directly by the engineer in closed-loop mode.
* The **controllers' tuning** (especially the PID gains) governs **how fast and how strongly** these MVs respond to faults or disturbances.

#### So:

> Tuning the **controller** (that affects the MV behavior) can **reduce ARLₒ.c** because the fault causes **more visible changes** in the monitored signal.

### Example from the Paper:

* Fault IDV(15): cooling water valve **stiction** affects XMV$$
11].
* The **PID controller** tries to compensate.
* If the **controller is tuned more aggressively** (e.g., higher proportional gain), the stiction becomes **more obvious** in the control signal.
* This makes it **easier for the detection method (scale CUSUM) to spot the anomaly**, so **ARLₒ.c decreases**.

### Key Point:

> It’s not the MV tuning that improves detection, but rather the tuning of the controller **that governs the MV** in closed-loop mode.

---

# Question: Explain PID with Example:

Here is a step-by-step **numerical example** showing how a **PID controller** works in the context of the **Tennessee Eastman Process (TEP)** and how **tuning it affects anomaly detection (ARLₒ.c)** — specifically for **Fault IDV(15)** (cooling water valve stiction).

## ▶ Setup: Simulated Subsystem (Simplified)

We simulate:

* A **process variable**: Reactor temperature $T$
* A **manipulated variable**: Cooling water flow rate $Q$ (XMV$$
11])
* A **fault**: Valve stiction starts at time $t = 30$ minutes
* A **PID controller** controls $Q$ to keep $T$ near a setpoint

We simulate for 100 timesteps (1 minute each). Fault starts at step 30.

## ▶ PID Controller Equation (Discrete Form)

Let:

* $e_t = \text{setpoint} - T_t$
* $u_t = u_{t-1} + K_p (e_t - e_{t-1}) + K_i e_t + K_d (e_t - 2e_{t-1} + e_{t-2})$

We'll simulate two cases:

1. **Low gain PID**: $K_p = 0.5, K_i = 0.1, K_d = 0.0$
2. **High gain PID**: $K_p = 2.0, K_i = 0.4, K_d = 0.0$

Fault effect: the valve (XMV$$
11]) cannot change freely — it **sticks** for 3 timesteps once moved (simulated as no change in `u_t` for 3 steps after a change).

## ▶ Simulation Code (Python)

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_pid(K_p: float, K_i: float, fault_start: int) -> dict[str, np.ndarray]:
    setpoint = 100.0
    T = np.zeros(100)
    u = np.zeros(100)
    e = np.zeros(100)
    T[0] = 95.0
    integral = 0.0
    stuck = False
    stuck_count = 0
    last_u = 0.0

    for t in range(1, 100):
        # Error
        e[t] = setpoint - T[t-1]
        integral += e[t]

        # Controller output
        delta_u = K_p * e[t] + K_i * integral

        if t >= fault_start:
            if not stuck:
                last_u = u[t-1] + delta_u
                stuck = True
                stuck_count = 3
            else:
                stuck_count -= 1
                if stuck_count == 0:
                    stuck = False
            u[t] = last_u
        else:
            u[t] = u[t-1] + delta_u

        # Process response: temperature affected by cooling flow
        T[t] = T[t-1] - 0.2 * (u[t] - 5.0) + np.random.normal(0, 0.1)

    return {'T': T, 'u': u, 'e': e}
```

## ▶ Visual Output

Run this twice:

```python
low_gain = simulate_pid(K_p=0.5, K_i=0.1, fault_start=30)
high_gain = simulate_pid(K_p=2.0, K_i=0.4, fault_start=30)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(low_gain['u'], label='Low Gain')
plt.plot(high_gain['u'], label='High Gain')
plt.axvline(30, color='r', linestyle='--', label='Fault Starts')
plt.title("Manipulated Variable: Cooling Flow (XMV[11])")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(low_gain['T'], label='Low Gain')
plt.plot(high_gain['T'], label='High Gain')
plt.axvline(30, color='r', linestyle='--')
plt.title("Process Variable: Temperature")
plt.legend()
plt.tight_layout()
plt.show()
```

## ▶ Interpretation

* **Low gain PID**: The valve signal (`u_t`) changes slowly and masks the stiction effect → small variance → **harder to detect** with CUSUM → higher **ARLₒ.c**.
* **High gain PID**: The controller reacts faster and overcomes stiction more visibly → larger variance or clear patterns in `u_t` → **easier detection** → lower **ARLₒ.c**.

## ▶ Summary

If you want to:

* **Increase detection sensitivity**, use **aggressive PID tuning** (higher `Kp`, `Ki`).
* **Reduce actuator wear or noise**, use **conservative tuning**, but at the cost of detection speed.

---

# Question: how  Combined monitoring via Hotelling’s T²?

Here is a **numerical and mathematical example** of **combined monitoring using Hotelling’s T²** statistic. We will simulate the combination of 3 univariate CUSUM statistics (from 3 variables) into a **single multivariate control chart**.

### **Step 1: CUSUM Vector at a Given Time**

Assume we monitor 3 variables:

* $C_1 =$ CUSUM for variable XMV$$
10] (fault IDV(3))
* $C_2 =$ CUSUM for variable XMEAS$$
21] (fault IDV(9))
* $C_3 =$ CUSUM for variable XMV$$
11] (fault IDV(15))

At time $t = 50$, we observe the following CUSUM values:

$$
\mathbf{x} = \begin{bmatrix}
C_1 \\
C_2 \\
C_3
\end{bmatrix}
=
\begin{bmatrix}
2.1 \\
1.3 \\
4.0
\end{bmatrix}
$$

### **Step 2: Reference Mean and Covariance Matrix**

We estimate the in-control mean and covariance of the CUSUM vector from historical (in-control) data.

#### Mean vector (from historical data):

$$
\boldsymbol{\mu} = \begin{bmatrix}
1.0 \\
1.0 \\
1.0
\end{bmatrix}
$$

#### Covariance matrix $\mathbf{S}$:

$$
\mathbf{S} = \begin{bmatrix}
0.5 & 0.1 & 0.0 \\
0.1 & 0.4 & 0.05 \\
0.0 & 0.05 & 0.9
\end{bmatrix}
$$

### **Step 3: Compute Hotelling’s T²**

The Hotelling’s T² statistic is:

$$
T^2 = (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{S}^{-1} (\mathbf{x} - \boldsymbol{\mu})
$$

First, compute the deviation vector:

$$
\mathbf{d} = \mathbf{x} - \boldsymbol{\mu} =\begin{bmatrix}
2.1 - 1.0 \\
1.3 - 1.0 \\
4.0 - 1.0
\end{bmatrix}
=\begin{bmatrix}
1.1 \\
0.3 \\
3.0
\end{bmatrix}
$$

### **Step 4: Invert the Covariance Matrix $\mathbf{S}^{-1}$**

We compute the inverse numerically or symbolically (use any software if needed). Let's assume:

$$
\mathbf{S}^{-1} \approx \begin{bmatrix}
2.1 & -0.5 & 0.03 \\
-0.5 & 2.7 & -0.15 \\
0.03 & -0.15 & 1.15
\end{bmatrix}
$$

### **Step 5: Final Matrix Computation**

Compute:

$$ T^2 = \mathbf{d}^\top \mathbf{S}^{-1} \mathbf{d} $$

Breakdown:

1. Multiply $\mathbf{S}^{-1} \mathbf{d}$:

   $$
   \mathbf{z} = \begin{bmatrix}
   2.1 & -0.5 & 0.03 \\
   -0.5 & 2.7 & -0.15 \\
   0.03 & -0.15 & 1.15
   \end{bmatrix}
   \cdot
   \begin{bmatrix}
   1.1 \\
   0.3 \\
   3.0
   \end{bmatrix} = \begin{bmatrix}
   2.31 \\
   0.66 \\
   3.42
   \end{bmatrix}
   $$

2. Compute $\mathbf{d}^\top \cdot \mathbf{z}$:

   $$
   T^2 = \begin{bmatrix}
   1.1 & 0.3 & 3.0
   \end{bmatrix}
   \cdot
   \begin{bmatrix}
   2.31 \\
   0.66 \\
   3.42
   \end{bmatrix}=1.1 \cdot 2.31 + 0.3 \cdot 0.66 + 3.0 \cdot 3.42 = 2.541 + 0.198 + 10.26 = 13.0
   $$

### **Step 6: Compare to Control Limit**

Assume significance level $\alpha = 0.01$, and 3 variables → degrees of freedom = 3. The control limit is:

$$
\text{UCL}_{T^2} = \frac{3(n - 1)}{n - 3} F_{0.99, 3, n-3}
$$

For large $n$, you can use $\chi^2_{0.99, 3} \approx 11.34$

Since $T^2 = 13.0 > 11.34$, this point signals an **out-of-control** condition.

### Conclusion:

The multivariate T² combines all 3 univariate CUSUMs. Even if one or two are moderate, the combined statistic may exceed the threshold — enabling **earlier and more reliable detection** of complex or subtle faults.

# Quesion: Show the difference between LCS and SCS implantations 

Here is a **mathematical and conceptual comparison** between **Location CUSUM (LCS)** and **Scale CUSUM (SCS)**, focusing on how they are **implemented differently** to detect **mean shifts vs variance shifts**.

## 1. **Purpose**

| Statistic | Detects                                       | Used for                    |
| --------- | --------------------------------------------- | --------------------------- |
| **LCS**   | Small shift in **mean** (location parameter)  | Faults like IDV(3)          |
| **SCS**   | Small shift in **variance** (scale parameter) | Faults like IDV(9), IDV(15) |

## 2. **Data Assumption**

Let $x_t$ be a univariate process signal at time $t$, assumed **i.i.d. normally distributed**, centered:

$$
x_t \sim \mathcal{N}(\mu, \sigma^2)
\quad \text{with known or estimated} \quad \mu, \sigma^2
$$

## 3. **Location CUSUM (LCS)** — for mean shifts

**Standardized residual:**

$$
z_t = \frac{x_t - \mu_0}{\sigma}
$$

**One-sided LCS (for upward shift):**

$$
S_t = \max(0, S_{t-1} + z_t - k)
\quad \text{with } S_0 = 0
$$

* $k$: reference value, typically \~0.5 for small shifts
* If $S_t > h$ (decision threshold), alarm is raised

**Example:**
Assume $\mu_0 = 0, \sigma = 1, k = 0.5$

* $x_1 = 0.8 \Rightarrow z_1 = 0.8$,
  $S_1 = \max(0, 0 + 0.8 - 0.5) = 0.3$

* $x_2 = 0.7 \Rightarrow z_2 = 0.7$,
  $S_2 = \max(0, 0.3 + 0.7 - 0.5) = 0.5$

LCS **accumulates** small mean shifts over time.

## 4. **Scale CUSUM (SCS)** — for variance shifts

**Normalized squared residual:**

$$
y_t = \frac{(x_t - \mu_0)^2}{\sigma^2}
$$

Under normal conditions, $y_t \sim \chi^2_1$ (mean = 1)

**One-sided SCS (for variance increase):**

$$
S_t = \max(0, S_{t-1} + y_t - k)
\quad \text{with } S_0 = 0
$$

* $k$: reference value ≈ 1.2–1.5 depending on sensitivity
* Alarm when $S_t > h$

**Example:**
Assume $\mu_0 = 0, \sigma = 1, k = 1.2$

* $x_1 = 1.5 \Rightarrow y_1 = (1.5)^2 = 2.25$,
  $S_1 = \max(0, 0 + 2.25 - 1.2) = 1.05$

* $x_2 = -1.3 \Rightarrow y_2 = 1.69$,
  $S_2 = \max(0, 1.05 + 1.69 - 1.2) = 1.54$

SCS **accumulates deviations in variance**, not mean.

## 5. **Summary of Implementation Differences**

| Step                  | LCS (Mean Shift)                 | SCS (Variance Shift)                     |
| --------------------- | -------------------------------- | ---------------------------------------- |
| Input transformation  | $z_t = \frac{x_t - \mu}{\sigma}$ | $y_t = \frac{(x_t - \mu)^2}{\sigma^2}$   |
| Baseline distribution | Normal(0,1)                      | Chi-squared(1)                           |
| Reference value $k$   | \~0.5                            | \~1.2                                    |
| Accumulation          | Linear deviations from mean      | Squared deviations from mean             |
| Fault detected        | Small shift in mean              | Increase in variability (variance shift) |

Let’s clarify the difference in the **input transformation** step between **Location CUSUM (LCS)** and **Scale CUSUM (SCS)** with intuition and numerical examples.

## 1. **What are we trying to detect?**

| Method  | Target           | What kind of change?                                 |
| ------- | ---------------- | ---------------------------------------------------- |
| **LCS** | Mean (location)  | Detects shift in the **center** of the distribution  |
| **SCS** | Variance (scale) | Detects change in the **spread** of the distribution |

So the **transformation** prepares the data to highlight *only* the type of change each method wants to detect.

## 2. **Same data, different goals**

Let’s assume:

* In-control mean $\mu_0 = 0$
* In-control standard deviation $\sigma = 1$

We get some values from a process:

$$
x = [-0.2,\ 0.1,\ 1.2,\ -1.3,\ 2.5,\ 0.0]
$$

## 3. **Location CUSUM (LCS)**

LCS wants to track **shift in mean**, so it looks at **how far x is from the mean** — linearly:

### Transformation:

$$
z_t = \frac{x_t - \mu_0}{\sigma}
$$

Let’s compute for each point:

| $x_t$ | $z_t = \frac{x_t - 0}{1}$ |
| ----- | ------------------------- |
| -0.2  | -0.2                      |
| 0.1   | 0.1                       |
| 1.2   | 1.2                       |
| -1.3  | -1.3                      |
| 2.5   | 2.5                       |
| 0.0   | 0.0                       |

These values go into the CUSUM to **accumulate directionally consistent deviations** (e.g. many positive values → upward shift in mean).

## 4. **Scale CUSUM (SCS)**

SCS wants to detect **change in spread**, so it looks at **squared distance from the mean** — i.e., the magnitude of deviation, regardless of sign.

### Transformation:

$$
y_t = \frac{(x_t - \mu_0)^2}{\sigma^2}
$$

Compute:

| $x_t$ | $y_t = \frac{(x_t - 0)^2}{1}$ |
| ----- | ----------------------------- |
| -0.2  | 0.04                          |
| 0.1   | 0.01                          |
| 1.2   | 1.44                          |
| -1.3  | 1.69                          |
| 2.5   | 6.25                          |
| 0.0   | 0.00                          |

SCS doesn’t care about *direction*, only the *magnitude* of deviation.

## 5. **Key Insight**

|              | LCS transforms with                                | SCS transforms with                       |
| ------------ | -------------------------------------------------- | ----------------------------------------- |
| Sensitive to | **Direction** of shift (positive or negative)      | **Size** of deviation only                |
| Formula      | $z_t = \frac{x_t - \mu}{\sigma}$                   | $y_t = \frac{(x_t - \mu)^2}{\sigma^2}$    |
| Intuition    | "Is the process moving consistently higher/lower?" | "Is the process fluctuating more wildly?" |
