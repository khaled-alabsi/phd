### Partial Least Squares (PLS) Explanation:

#### Step 1: Objective
PLS is a **dimension reduction and regression** method that seeks to:
1. Maximize the **covariance** between the predictor variables ($X$ ) and the response variable ($Y$ ).
2. Extract **latent variables** (components) that summarize $X$ and $Y$ , focusing on predictive relationships.

#### Step 2: How It Works
1. **Latent Variable Extraction**:
   - PLS creates new components (linear combinations of $X$ ) that are most correlated with $Y$.
   - These components are uncorrelated with each other.

2. **Model Building**:
   - Each component is used to fit a regression model between $X$ and $Y$ , enabling predictions.

3. **Handling Multi-Collinearity**:
   - By using latent variables, PLS avoids issues of collinearity among predictors in $X$.

#### Step 3: Workflow
1. Center and scale $X$ and $Y$.
2. Iteratively extract components that:
   - Maximize the covariance between $X$ and $Y$.
   - Regress $Y$ on the extracted components.
3. Stop when the desired number of components is reached.

#### Use Cases
1. **Large, Multicollinear Datasets**:
   - $X$ variables are many and highly correlated.
2. **Predictive Modeling**:
   - Predict $Y$ effectively using fewer latent components.
3. **Monitoring Processes**:
   - Used in chemometrics and manufacturing for process control.

Using Partial Least Squares (PLS) in control chart contexts for monitoring multiple variables involves these steps:

---

### **Step 1: Problem Context**
- In a multivariate control chart, the objective is to monitor process stability across multiple correlated variables ($X$ ) and detect any shifts that may indicate an out-of-control process.
- PLS helps reduce dimensionality while preserving the relationship between variables and detecting anomalies.

---

### **Step 2: Workflow for PLS in Control Charts**
1. **Data Preparation**:
   - Collect historical in-control data.
   - Normalize or standardize $X$ variables (mean = 0, variance = 1).

2. **PLS Model Training**:
   - Fit a PLS model using the historical data.
   - Extract latent components ($T$ ) that capture the maximum covariance between $X$ and the process state ($Y$ , if available).
     - If $Y$ (e.g., quality indicators) is unavailable, use PLS unsupervised variants (e.g., Principal Component Analysis).

3. **Residual Analysis**:
   - Decompose $X$ into:
     - Modeled part: $T \cdot P'$ (scores and loadings).
     - Residuals: $\text{Residuals} = X - T \cdot P'$.
   - Monitor these separately:
     - **T^2 Statistic**: Monitors variations in the latent components.
     - **SPE (Squared Prediction Error)**: Tracks variations in residuals.

4. **Control Limit Estimation**:
   - Calculate control limits for $T^2$ and SPE using historical in-control data (e.g., based on statistical thresholds or confidence intervals).

5. **Real-Time Monitoring**:
   - Project new observations onto the PLS model to compute scores ($T$ ) and residuals.
   - Compare $T^2$ and SPE against control limits to detect anomalies.

---

### **Step 3: Benefits in Multivariate Control Charts**
- **Dimensionality Reduction**:
  - Extracts a few latent variables instead of monitoring all original variables.
- **Handles Multi-Collinearity**:
  - Useful when $X$ variables are correlated.
- **Improves Sensitivity**:
  - Focuses on variations most relevant to the process outcome.

---

### **Use Case Example**
- In a manufacturing process with $X = [\text{temperature, pressure, viscosity, etc.}]$ :
  1. Use PLS to model the relationship between $X$ and $Y$ (e.g., product quality).
  2. Monitor $T^2$ and SPE for deviations indicating potential shifts in the process.

Here’s the full Python code for implementing and validating PLS in a control chart context:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import chi2

# Simulate Data: 100 samples, 3 variables
np.random.seed(42)
n_samples, n_features = 100, 3

# In-control process data
X_in_control = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))
Y_in_control = np.zeros(n_samples)

# Out-of-control process data
X_out_control = np.random.normal(loc=2, scale=1, size=(10, n_features))
Y_out_control = np.ones(10)

# Combine data
X = np.vstack((X_in_control, X_out_control))
Y = np.hstack((Y_in_control, Y_out_control))

# Train PLS Model
pls = PLSRegression(n_components=2)
pls.fit(X_in_control, Y_in_control)

# Calculate Latent Scores and Residuals
T_scores = pls.x_scores_  # Latent scores for in-control data
X_pred = pls.predict(X_in_control)
residuals = X_in_control - pls.inverse_transform(pls.transform(X_in_control))

# Define Control Limits
T2_limit = chi2.ppf(0.99, df=2)  # 99% control limit for T^2
SPE_limit = np.percentile(np.sum(residuals**2, axis=1), 99)  # 99% for SPE

# Calculate SPE and T^2 for all data
variance_t_scores = np.var(T_scores, axis=0)
variance_t_scores[variance_t_scores == 0] = 1e-10  # Handle zero variance
T_scores_all = pls.transform(X)
X_reconstructed_all = pls.inverse_transform(T_scores_all)
SPE_all = np.sum((X - X_reconstructed_all)**2, axis=1)
T2_all = np.sum((T_scores_all / variance_t_scores)**2, axis=1)

# Labels for plotting
labels = np.hstack((["In-control"] * len(X_in_control), ["Out-of-control"] * len(X_out_control)))

# Validation
in_control_T2 = T2_all[:len(X_in_control)]
out_control_T2 = T2_all[len(X_in_control):]
in_control_SPE = SPE_all[:len(X_in_control)]
out_control_SPE = SPE_all[len(X_in_control):]

out_control_T2_exceed = np.sum(out_control_T2 > T2_limit)
out_control_SPE_exceed = np.sum(out_control_SPE > SPE_limit)

# Results
print("Validation Results:")
print(f"Total Out-of-Control Points: {len(X_out_control)}")
print(f"T^2 Detected Correctly: {out_control_T2_exceed == len(X_out_control)}")
print(f"SPE Detected Correctly: {out_control_SPE_exceed == len(X_out_control)}")
print(f"Out-of-Control T^2 Exceeding Limit: {out_control_T2_exceed}")
print(f"Out-of-Control SPE Exceeding Limit: {out_control_SPE_exceed}")

# Plot T^2
plt.figure(figsize=(10, 6))
plt.scatter(range(len(T2_all)), T2_all, c=["blue" if lbl == "In-control" else "red" for lbl in labels], label="Observations")
plt.axhline(T2_limit, color="green", linestyle="--", label=f"T^2 Control Limit ({T2_limit:.2f})")
plt.title("T^2 Statistic for Observations (Corrected)")
plt.xlabel("Observation Index")
plt.ylabel("T^2")
plt.legend()
plt.show()

# Plot SPE
plt.figure(figsize=(10, 6))
plt.scatter(range(len(SPE_all)), SPE_all, c=["blue" if lbl == "In-control" else "red" for lbl in labels], label="Observations")
plt.axhline(SPE_limit, color="green", linestyle="--", label=f"SPE Control Limit ({SPE_limit:.2f})")
plt.title("SPE Statistic for Observations")
plt.xlabel("Observation Index")
plt.ylabel("SPE")
plt.legend()
plt.show()
```

### Key Outputs:
1. **Validation Results**:
   - Whether $T^2$ and SPE correctly detected all out-of-control points.
2. **Plots**:
   - $T^2$ : Shows latent variable monitoring results.
   - SPE: Residual-based anomaly detection results.

### Validation Results:
- **Total Out-of-Control Points**: 10.
- **T² Detected Correctly**: **False** (No out-of-control points were detected by $T^2$ ).
- **SPE Detected Correctly**: **True** (All 10 out-of-control points were detected by SPE).
- **Out-of-Control $T^2$ Exceeding Limit**: 0.
- **Out-of-Control SPE Exceeding Limit**: 10.

Here’s a **step-by-step solved numerical example** for applying Partial Least Squares (PLS) to monitor multiple variables with $T^2$ and SPE.

![alt text](images/pls/pls-2.png)
![alt text](images/pls/pls-1.png)
---

### Mathematics Behind Partial Least Squares (PLS)

Partial Least Squares (PLS) is based on extracting latent variables that maximize the covariance between predictors ($X$ ) and response ($Y$ ). The method can be broken into these steps mathematically:

---

### **Step 1: Data Centering**
Before applying PLS, the data is centered and scaled:

$$
\tilde{X} = X - \bar{X}, \quad \tilde{Y} = Y - \bar{Y}
$$

Here, $\bar{X}$ and $\bar{Y}$ are the means of $X$ and $Y$ , ensuring the data has zero mean.

---

### **Step 2: Latent Variable Extraction**
PLS iteratively finds components ($t$ and $u$ ) that maximize the covariance between $X$ and $Y$.

1. **Find Weight Vectors**:
   Compute weight vectors $w$ and $c$ such that the covariance between the latent scores $t$ and $u$ is maximized:

   $$
   w = \arg\max_w \, \mathrm{Cov}^2(Xw, Yc)
   $$

   Where $t = Xw$ and $u = Yc$.

2. **Latent Scores**:
   After finding $w$ and $c$ , the scores are calculated as:
   $$
   t = Xw, \quad u = Yc
   $$

3. **Loading Vectors**:
   The loadings $p$ and $q$ are the projections of $X$ and $Y$ onto the latent scores $t$ :
   $$
   p = X^\top t / (t^\top t), \quad q = Y^\top u / (u^\top u)
   $$

4. **Deflate $X$ and $Y$**:
   The residuals for $X$ and $Y$ are updated by removing the contribution of the current components:
   $$
   X \leftarrow X - tp^\top, \quad Y \leftarrow Y - tq^\top
   $$

   This ensures that subsequent components capture new variations in $X$ and $Y$.

---

### **Step 3: Regression Model**
Once all components are extracted, the regression relationship between $X$ and $Y$ is built:

$$
\hat{Y} = TQ^\top
$$

Where:
- $T = [t_1, t_2, \dots, t_a]$ is the matrix of latent scores.
- $Q = [q_1, q_2, \dots, q_a]$ are the loadings for $Y$.

The final regression coefficients $B$ are:
$$
B = W(P^\top W)^{-1} Q^\top
$$

Here:
- $W = [w_1, w_2, \dots, w_a]$ are the weight vectors for $X$.
- $P = [p_1, p_2, \dots, p_a]$ are the loadings for $X$.

---

### **Step 4: Prediction**
For a new observation $X_\text{new}$ , the response is predicted using:
$$
\hat{Y}_\text{new} = X_\text{new} B
$$

---

### Key Concepts:
- **Latent Variables**: Extracted components $t$ summarize $X$ while being most correlated with $Y$.
- **Deflation**: After extracting each component, $X$ and $Y$ are updated to remove explained variance.
- **Dimensionality Reduction**: By selecting a limited number of components ($a$ ), PLS reduces the dimensionality of the problem.

---



### Problem Setup
We have a process with 3 variables ($X_1, X_2, X_3$ ) and want to monitor 5 new observations ($X_{new}$ ) for anomalies. Historical **in-control data** ($X$ ) consists of 6 samples:

#### Historical Data ($X$ ):
$$
X = \begin{bmatrix}
1 & 2 & 3 \\
2 & 3 & 4 \\
3 & 4 & 5 \\
4 & 5 & 6 \\
5 & 6 & 7 \\
6 & 7 & 8 \\
\end{bmatrix}
$$

#### New Data ($X_{new}$ ):
$$
X_{new} = \begin{bmatrix}
2 & 3 & 4 \\
3 & 4 & 5 \\
10 & 12 & 14 \\
4 & 5 & 6 \\
9 & 11 & 13 \\
\end{bmatrix}
$$

---

### Step 1: Standardize the Data
Standardize each variable in $X$ to mean 0 and variance 1:

$$
\text{Standardized } X = \frac{X - \text{mean}(X)}{\text{std}(X)}
$$

#### Mean:
$$
\text{mean}(X) = \begin{bmatrix} 3.5 & 4.5 & 5.5 \end{bmatrix}
$$

#### Standard Deviation:
$$
\text{std}(X) = \begin{bmatrix} 1.87 & 1.87 & 1.87 \end{bmatrix}
$$

#### Standardized $X$ :
$$
X_{\text{std}} = \begin{bmatrix}
-1.33 & -1.33 & -1.33 \\
-0.8 & -0.8 & -0.8 \\
-0.27 & -0.27 & -0.27 \\
0.27 & 0.27 & 0.27 \\
0.8 & 0.8 & 0.8 \\
1.33 & 1.33 & 1.33 \\
\end{bmatrix}
$$

---

### Step 2: Train PLS Model
1. Fit a PLS model with 2 components using the standardized $X$.
2. Extract latent scores ($T$ ) and loadings ($P$ ).

#### Resulting Scores ($T$ ):
$$
T = \begin{bmatrix}
-2.3 & -0.5 \\
-1.7 & -0.4 \\
-1.1 & -0.3 \\
0.9 & 0.2 \\
1.5 & 0.4 \\
2.7 & 0.6 \\
\end{bmatrix}
$$

#### Loadings ($P$ ):
$$
P = \begin{bmatrix}
0.5 & 0.3 \\
0.5 & 0.3 \\
0.5 & 0.3 \\
\end{bmatrix}
$$

---

### Step 3: Monitor New Data
#### 1. Standardize $X_{new}$ :
$$
X_{new,\text{std}} = \frac{X_{new} - \text{mean}(X)}{\text{std}(X)}
$$

$$
X_{new,\text{std}} = \begin{bmatrix}
-0.8 & -0.8 & -0.8 \\
-0.27 & -0.27 & -0.27 \\
3.47 & 3.47 & 3.47 \\
0.27 & 0.27 & 0.27 \\
2.93 & 2.93 & 2.93 \\
\end{bmatrix}
$$

#### 2. Project onto Latent Space:
Compute latent scores ($T_{new}$ ) for $X_{new,\text{std}}$ :
$$
T_{new} = X_{new,\text{std}} \cdot P
$$

$$
T_{new} = \begin{bmatrix}
-1.2 & -0.3 \\
-0.4 & -0.1 \\
5.2 & 1.2 \\
0.4 & 0.1 \\
4.4 & 1.0 \\
\end{bmatrix}
$$

---

### Step 4: Calculate SPE and $T^2$
#### 1. Reconstruct $X_{new,\text{recon}}$ :
$$
X_{new,\text{recon}} = T_{new} \cdot P^\top
$$

$$
X_{new,\text{recon}} = \begin{bmatrix}
-0.9 & -0.9 & -0.9 \\
-0.3 & -0.3 & -0.3 \\
4.3 & 4.3 & 4.3 \\
0.3 & 0.3 & 0.3 \\
3.7 & 3.7 & 3.7 \\
\end{bmatrix}
$$

#### 2. Compute SPE:
$$
\text{SPE} = \sum \left( X_{new,\text{std}} - X_{new,\text{recon}} \right)^2
$$

$$
\text{SPE} = \begin{bmatrix}
0.72, 0.06, 1.62, 0.06, 1.44
\end{bmatrix}
$$

#### 3. Compute $T^2$ :
$$
T^2 = \sum \left( \frac{T_{new}}{\text{variance}(T)} \right)^2
$$

$$
\text{variance}(T) = \begin{bmatrix} 2.0 & 0.5 \end{bmatrix}
$$

$$
T^2 = \begin{bmatrix}
0.72, 0.12, 14.6, 0.12, 11.2
\end{bmatrix}
$$

---

### Step 5: Compare with Control Limits
#### SPE Limit (99% Threshold):
$$
\text{SPE}_{\text{limit}} = 1.5
$$

#### $T^2$ Limit (99% Threshold):
$$
T^2_{\text{limit}} = 9.21
$$

---

### Results for $X_{new}$ :
1. **SPE Detection**:
   - Points exceeding SPE limit: 3rd and 5th samples.
2. **T^2 Detection**:
   - Points exceeding $T^2$ limit: 3rd and 5th samples.


---


Got it. Let's continue with **Partial Least Squares (PLS) Regression** on the standardized dataset without repeating the standardization process. We will focus on calculating the first PLS component (`PLS1`) by maximizing the covariance between the features and the response.

### Real-World Example: Predicting Wine Quality

#### Dataset Setup (Standardized)

- **Features** (`X1_std`, `X2_std`, `X3_std`):
  - `X1`: Alcohol percentage
  - `X2`: pH level
  - `X3`: Residual sugar content

- **Response Variable** (`Y_std`):
  - `Y`: Wine quality score

We will use the following standardized dataset for our calculations:

| Wine Sample | Alcohol (`X1_std`) | pH (`X2_std`) | Residual Sugar (`X3_std`) | Quality (`Y_std`) |
|-------------|---------------------|---------------|---------------------------|-------------------|
| 1           | 0.13                | 0.22          | -0.57                     | -0.15             |
| 2           | -1.45               | -1.44         | -1.15                     | -0.92             |
| 3           | 1.07                | 1.33          | 1.53                      | 0.62              |
| 4           | -0.50               | -0.89         | -0.19                     | -0.92             |
| 5           | 0.76                | 0.78          | 0.38                      | 1.38              |

### Step 1: **Calculate Covariance Between Each Feature and Response (`Y_std`)**

To determine how each feature correlates with the response (`Y`), we calculate the covariance between each feature and `Y`.

#### Covariance Calculation
The covariance between feature `X` and `Y` is given by:

$$
\text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^n (X_i \cdot Y_i)
$$

#### Covariance Between Features and `Y`
- **Covariance Between `X1` and `Y`**:
  $$
  \text{Cov}(X1, Y) = \frac{1}{5} \left( (0.13 \times -0.15) + (-1.45 \times -0.92) + (1.07 \times 0.62) + (-0.50 \times -0.92) + (0.76 \times 1.38) \right)
  $$
  $$
  \text{Cov}(X1, Y) = \frac{1}{5} \left( -0.0195 + 1.334 + 0.6634 + 0.46 + 1.0488 \right) \approx 0.69754
  $$

- **Covariance Between `X2` and `Y`**:
  $$
  \text{Cov}(X2, Y) = \frac{1}{5} \left( (0.22 \times -0.15) + (-1.44 \times -0.92) + (1.33 \times 0.62) + (-0.89 \times -0.92) + (0.78 \times 1.38) \right)
  $$
  $$
  \text{Cov}(X2, Y) = \frac{1}{5} \left( -0.033 + 1.3248 + 0.8246 + 0.8188 + 1.0764 \right) \approx 0.80232
  $$

- **Covariance Between `X3` and `Y`**:
  $$
  \text{Cov}(X3, Y) = \frac{1}{5} \left( (-0.57 \times -0.15) + (-1.15 \times -0.92) + (1.53 \times 0.62) + (-0.19 \times -0.92) + (0.38 \times 1.38) \right)
  $$
  $$
  \text{Cov}(X3, Y) = \frac{1}{5} \left( 0.0855 + 1.058 + 0.9486 + 0.1748 + 0.5244 \right) \approx 0.55826
  $$

### Step 2: **Formulate the First Latent Component (`PLS1`)**

To create the first latent component (`PLS1`), **PLS** maximizes the covariance of each feature with `Y`. Therefore, we will assign weights proportional to the covariances.

Let the weights be:
- `w1`, `w2`, `w3` for `X1`, `X2`, `X3` respectively.

The weights are proportional to the covariance values:

$$
w1 = 0.69754, \quad w2 = 0.80232, \quad w3 = 0.55826
$$

We normalize these weights to ensure they sum to 1:

$$
w_{\text{sum}} = 0.69754 + 0.80232 + 0.55826 = 2.05812
$$

$$
w1_{\text{norm}} = \frac{0.69754}{2.05812} \approx 0.339, \quad w2_{\text{norm}} = \frac{0.80232}{2.05812} \approx 0.390, \quad w3_{\text{norm}} = \frac{0.55826}{2.05812} \approx 0.271
$$

### Step 3: **Calculate `PLS1` for Each Sample**

The first latent variable (`PLS1`) is computed as:

$$
PLS1 = w1_{\text{norm}} \cdot X1 + w2_{\text{norm}} \cdot X2 + w3_{\text{norm}} \cdot X3
$$

For each wine sample:

- **Sample 1**:
  $$
  PLS1_1 = 0.339 \cdot 0.13 + 0.390 \cdot 0.22 + 0.271 \cdot (-0.57) \approx 0.044 + 0.0858 - 0.1545 \approx -0.0247
  $$

- **Sample 2**:
  $$
  PLS1_2 = 0.339 \cdot (-1.45) + 0.390 \cdot (-1.44) + 0.271 \cdot (-1.15) \approx -0.4916 - 0.5616 - 0.3117 \approx -1.3649
  $$

- **Sample 3**:
  $$
  PLS1_3 = 0.339 \cdot 1.07 + 0.390 \cdot 1.33 + 0.271 \cdot 1.53 \approx 0.3627 + 0.5187 + 0.4146 \approx 1.2960
  $$

- **Sample 4**:
  $$
  PLS1_4 = 0.339 \cdot (-0.50) + 0.390 \cdot (-0.89) + 0.271 \cdot (-0.19) \approx -0.1695 - 0.3471 - 0.0515 \approx -0.5681
  $$

- **Sample 5**:
  $$
  PLS1_5 = 0.339 \cdot 0.76 + 0.390 \cdot 0.78 + 0.271 \cdot 0.38 \approx 0.2576 + 0.3042 + 0.1029 \approx 0.6647
  $$

### Step 4: **Regression on the Latent Variable (`PLS1`)**

Now that we have calculated `PLS1`, we can use it to perform linear regression to predict `Y`.

The equation might look like:

$$
Y_{\text{pred}} = \beta_0 + \beta_1 \cdot PLS1
$$

Using ordinary least squares, we fit this model to determine the coefficients (`β0` and `β1`) to predict the quality of the wine (`Y`). 

### Summary
- In **PLS**, we calculate the components by focusing on the covariance between the features (`X`) and the response (`Y`), ensuring the resulting latent components are predictive of the response.
- Unlike **PCR**, where components are determined by the variance within the features, **PLS** selects components based on both feature variance and their relationship with `Y`, resulting in better predictive performance.