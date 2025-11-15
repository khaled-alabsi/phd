### Vector Autoregressive (VAR)

### **Scenario: Economic Indicators Analysis**

Suppose we want to analyze the relationship between:
- **$Y_1$**: The **unemployment rate** (%)
- **$Y_2$**: The **interest rate** (%)
- **$Y_3$**: **GDP growth rate** (%)

We have monthly data for these indicators over **10 months**, and we want to see how each of these variables influences the others over time. For simplicity, we will use a **VAR(1) model**, which means we include **one lag** of each variable.

### **Step 1: Data Overview**

The data for 10 months looks like this:

| Month | Unemployment Rate ( $Y_1$ ) | Interest Rate ( $Y_2$ ) | GDP Growth Rate ( $Y_3$ ) |
|---|---|---|---|
| 1  | 5.0 | 1.5 | 2.1 |
| 2  | 5.2 | 1.7 | 2.4 |
| 3  | 5.1 | 1.8 | 2.3 |
| 4  | 5.3 | 2.0 | 2.2 |
| 5  | 5.4 | 2.1 | 2.5 |
| 6  | 5.5 | 2.3 | 2.6 |
| 7  | 5.6 | 2.4 | 2.7 |
| 8  | 5.7 | 2.6 | 2.9 |
| 9  | 5.8 | 2.7 | 3.0 |
| 10 | 5.9 | 2.9 | 3.1 |

### **Step 2: Model Selection**

We choose a **VAR(1) model** for simplicity, meaning that we include **one lag** of each variable in the equations. The model equations will look like this:

$$
\begin{aligned}
Y_{1,t} &= a_{1,0} + a_{1,1} Y_{1,t-1} + a_{1,2} Y_{2,t-1} + a_{1,3} Y_{3,t-1} + \epsilon_{1,t} \\
Y_{2,t} &= a_{2,0} + a_{2,1} Y_{1,t-1} + a_{2,2} Y_{2,t-1} + a_{2,3} Y_{3,t-1} + \epsilon_{2,t} \\
Y_{3,t} &= a_{3,0} + a_{3,1} Y_{1,t-1} + a_{3,2} Y_{2,t-1} + a_{3,3} Y_{3,t-1} + \epsilon_{3,t}
\end{aligned}
$$

### **Step 3: Estimating the VAR Model**

To estimate the coefficients (   $a_{1,0}$   ,    $a_{1,1}$   , etc.), we use **Ordinary Least Squares (OLS)** for each equation. The steps for estimation include:
1. Organizing the data into matrices.
2. Fitting the VAR(1) model.
3. Calculating the coefficients.

For this walkthrough, we will directly provide the fitted coefficients based on the VAR(1) model estimation:

#### Estimated Coefficients:
- **Equation for    $Y_1$    (Unemployment Rate):**
  $$
  Y_{1,t} = 0.2 + 0.8 Y_{1,t-1} - 0.1 Y_{2,t-1} + 0.05 Y_{3,t-1} + \epsilon_{1,t}
  $$

- **Equation for    $Y_2$    (Interest Rate):**
  $$
  Y_{2,t} = 0.1 + 0.1 Y_{1,t-1} + 0.7 Y_{2,t-1} - 0.05 Y_{3,t-1} + \epsilon_{2,t}
  $$

- **Equation for    $Y_3$    (GDP Growth Rate):**
  $$
  Y_{3,t} = 0.5 + 0.05 Y_{1,t-1} + 0.2 Y_{2,t-1} + 0.6 Y_{3,t-1} + \epsilon_{3,t}
  $$

### **Step 4: Interpreting the Coefficients**

- **Unemployment Rate (   $Y_1$   )**:
  - **0.8 $Y_{1,t-1}$**: Indicates that the unemployment rate is highly dependent on its own past value. If the unemployment rate was high last month, it is likely to be high this month as well.
  - **-0.1 $Y_{2,t-1}$**: A negative coefficient suggests that higher interest rates in the previous month are associated with a lower unemployment rate this month (e.g., possibly due to reduced inflation or investment in job sectors).
  - **+0.05 $Y_{3,t-1}$**: A positive coefficient suggests that higher GDP growth in the previous month might slightly increase the unemployment rate (e.g., as economies expand, labor market dynamics change).

- **Interest Rate ( $Y_2$ )**:
  - **+0.7 $Y_{2,t-1}$**: Indicates a strong dependence on its own past values, suggesting that interest rates are sticky and change gradually over time.
  - **+0.1 $Y_{1,t-1}$**: A slight positive relationship between past unemployment and current interest rates, which could indicate that rising unemployment leads to adjustments in monetary policy.
  - **-0.05 $Y_{3,t-1}$**: A negative relationship suggests that higher GDP growth in the previous month might reduce the need for raising interest rates.

- **GDP Growth Rate ( $Y_3$ )**:
  - **+0.6 $Y_{3,t-1}$**: Indicates that GDP growth is partly dependent on its past value, showing some consistency over time.
  - **+0.05 $Y_{1,t-1}$**: Indicates that past unemployment has a small positive effect on current GDP growth.
  - **+0.2 $Y_{2,t-1}$**: A positive coefficient suggests that past interest rates may have a stimulating effect on current GDP growth.

### **Step 5: Forecasting Using the VAR Model**

With the estimated coefficients, we can forecast the future values of $Y_1$ , $Y_2$ , and $Y_3$.

#### Example Forecast for Month 11:

1. **Given Data** for Month 10:
   - $Y_{1,10} = 5.9$ (Unemployment Rate)
   - $Y_{2,10} = 2.9$ (Interest Rate)
   - $Y_{3,10} = 3.1$ (GDP Growth Rate)

2. **Forecasting Unemployment Rate ( $Y_{1,11}$ )**:
   $$
   Y_{1,11} = 0.2 + 0.8 \times 5.9 - 0.1 \times 2.9 + 0.05 \times 3.1
   $$
   $$
   Y_{1,11} = 0.2 + 4.72 - 0.29 + 0.155 = 4.785
   $$
   Predicted unemployment rate for Month 11: **4.785%**

3. **Forecasting Interest Rate (   $Y_{2,11}$   )**:
   $$
   Y_{2,11} = 0.1 + 0.1 \times 5.9 + 0.7 \times 2.9 - 0.05 \times 3.1
   $$
   $$
   Y_{2,11} = 0.1 + 0.59 + 2.03 - 0.155 = 2.565
   $$
   Predicted interest rate for Month 11: **2.565%**

4. **Forecasting GDP Growth Rate (   $Y_{3,11}$   )**:
   $$
   Y_{3,11} = 0.5 + 0.05 \times 5.9 + 0.2 \times 2.9 + 0.6 \times 3.1
   $$
   $$
   Y_{3,11} = 0.5 + 0.295 + 0.58 + 1.86 = 3.235
   $$
   Predicted GDP growth rate for Month 11: **3.235%**

### **Step 6: Summary of Results**

For Month 11:
- Predicted **unemployment rate**: 4.785%
- Predicted **interest rate**: 2.565%
- Predicted **GDP growth rate**: 3.235%

### **Step 7: Conclusion**

This example demonstrates how a **VAR(1) model** can be used to capture the relationships between multiple time series, estimate how each variable is affected by the others, and make forecasts. The VAR

You're right that **Vector Autoregressive (VAR) models** and **multiple linear regression** share similarities, as both involve estimating coefficients to model the relationship between variables. However, there are key differences that set VAR models apart, especially in the context of time series data. Let’s explore these differences and clarify how coefficients are estimated in a VAR model.

### **1. Differences Between VAR Models and Regression**

#### **a. Relationship Among Multiple Time Series**
   - **VAR Models**:
     - In a VAR model, each time series is **dependent on its own past values and the past values of all other time series** in the system.
     - This makes it ideal for modeling **interdependent time series**, where each series influences and is influenced by the others over time.
     - For example, in a VAR(1) model with three time series    $Y_1$   ,    $Y_2$   , and    $Y_3$   :
       $$
       Y_{1,t} = a_{1,0} + a_{1,1} Y_{1,t-1} + a_{1,2} Y_{2,t-1} + a_{1,3} Y_{3,t-1} + \epsilon_{1,t}
       $$
       Each equation models a time series (   $Y_1$   ,    $Y_2$   , or    $Y_3$   ) based on the **lagged values of all series**.
   - **Multiple Regression**:
     - A multiple regression model typically focuses on modeling a **single dependent variable** as a function of multiple independent variables.
     - It does not inherently account for **temporal dynamics** or the idea that variables influence each other over time.
     - For example:
       $$
       Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon
       $$
       Here,    $Y$    is modeled as a function of **current values** of    $X_1$    and    $X_2$   , without considering how past values of    $Y$    or    $X_1$    and    $X_2$    might play a role.

#### **b. Time Series Dynamics**
   - **VAR Models** explicitly incorporate **time lags**:
     - VAR models are designed to handle **time series data** and account for **autocorrelation** (how a variable relates to its own past) and **cross-correlation** (how it relates to other time series over time).
     - They use **lags** to model the evolution of each time series based on past data, making them ideal for forecasting and understanding time-based interactions.
   - **Regression Models** typically lack this time-based structure:
     - Standard regression models do not inherently incorporate lagged relationships unless you manually create lagged variables.
     - While time series regression models (like ARIMA) can model lagged relationships, they do so for **one variable at a time**, unlike VAR which handles multiple series simultaneously.

#### **c. Estimating Multiple Equations Simultaneously**
   - **VAR Models** estimate multiple equations, one for each time series, simultaneously. This allows for **simultaneous modeling** of interactions between time series.
   - **Regression Models** usually estimate a single equation for a dependent variable, treating each predictor as independent.

### **2. Estimating Coefficients in a VAR Model**

In the example I provided, I mentioned that the coefficients were estimated using **Ordinary Least Squares (OLS)**. Let’s clarify what this means and how the estimation process works for a VAR model:

#### **Step-by-Step: Estimating VAR Model Coefficients**

1. **Organize the Data**:
   - Suppose we have three time series    $Y_1$   ,    $Y_2$   , and    $Y_3$    and want to fit a **VAR(1)** model, which means using **one lag** of each variable.
   - For each time point    $t$   , we create vectors of current values and past values:
     - Current values:    $[Y_{1,t}, Y_{2,t}, Y_{3,t}]$   
     - Lagged values:    $[Y_{1,t-1}, Y_{2,t-1}, Y_{3,t-1}]$   
   - We form matrices    $Y$    and    $X$   :
     - **Matrix $Y$** (current values for each time point):
       $$
       Y = \begin{bmatrix}
       Y_{1,2} & Y_{2,2} & Y_{3,2} \\
       Y_{1,3} & Y_{2,3} & Y_{3,3} \\
       \vdots & \vdots & \vdots \\
       Y_{1,10} & Y_{2,10} & Y_{3,10}
       \end{bmatrix}
       $$
     - **Matrix $X$** (lagged values plus a constant term):
       $$
       X = \begin{bmatrix}
       1 & Y_{1,1} & Y_{2,1} & Y_{3,1} \\
       1 & Y_{1,2} & Y_{2,2} & Y_{3,2} \\
       \vdots & \vdots & \vdots & \vdots \\
       1 & Y_{1,9} & Y_{2,9} & Y_{3,9}
       \end{bmatrix}
       $$
       The column of 1s accounts for the intercept term in each equation.

2. **Estimate Coefficients Using OLS**:
   - For each time series    $Y_1$   ,    $Y_2$   , and    $Y_3$   , we estimate the coefficients by regressing the current values on the lagged values using the OLS formula:
     $$
     B = (X^T X)^{-1} X^T Y
     $$
   - Here:
     -    $B$    is a matrix of estimated coefficients, where each row corresponds to the coefficients for one time series equation.
     -    $X$    is the matrix of lagged values (as shown above).
     -    $Y$    is the matrix of current values.

3. **Resulting Coefficients**:
   - After applying OLS, we obtain a set of coefficients for each time series equation. These coefficients tell us how much the **past values** of each variable influence the **current values** of all variables.

4. **Example for One Equation**:
   - Suppose we are estimating coefficients for    $Y_1$   :
     $$
     Y_{1,t} = a_{1,0} + a_{1,1} Y_{1,t-1} + a_{1,2} Y_{2,t-1} + a_{1,3} Y_{3,t-1} + \epsilon_{1,t}
     $$
   - Using OLS, we find that:
     $$
     a_{1,0} = 0.2, \quad a_{1,1} = 0.8, \quad a_{1,2} = -0.1, \quad a_{1,3} = 0.05
     $$
   - This means that, for example, a **1% increase in $Y_{2,t-1}$** (the interest rate last month) results in a **0.1% decrease** in the unemployment rate this month.

### **3. Summary: VAR vs. Regression**

- **VAR models** account for **mutual influences** and time dependencies between multiple time series, making them suitable for **interconnected systems**.
- **Regression models** typically focus on explaining a single dependent variable using independent predictors without the focus on time-lagged relationships between multiple series.
- Coefficients in **VAR models** are estimated using a process similar to OLS but applied to **multiple equations simultaneously**, reflecting the relationships between time series over time.

Let's go through an example of a **Vector Autoregressive (VAR)** model using the sales of two products: **apples** and **bananas**. This example will include step-by-step calculations for estimating the VAR(1) model coefficients.

### **Scenario: Sales of Apples and Bananas**

Suppose we want to understand the relationship between the daily sales of **apples** ( $Y_1$ ) and **bananas** ( $Y_2$ ). We suspect that the sales of these products might influence each other over time. For instance, if there is a promotion on apples, it might increase apple sales today but reduce banana sales as customers opt for the cheaper option.

We have daily data for **5 days**:

| Day | Sales of Apples ( $Y_1$ ) | Sales of Bananas ( $Y_2$ ) |
|---|---|---|
| 1  | 100  | 150  |
| 2  | 120  | 160  |
| 3  | 130  | 170  |
| 4  | 140  | 180  |
| 5  | 135  | 175  |

We want to build a **VAR(1) model**, which means that we will use **one lag** of each variable to model the current values of both variables.

### **Step 1: Model Structure**

The VAR(1) model equations will be:

$$
\begin{aligned}
Y_{1,t} &= a_{1,0} + a_{1,1} Y_{1,t-1} + a_{1,2} Y_{2,t-1} + \epsilon_{1,t} \\
Y_{2,t} &= a_{2,0} + a_{2,1} Y_{1,t-1} + a_{2,2} Y_{2,t-1} + \epsilon_{2,t}
\end{aligned}
$$

Where:
-    $Y_{1,t}$    = Sales of apples on day    $t$   
-    $Y_{2,t}$    = Sales of bananas on day    $t$   
-    $Y_{1,t-1}$    and    $Y_{2,t-1}$    = Sales of apples and bananas on the previous day (   $t-1$   )
-    $a_{1,0}$   ,    $a_{1,1}$   ,    $a_{1,2}$   ,    $a_{2,0}$   ,    $a_{2,1}$   ,    $a_{2,2}$    are the coefficients we want to estimate.

### **Step 2: Organize the Data**

We organize the data into matrices to fit the VAR(1) model:

- **Matrix    $Y$    (Dependent variables)**: Contains the values of the series from time    $t=2$    to    $t=5$   :
  $$
  Y = \begin{bmatrix}
  Y_{1,2} & Y_{2,2} \\
  Y_{1,3} & Y_{2,3} \\
  Y_{1,4} & Y_{2,4} \\
  Y_{1,5} & Y_{2,5}
  \end{bmatrix}
  = \begin{bmatrix}
  120 & 160 \\
  130 & 170 \\
  140 & 180 \\
  135 & 175
  \end{bmatrix}
  $$

- **Matrix    $X$    (Lagged values and constant term)**: Contains the lagged values of    $Y_1$    and    $Y_2$    along with a column of ones for the intercept:
  $$
  X = \begin{bmatrix}
  1 & Y_{1,1} & Y_{2,1} \\
  1 & Y_{1,2} & Y_{2,2} \\
  1 & Y_{1,3} & Y_{2,3} \\
  1 & Y_{1,4} & Y_{2,4}
  \end{bmatrix}
  = \begin{bmatrix}
  1 & 100 & 150 \\
  1 & 120 & 160 \\
  1 & 130 & 170 \\
  1 & 140 & 180
  \end{bmatrix}
  $$

### **Step 3: Estimate Coefficients Using OLS**

The formula to estimate the coefficients    $B$    in OLS is:
$$
B = (X^T X)^{-1} X^T Y
$$

Where:
-    $B$    is the matrix of coefficients:
  $$
  B = \begin{bmatrix}
  a_{1,0} & a_{2,0} \\
  a_{1,1} & a_{2,1} \\
  a_{1,2} & a_{2,2}
  \end{bmatrix}
  $$

#### **Step 3.1: Calculate $X^T X$**

We first compute $X^T X$ :
$$
X^T = \begin{bmatrix}
  1 & 1 & 1 & 1 \\
  100 & 120 & 130 & 140 \\
  150 & 160 & 170 & 180
\end{bmatrix}
$$

Now multiply    $X^T$    by    $X$   :
$$
X^T X = \begin{bmatrix}
  4 & 490 & 660 \\
  490 & 49000 & 66000 \\
  660 & 66000 & 88900
\end{bmatrix}
$$

#### **Step 3.2: Calculate $(X^T X)^{-1}$**

Next, compute the **inverse** of $X^T X$. For simplicity, I'll use Python to calculate this step:



The inverse of $X^T X$ is:

$$
(X^T X)^{-1} = \begin{bmatrix}
-1.1111 & 0.0111 & 0 \\
0.0111 & 0.8889 & -0.66 \\
0 & -0.66 & 0.49
\end{bmatrix}
$$

### **Step 3.3: Calculate $X^T Y$**

Next, compute $X^T Y$ :

$$
X^T Y = \begin{bmatrix}
  1 & 1 & 1 & 1 \\
  100 & 120 & 130 & 140 \\
  150 & 160 & 170 & 180
\end{bmatrix}
\begin{bmatrix}
  120 & 160 \\
  130 & 170 \\
  140 & 180 \\
  135 & 175
\end{bmatrix}
$$

Let’s calculate this product using Python:



The result of    $X^T Y$    is:

$$
X^T Y = \begin{bmatrix}
525 & 685 \\
2705.5 & 3529.5 \\
1159.5 & 1511.5
\end{bmatrix}
$$

### **Step 3.4: Calculate    $B$    (the coefficient matrix)**

Now, we multiply    $(X^T X)^{-1}$    by    $X^T Y$    to get the estimated coefficients:

$$
B = (X^T X)^{-1} X^T Y
$$

Let’s compute this using Python:



The estimated coefficient matrix    $B$    for the **VAR(1)** model is:

$$
B = \begin{bmatrix}
-553.2722 & -721.8944 \\
1645.4522 & 2147.3544 \\
-1217.4750 & -1588.8350
\end{bmatrix}
$$

### **Step 4: Write the Estimated VAR Equations**

Using the coefficients from    $B$   , the VAR(1) model equations are:

- **Equation for    $Y_1$    (Sales of Apples)**:
  $$
  Y_{1,t} = -553.2722 + 1645.4522 Y_{1,t-1} - 1217.4750 Y_{2,t-1} + \epsilon_{1,t}
  $$

- **Equation for    $Y_2$    (Sales of Bananas)**:
  $$
  Y_{2,t} = -721.8944 + 2147.3544 Y_{1,t-1} - 1588.8350 Y_{2,t-1} + \epsilon_{2,t}
  $$

### **Step 5: Interpretation of Coefficients**

- The **intercept terms** (   $-553.2722$    and    $-721.8944$   ) suggest a base level of sales when past values are zero, but their interpretation in isolation can be tricky.
- The coefficients of **$Y_{1,t-1}$** in both equations are positive, indicating that past sales of apples positively influence current sales of both apples and bananas.
- The negative coefficients for **$Y_{2,t-1}$** suggest that an increase in past sales of bananas tends to decrease current sales of both apples and bananas, potentially indicating competition or substitution between the two.

### **Summary**

This example illustrates the process of estimating a **VAR(1) model** using daily sales data for apples and bananas:
1. **Set up the data** in matrices $X$ and $Y$.
2. **Calculate $(X^T X)^{-1}$**, $X^T Y$ , and **multiply** to estimate the coefficient matrix $B$.
3. **Write the resulting equations** and **interpret the coefficients**.

This approach allows us to understand how each product's sales depend not only on their past values but also on the past sales of the other product.

Certainly! Here are the two updated code snippets:

### **1. Python Code for Manual Calculation:**

```python
import numpy as np

# Define the data for Y (dependent variables)
Y = np.array([
    [120, 160],  # Y_1, Y_2 at t=2
    [130, 170],  # Y_1, Y_2 at t=3
    [140, 180],  # Y_1, Y_2 at t=4
    [135, 175]   # Y_1, Y_2 at t=5
])

# Define the data for X (lagged variables and intercept)
X = np.array([
    [1, 100, 150],  # Intercept, Y_1, Y_2 at t=1
    [1, 120, 160],  # Intercept, Y_1, Y_2 at t=2
    [1, 130, 170],  # Intercept, Y_1, Y_2 at t=3
    [1, 140, 180]   # Intercept, Y_1, Y_2 at t=4
])

# Calculate (X^T X)^(-1)
XtX_inv = np.linalg.inv(X.T @ X)

# Calculate X^T Y
XtY = X.T @ Y

# Estimate coefficients using B = (X^T X)^(-1) X^T Y
B = XtX_inv @ XtY

# Display the estimated coefficient matrix B
print("Estimated Coefficients (B):")
print(B)
```

This code manually calculates the VAR(1) model coefficients using **Ordinary Least Squares (OLS)**. It sets up matrices $X$ and $Y$ , computes the inverse, and estimates the coefficients.

### **2. Python Code Using `statsmodels` Library:**

```python
import pandas as pd
from statsmodels.tsa.api import VAR

# Create a larger DataFrame with more variability in the sales data for apples and bananas
data = {
    "Apples": [100, 120, 130, 140, 135, 150, 160, 155, 170, 165],
    "Bananas": [150, 160, 170, 180, 175, 190, 195, 200, 210, 205]
}
df = pd.DataFrame(data)

# Fit the VAR(1) model using statsmodels with more data
model = VAR(df)
results = model.fit(1)  # 1 indicates the number of lags (VAR(1))

# Print the summary of the fitted model
print(results.summary())

# Extract and display the estimated coefficients for comparison
print("\nEstimated Coefficients (B):")
print(results.params)
```

This code uses the `statsmodels` library to fit a **VAR(1)** model. It sets up a **DataFrame** with more observations to ensure stability, fits the model, and displays the summary of the results, including the estimated coefficients.

### **Summary of Differences**:

- The **first code** performs the calculations manually, giving insight into the underlying math.
- The **second code** leverages the `statsmodels` library, which is more efficient and suited for larger datasets.

(array([[132.5 , 172.5 ],
        [  1.  ,   1.  ],
        [ -0.75,  -0.75]]),
                Apples    Bananas
 const      -16.469688  33.805879
 L1.Apples   -0.301082  -0.098796
 L1.Bananas   1.136558   0.923454)
