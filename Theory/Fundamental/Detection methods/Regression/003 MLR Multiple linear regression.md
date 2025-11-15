### Multiple linear regression:
 is a statistical method used to model the relationship between one dependent variable ($Y$ ) and multiple independent variables ($X_1, X_2, \ldots, X_p$ ). It extends simple linear regression by allowing for more predictors.

### 1. **Model Equation**
The general form of a multiple linear regression model is:
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon
$$
Where:
- $Y$ : Dependent variable (outcome to predict)
- $X_1, X_2, \ldots, X_p$ : Independent variables (predictors)
- $\beta_0$ : Intercept (value of $Y$ when all $X_i = 0$ )
- $\beta_1, \beta_2, \ldots, \beta_p$ : Coefficients (effect of each $X_i$ on $Y$ , holding others constant)
- $\epsilon$ : Random error (captures variability not explained by predictors)

### 2. **Assumptions**
For valid inference, multiple regression makes the following assumptions:
1. **Linearity**: The relationship between $Y$ and each $X_i$ is linear.
2. **Independence**: Observations are independent of each other.
3. **Homoscedasticity**: The variance of residuals ($\epsilon$ ) is constant across all values of $X_i$.
4. **Normality**: Residuals are normally distributed.
5. **No Multicollinearity**: Predictors are not excessively correlated.

### 3. **Purpose**
- To understand how multiple factors (predictors) influence a single outcome.
- To predict values of $Y$ based on given values of $X_1, X_2, \ldots, X_p$.
- To identify significant predictors by analyzing coefficients.

### 4. **Interpretation**
- Each $\beta_i$ represents the expected change in $Y$ when $X_i$ increases by 1 unit, holding other predictors constant.
- The sign of $\beta_i$ (positive or negative) indicates the direction of the relationship.

### 5. **Fitting the Model**
- The coefficients ($\beta_i$ ) are estimated using the **least squares method**, which minimizes the sum of squared differences between actual and predicted values of $Y$.

### 6. **Applications**
- Predicting outcomes (e.g., sales, house prices).
- Understanding relationships between variables in areas like economics, healthcare, and manufacturing. 

Here’s an example of applying **multiple linear regression** in the context of **control charts** for monitoring a manufacturing process:

---

### **Scenario**
You are monitoring a production process where the dependent variable ($Y$ ) is the **quality index** of a product. The independent variables ($X_1, X_2, X_3$ ) represent process parameters like temperature, pressure, and speed.

| Timestamp | Temperature ($X_1$ ) | Pressure ($X_2$ ) | Speed ($X_3$ ) | Quality Index ($Y$ ) |
|-----------|-------------------------|---------------------|------------------|-----------------------|
| 1         | 100                    | 20                  | 50               | 85                    |
| 2         | 102                    | 21                  | 48               | 83                    |
| 3         | 98                     | 19                  | 52               | 86                    |
| 4         | 101                    | 20                  | 49               | 84                    |

---

### **Step 1: Model**
The regression model is:
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \epsilon
$$
Where:
- $Y$ : Quality Index (target variable)
- $X_1, X_2, X_3$ : Process parameters (predictors)

---

### **Step 2: Fit the Model**
We use the least squares method to estimate coefficients $\beta_0, \beta_1, \beta_2, \beta_3$.

---

### **Step 3: Coefficients (Example Values)**
Assume the fitted model yields:
$$
Y = 10 + 0.3X_1 + 1.5X_2 - 0.2X_3
$$

---

### **Step 4: Interpretation**
1. **Intercept ($\beta_0 = 10$ )**:
   - When $X_1, X_2, X_3 = 0$ , the predicted Quality Index is 10.
2. **Temperature ($\beta_1 = 0.3$ )**:
   - A 1-unit increase in temperature raises the Quality Index by 0.3, holding other variables constant.
3. **Pressure ($\beta_2 = 1.5$ )**:
   - A 1-unit increase in pressure raises the Quality Index by 1.5, holding other variables constant.
4. **Speed ($\beta_3 = -0.2$ )**:
   - A 1-unit increase in speed reduces the Quality Index by 0.2, holding other variables constant.

---

### **Step 5: Application in Control Charts**
1. Use the model to predict the expected Quality Index for new process data.
2. Plot the residuals ($Y - \hat{Y}$ ) on a control chart:
   - If residuals fall outside control limits, it indicates that process deviations cannot be explained by the model alone.
3. Use significant predictors ($X_1, X_2, X_3$ ) to adjust process settings and optimize quality.

---

This integrates regression analysis into process monitoring, enabling dynamic quality control beyond static control limits.

### **Pros of Using Multiple Linear Regression in Control Charts**
1. **Enhanced Monitoring**:
   - Accounts for relationships between multiple variables, improving the understanding of process dynamics.
   - Detects more complex patterns of variation that univariate control charts cannot.

2. **Predictive Capabilities**:
   - Predicts the expected outcome ($Y$ ) based on current process conditions.
   - Identifies trends or shifts before they lead to defects.

3. **Dimensionality Reduction**:
   - Summarizes the influence of multiple variables into a single model, reducing the need for separate univariate control charts.

4. **Root Cause Analysis**:
   - Highlights which predictors ($X_1, X_2, \ldots$ ) significantly affect the process outcome ($Y$ ), aiding in diagnosing process issues.

5. **Flexibility**:
   - Can be updated dynamically as new data becomes available, making it adaptive to process changes.

---

### **Cons of Using Multiple Linear Regression in Control Charts**
1. **Assumption Sensitivity**:
   - Relies on assumptions like linearity, independence, and homoscedasticity, which may not always hold in real-world processes.

2. **Multicollinearity**:
   - High correlation among predictors can distort coefficient estimates and reduce interpretability.

3. **Overfitting Risk**:
   - Including too many predictors or irrelevant variables may lead to overfitting, reducing model reliability.

4. **Complexity**:
   - More complex than traditional univariate control charts, requiring statistical knowledge and computational tools.

5. **Residual Dependence**:
   - Control charts based on residuals assume that the regression model fully captures the process. If the model is misspecified, residuals may not reflect actual deviations.

6. **Implementation Cost**:
   - Requires additional effort for data collection, model fitting, and validation, which may not be feasible for all production environments.

---

While multiple linear regression enhances process monitoring and anomaly detection in control charts, its effectiveness depends on proper data preparation, assumption testing, and model validation.

### 1. **Assumption Sensitivity**
Multiple linear regression relies on several assumptions. When these assumptions are violated, the model may not accurately represent the process.

#### Assumptions:
- **Linearity**: The relationship between predictors and the outcome must be linear.
- **Independence**: Observations should not influence one another.
- **Homoscedasticity**: Residual variance should be constant across all predictor levels.
- **Normality**: Residuals should follow a normal distribution.

#### Example:
- Suppose you are monitoring product quality ($Y$ ) with predictors like machine temperature ($X_1$ ) and pressure ($X_2$ ).
  - If the relationship between $Y$ and $X_1$ is quadratic (e.g., quality improves up to a point but declines beyond it), a linear regression model will misfit the data.
  - Violating homoscedasticity (e.g., residuals grow larger at high values of $X_2$ ) can distort hypothesis tests and confidence intervals.

#### Impact:
- Results may show biased coefficients or misleading significance levels, making it harder to trust the control chart’s residual-based decisions.

---

### 2. **Multicollinearity**
Multicollinearity occurs when predictors are highly correlated, causing issues in coefficient estimation.

#### Example:
- Predictors: $X_1$ = Machine Speed, $X_2$ = Product Throughput
  - These variables are strongly correlated because increasing speed increases throughput.
  - The regression model struggles to distinguish their independent contributions to product quality ($Y$ ).

#### Impact:
- Coefficients for $X_1$ and $X_2$ may become unstable or counterintuitive.
  - $X_1$ : $+3.5$ (suggesting a positive relationship)
  - $X_2$ : $-3.0$ (suggesting a negative relationship, despite throughput logically improving quality).

#### Practical Issues:
- The model misguides process engineers about which factor to adjust, potentially causing incorrect decisions.

---

### 3. **Overfitting Risk**
Overfitting happens when the model captures noise in the data instead of the underlying relationship.

#### Example:
- Predictors: $X_1 =$ Operator Experience, $X_2 =$ Ambient Temperature, $X_3 =$ Machine Pressure
  - Adding irrelevant variables like ambient temperature ($X_2$ ) can cause the model to adapt to random fluctuations.
  - The model fits the training data perfectly but fails to generalize to new data.

#### Impact:
- The control chart will flag deviations based on irrelevant noise, increasing false alarms.
- Engineers waste time investigating “issues” that are not actual process problems.

---

### 4. **Complexity**
Using regression for control charts adds complexity compared to traditional methods.

#### Example:
- A univariate control chart for quality might simply track $Y$ with control limits ($UCL, LCL$ ).
- Regression requires data collection for multiple predictors, fitting the model, validating assumptions, and updating it periodically.

#### Practical Challenges:
- Requires expertise in regression modeling and statistical software.
- Errors in any step (e.g., failing to test for assumptions) can lead to incorrect process monitoring.

#### Impact:
- Smaller manufacturing units or teams without dedicated data scientists might struggle to implement this effectively.

---

### 5. **Residual Dependence**
Regression-based control charts rely on residuals ($Y - \hat{Y}$ ) to detect anomalies. If the model is poorly specified, the residuals won't accurately represent true deviations.

#### Example:
- Predictors: $X_1 =$ Machine Speed, $X_2 =$ Machine Temperature
  - If $Y$ is also influenced by humidity ($X_3$ ), which is missing from the model, residuals will capture this unaccounted variation.

#### Impact:
- Control charts will show frequent out-of-control signals caused by unmodeled factors, even when the process is under control.
- Engineers may adjust the wrong parameters, worsening the process.

---

### 6. **Implementation Cost**
Implementing regression-based control charts requires significant resources.

#### Example:
- Data collection: Sensors must measure $X_1, X_2, \ldots$ , which can be costly.
- Model development: Engineers or data scientists must build and validate the regression model.
- Maintenance: Regularly updating the model is essential as process dynamics evolve.

#### Impact:
- High upfront and ongoing costs may not justify the benefits for smaller operations.
- Teams with limited budgets might find it impractical compared to simpler control chart methods. 

---

### **Comparison of Multiple Linear Regression (MLR), Principal Component Regression (PCR), Partial Least Squares (PLS), and Principal Component Analysis (PCA) in Control Charts**

#### **1. Multicollinearity Handling**
| **Method** | **Multicollinearity Handling** | **Explanation** |
|------------|--------------------------------|------------------|
| **MLR**    | Poor                          | Coefficients become unstable if predictors are highly correlated. |
| **PCR**    | Excellent                     | Reduces predictors to uncorrelated principal components before regression. |
| **PLS**    | Excellent                     | Combines predictors into latent components correlated with the outcome variable. |
| **PCA**    | Excellent                     | Identifies uncorrelated principal components but does not directly model the outcome variable. |

---

#### **2. Interpretability**
| **Method** | **Interpretability** | **Explanation** |
|------------|----------------------|------------------|
| **MLR**    | High                 | Coefficients directly indicate the effect of each predictor on the outcome. |
| **PCR**    | Moderate             | Principal components are linear combinations of predictors, making interpretation indirect. |
| **PLS**    | Moderate             | Latent components are interpretable but less intuitive than raw variables. |
| **PCA**    | Low                  | Principal components are mathematical constructs without direct process relevance. |

---

#### **3. Prediction Accuracy**
| **Method** | **Prediction Accuracy** | **Explanation** |
|------------|--------------------------|------------------|
| **MLR**    | Good (if assumptions hold) | Assumes linearity and low multicollinearity for reliable predictions. |
| **PCR**    | Good                     | Removes noise from multicollinearity, improving predictions in complex datasets. |
| **PLS**    | Excellent                | Optimized for predicting the outcome by balancing variance in predictors and outcome. |
| **PCA**    | Not Applicable           | PCA is not a predictive method but aids in dimensionality reduction. |

---

#### **4. Complexity**
| **Method** | **Complexity** | **Explanation** |
|------------|----------------|------------------|
| **MLR**    | Low            | Easy to implement and interpret for small datasets. |
| **PCR**    | Moderate       | Requires PCA preprocessing before regression. |
| **PLS**    | High           | Requires latent variable modeling, which is computationally intensive. |
| **PCA**    | Low            | Straightforward dimensionality reduction but lacks a predictive component. |

---

#### **5. Control Chart Use**
| **Method** | **Use in Control Charts** | **Explanation** |
|------------|---------------------------|------------------|
| **MLR**    | Residual-based            | Monitors residuals after accounting for predictors’ effects. |
| **PCR**    | Residual-based            | Uses regression residuals from reduced components for monitoring. |
| **PLS**    | Residual-based + Component Monitoring | Monitors both residuals and latent variables for better process understanding. |
| **PCA**    | Component Monitoring      | Monitors the principal components representing process variability. |

---

### **Summary**
- **MLR**: Best for small datasets with no multicollinearity; straightforward and interpretable but sensitive to assumption violations.
- **PCR**: Suitable when predictors are highly correlated; reduces dimensionality while retaining variability.
- **PLS**: Ideal for predictive modeling when outcome relevance and multicollinearity are key concerns.
- **PCA**: Effective for exploratory analysis and dimensionality reduction but lacks direct outcome prediction.

#### Recommendation:
- Use **PLS** for predictive control charts in complex systems.
- Use **PCR** or **PCA** if the goal is dimensionality reduction and exploratory monitoring.
- Use **MLR** for simpler processes with manageable multicollinearity.


