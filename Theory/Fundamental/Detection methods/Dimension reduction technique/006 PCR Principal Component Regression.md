### PCR
### 1. **Goal of PCR**
   - **Example**: Suppose you are working with a chemical manufacturing process dataset containing 100 different sensor measurements (features) that indicate various aspects of the process. Your goal is to predict the quality of the product.
     - Many sensors may be correlated (e.g., temperature and pressure sensors may influence each other), leading to multicollinearity. PCR’s goal here would be to reduce these correlated features into a smaller number of independent components to make regression more robust.

### 2. **Step-by-Step Procedure of PCR**
   
   **Example**:
   - Imagine a dataset containing three features: `Temperature`, `Pressure`, and `Humidity`. We will follow the PCR steps:

   1. **Standardize the Data**:
      - Standardizing ensures `Temperature`, `Pressure`, and `Humidity` are on the same scale.
      - Example: Original `Temperature` ranges from 30-150°C, `Pressure` from 10-500 kPa, `Humidity` from 0-100%. Standardizing makes their mean 0 and standard deviation 1.

   2. **Calculate Principal Components**:
      - Suppose we calculate the principal components (using SVD or another technique) and find that:
        - `PC1` explains 70% of the variance.
        - `PC2` explains 20%.
        - `PC3` explains 10%.

   3. **Select the Principal Components**:
      - If we choose to keep `PC1` and `PC2`, they explain 90% of the total variance, which is a good balance between reducing complexity and retaining information.

   4. **Perform Regression**:
      - Perform regression on `PC1` and `PC2` rather than the original features (`Temperature`, `Pressure`, `Humidity`). This way, the regression is not affected by the original multicollinearity among these features.

### 3. **When to Use PCR**
   
   **Example**:
   - **Multicollinearity**: In a financial dataset containing `Inflation Rate`, `Interest Rate`, and `Currency Exchange Rate`, these features are often correlated. If you want to predict GDP growth, using PCR helps reduce this multicollinearity by transforming these correlated features into principal components.
   - **High Dimensionality**: In genomic studies, you may have thousands of genes (features) with only a few hundred samples (observations). Running a typical regression would be computationally complex and prone to overfitting, but PCR can help by selecting a manageable number of components.
   - **Feature Reduction**: Suppose you have a marketing dataset with 200 variables about customer demographics, interests, and behaviors. Instead of using all 200 features, PCR helps reduce it to, say, 10 components that explain most of the variation in customer behavior.

### 4. **Why PCR is Needed**

   **Example**:
   - **Avoid Overfitting**: Imagine a housing dataset where you have 50 features (like `location`, `size`, `amenities`, etc.) but only 100 observations (houses). A standard regression would likely overfit, trying to perfectly fit all 50 variables. By reducing the number of predictors to a handful of principal components, PCR simplifies the model.
   - **Addressing Multicollinearity**: If two features, `Age` of a car and `Mileage`, are highly correlated, a standard regression might have trouble determining their individual effects on `Price`. PCR transforms these into independent components, allowing for a more stable model.
   - **Dimensionality Reduction**: A medical dataset may have 200 health indicators. PCR helps reduce these to, say, 20 components while retaining the most important information about a patient's health, making subsequent analysis faster and easier.

### 5. **Strengths of PCR**

   **Example**:
   - **Reduces Multicollinearity**: In a retail sales dataset, `Advertising Spend` across multiple media (TV, radio, online) may be correlated. PCR creates uncorrelated components from these media spend data, leading to a stable model that avoids issues with multicollinearity.
   - **Dimensionality Reduction**: In an image recognition task, where each pixel is a feature, there are thousands of features per image. PCR can reduce the number of features to a small number of components, significantly simplifying computation while retaining visual information.
   - **Flexibility**: Suppose you are analyzing customer satisfaction with thousands of survey questions, but you only have a limited number of survey responses. PCR allows you to reduce the features and still proceed with regression without having to collect more responses.

### 6. **Weaknesses of PCR**

   **Example**:
   - **Ignores the Response Variable**: Suppose in a medical study you want to predict `Disease Severity` based on patient features like `Age`, `Blood Pressure`, and `Heart Rate`. PCR selects principal components that best capture variance among features like `Blood Pressure` and `Heart Rate` but does not consider whether those components are actually predictive of `Disease Severity`. This could result in a model that doesn't perform well.
   - **Interpretability**: If a component is derived as a mix of `Income`, `Education Level`, and `Age`, interpreting how changes in one variable affect the target (`Loan Default Risk`) becomes challenging. It's less intuitive compared to directly interpreting coefficients of the original features.
   - **Potential Loss of Information**: If a dataset has some features (`Blood Sugar Level`, `Cholesterol Level`) that are not major sources of variance but are critical for predicting heart disease, PCR may drop these features during component selection because they don’t contribute much to the overall variance.

Let's explain how **PLS** and **PCR** differ, using a numerical example. Suppose we have three features (`X1`, `X2`, `X3`) and one response variable (`Y`). We'll assign some data values to illustrate this difference.

### Numerical Example Setup

- Features (`X1`, `X2`, `X3`):
  - `X1`: [1, 2, 3, 4, 5]
  - `X2`: [10, 20, 30, 40, 50]
  - `X3`: [5, 4, 3, 2, 1]

- Response (`Y`):
  - `Y`: [1, 4, 9, 16, 25]  (Quadratic relationship with `X1`)

We can clearly see that `Y` has a **quadratic relationship** with `X1`, while `X2` and `X3` have linear and inverse linear relationships respectively, but these relationships are not as strong as that with `X1`.

#### Step-by-Step Comparison of PCR and PLS:

### **Step 1: Principal Component Regression (PCR)**
  
1. **Standardize the Features**: In PCR, we first standardize the features (`X1`, `X2`, `X3`) to have mean 0 and standard deviation 1.

   - After standardization:
     - `X1_std`: [-1.41, -0.71, 0, 0.71, 1.41]
     - `X2_std`: [-1.41, -0.71, 0, 0.71, 1.41]
     - `X3_std`: [1.41, 0.71, 0, -0.71, -1.41]

2. **Calculate Principal Components**:
   - Principal components are computed based on the variance in the data.
   - Suppose the first principal component (`PC1`) is primarily influenced by `X2` because `X2` has the highest range and therefore contributes the most variance.
   - **PC1** could be something like:
     - `PC1` = `0.9*X2_std + 0.2*X1_std - 0.1*X3_std`

3. **Interpretation**:
   - Since **PCR** doesn't consider the response variable (`Y`), it selects **PC1** based on the largest variance in features, which may come from `X2`. However, `X2` does not have a strong relationship with `Y`. Therefore, PCR may not give the best components for predicting `Y`.

### **Step 2: Partial Least Squares (PLS) Regression**

1. **Standardize the Features**: Like PCR, PLS also standardizes the features, resulting in the same standardized values.

2. **Calculate PLS Components**:
   - PLS calculates components by **maximizing the covariance** between the features (`X`) and the response (`Y`).
   - In this case, PLS will look for combinations of `X1`, `X2`, and `X3` that are highly correlated with `Y`.
   - Since `Y` is clearly related to `X1` (quadratic relationship), the first PLS component (`PLS1`) will strongly weight `X1`.
   - For example, **PLS1** could be:
     - `PLS1` = `0.9*X1_std + 0.05*X2_std - 0.05*X3_std`

3. **Interpretation**:
   - **PLS** focuses on **maximizing the relationship with `Y`**, which means that `PLS1` places a much stronger weight on `X1` (the feature most predictive of `Y`) compared to `X2` and `X3`.
   - This results in a component that is more useful for predicting `Y` compared to the principal component from PCR.

### **Key Differences Illustrated with Numbers**

- In **PCR**, the first component (`PC1`) was dominated by `X2` because of its high variance, but this component was not very useful for predicting `Y` (since `Y` is more strongly related to `X1`).
- In **PLS**, the first component (`PLS1`) is dominated by `X1`, which has a strong relationship with `Y`, making it much more effective for prediction.

#### Summary

- **PCR** selects components based solely on the **variance** in `X`. This means it might select components that don't contribute much to predicting `Y` if those components have higher variance in `X` alone.
- **PLS** selects components by **maximizing covariance with `Y`**, which means it prioritizes the components that are actually helpful in predicting the response variable.

This difference makes **PLS** often more effective when the goal is to create a model that is directly focused on predicting `Y`, especially in cases where important predictive features might have low variance and hence be overlooked by PCR.