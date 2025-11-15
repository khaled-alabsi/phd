### Boxplots

This code plots **boxplots** for **each feature column** (excluding the first 3 columns) in a DataFrame (`df_training_concated`), using the preprocessed data in another DataFrame (`pp`). Each feature gets its **own subplot**, arranged in a grid.


#### WHAT KIND OF PLOT THIS IS

**Boxplot (a.k.a. box-and-whisker plot)**

* Visualizes **distribution** of a feature
* Shows:

  * **Median**
  * **Interquartile range (IQR)**
  * **Whiskers** (usually 1.5 × IQR)
  * **Outliers**

#### WHAT IT SHOWS

For **each feature column**, you see:

* Spread of the data
* Central tendency (median)
* Presence of outliers
* Skewness



#### WHAT IT'S GOOD FOR

* **Detecting outliers**
* **Comparing spread** across features
* **Checking data quality** before modeling
* Identifying **skewed distributions**



#### SUMMARY

* **Type:** Multi-panel boxplot
* **Purpose:** Visual inspection of feature distributions and outliers
* **Good for:** Preprocessing, EDA (Exploratory Data Analysis, EDA = structured first look at your data to decide what to do next.)
