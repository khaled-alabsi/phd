### Student distribution and t-test

1. **Probability Distributions**  
2. **Student's t-distribution**  
3. **Sampling Distributions**  
4. **Central Limit Theorem**  
5. **Hypothesis Testing**  
6. **One-sample t-test**  
7. **Two-sample t-test**  
8. **Paired t-test**  
9. **Degrees of Freedom**  
10. **Type I and Type II Errors**  
11. **P-values and Significance Level**  
12. **Effect Size**  
13. **Assumptions of t-tests (normality, independence, etc.)**

### 1. Probability Distributions  

A **probability distribution** describes how values of a random variable are distributed. It provides the probabilities of different outcomes in a dataset. There are two main types:

#### **1.1 Discrete Probability Distributions**  
Used for discrete random variables (finite or countable values). Examples:  
- **Bernoulli Distribution**: Binary outcomes (success/failure).  
- **Binomial Distribution**: Number of successes in a fixed number of trials.  
- **Poisson Distribution**: Probability of a number of events occurring in a fixed interval.

#### **1.2 Continuous Probability Distributions**  
Used for continuous random variables (infinite possible values). Examples:  
- **Uniform Distribution**: Equal probability across a range.  
- **Normal Distribution**: Bell-shaped curve, defined by mean $\mu$ and standard deviation $\sigma$.  
- **Student’s t-distribution**: Similar to normal but with heavier tails, used when sample size is small.  

#### **1.3 Properties of Probability Distributions**  
- **Probability Density Function (PDF) for continuous variables**  
  $$
  P(a \leq X \leq b) = \int_a^b f(x) dx
  $$
  where $f(x)$ is the probability density function.  
- **Probability Mass Function (PMF) for discrete variables**  
  $$
  P(X = x) = f(x)
  $$
  which gives the probability of specific values.  
- **Cumulative Distribution Function (CDF)**  
  $$
  F(x) = P(X \leq x)
  $$
  which gives the probability that $X$ is less than or equal to a given value.  
- **Mean (Expectation)**:  
  $$
  E[X] = \sum x P(x) \quad \text{(discrete)}, \quad E[X] = \int x f(x) dx \quad \text{(continuous)}
  $$
- **Variance**:  
  $$
  Var(X) = E[X^2] - (E[X])^2
  $$

Probability distributions are essential because Student’s t-distribution is a special case of a continuous probability distribution used in hypothesis testing.

---  

### 2. Student’s t-Distribution  

The **Student’s t-distribution** is a continuous probability distribution used when estimating the mean of a normally distributed population but the sample size is small, and the population standard deviation is unknown.

#### **2.1 Definition and Formula**  
The t-distribution is defined by the **degrees of freedom (df)**, which determine its shape. If a random variable $X$ follows a t-distribution with $v$ degrees of freedom, it is written as:  
$$
X \sim t_v
$$  
The t-distribution is derived from the standard normal distribution and follows the formula:  
$$
t = \frac{\bar{X} - \mu}{\frac{s}{\sqrt{n}}}
$$  
where:  
- $\bar{X}$ = sample mean  
- $\mu$ = population mean (hypothesized)  
- $s$ = sample standard deviation  
- $n$ = sample size  

#### **2.2 Properties of the t-Distribution**  
- **Bell-shaped and symmetric**, like the normal distribution.  
- **Heavier tails** than the normal distribution (more probability in the tails), meaning it accounts for more variability in small samples.  
- As **degrees of freedom increase**, the t-distribution approaches the standard normal distribution $N(0,1)$.  
- The mean of the t-distribution is **zero**, and its variance is **greater than 1** (depends on $v$ ):  
  $$
  Var(t) = \frac{v}{v - 2}, \quad v > 2
  $$  

#### **2.3 Why the t-Distribution is Important**  
- Used in **hypothesis testing** when the population standard deviation is unknown.  
- Forms the basis of the **t-test**, which compares means between one or two groups.  
- Handles **small sample sizes** better than the normal distribution.  

---

### 3. Sampling Distributions  

A **sampling distribution** is the probability distribution of a statistic (e.g., mean, variance) based on a random sample from a population. It describes how the statistic behaves across different samples.

#### **3.1 Definition and Key Concept**  
A sample statistic (e.g., sample mean $\bar{X}$ ) varies from sample to sample. The distribution of these sample statistics is called the **sampling distribution**. The most important cases are:

- **Sampling Distribution of the Sample Mean**  
  If $X_1, X_2, ..., X_n$ are independent and identically distributed (i.i.d.) from a population with mean $\mu$ and variance $\sigma^2$ , the sample mean is:  
  $$
  \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
  $$
  The distribution of $\bar{X}$ depends on the sample size and the population distribution.

- **Standard Error (SE) of the Mean**  
  Measures how much the sample mean is expected to vary:  
  $$
  SE = \frac{\sigma}{\sqrt{n}}
  $$
  where $\sigma$ is the population standard deviation.

#### **3.2 Central Role in t-Tests**  
The sampling distribution is critical for the t-test because:  
- If the population is **normal**, then the sample mean follows a normal or t-distribution depending on whether $\sigma$ is known.  
- If the population is **not normal**, the Central Limit Theorem (next topic) ensures the sampling distribution approximates normality for large $n$.  

#### **3.3 Types of Sampling Distributions**  
1. **Normal (Gaussian) Sampling Distribution**: When the population is normal or $n$ is large.  
2. **t-Distribution**: When the population variance is unknown, and the sample size is small.  
3. **Chi-Square Distribution**: Distribution of sample variances.  
4. **F-Distribution**: Ratio of two sample variances, used in ANOVA.  

---

### **Mathematical Derivation of Standard Error**  

The **Standard Error (SE)** of the mean is given by:  
$$
SE = \frac{\sigma}{\sqrt{n}}
$$  
where $\sigma$ is the population standard deviation, and $n$ is the sample size.  

To derive this, we use **properties of variance**.

#### **Step 1: Variance of a Sum of Independent Variables**  
Consider a random sample $X_1, X_2, ..., X_n$ drawn from a population with mean $\mu$ and variance $\sigma^2$.  

The sample mean is:  
$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

Using the **variance properties**:
1. If $X_i$ are independent, then:
   $$
   Var\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} Var(X_i) = n\sigma^2
   $$

2. Since we divide by $n$ , applying the variance scaling rule $Var(aX) = a^2 Var(X)$ :
   $$
   Var(\bar{X}) = Var\left(\frac{1}{n} \sum_{i=1}^{n} X_i\right) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}
   $$

3. The **standard deviation** (which is the standard error) is the square root of variance:
   $$
   SE = \sqrt{Var(\bar{X})} = \sqrt{\frac{\sigma^2}{n}} = \frac{\sigma}{\sqrt{n}}
   $$

---

### **Intuition Behind $\frac{\sigma}{\sqrt{n}}$**  

1. **Larger Samples Reduce Variability**  
   - If you take multiple samples, small samples fluctuate more.
   - As $n$ increases, the variability in the sample mean decreases.

2. **Averaging Reduces Spread**  
   - When you take an average of multiple independent variables, extreme values cancel out.
   - The spread of the mean is less than the spread of individual values.

3. **Law of Large Numbers**  
   - As $n \to \infty$ , $\bar{X}$ converges to $\mu$.
   - The formula $\frac{\sigma}{\sqrt{n}}$ shows how quickly the variability shrinks.


This result is key for **confidence intervals** and **t-tests** because it quantifies uncertainty in the estimate of $\mu$.

### 4. Central Limit Theorem (CLT)  

The **Central Limit Theorem (CLT)** states that, regardless of the population distribution, the **sampling distribution of the sample mean** approaches a normal distribution as the sample size increases.  

#### **4.1 Formal Statement**  
Let $X_1, X_2, ..., X_n$ be a random sample from any population with mean $\mu$ and variance $\sigma^2$. Define the sample mean:  
$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$  
Then, as $n \to \infty$ , the distribution of $\bar{X}$ follows a normal distribution:  
$$
\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n} \right)
$$  
Mathematically, for large $n$ :  
$$
\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \xrightarrow{d} N(0,1)
$$  
where $\xrightarrow{d}$ denotes convergence in distribution.

---

#### **4.2 Why CLT is Important**
1. **Allows Normal Approximation for Any Distribution**  
   - Even if the population distribution is skewed or non-normal, the sample mean follows a normal distribution for large $n$.  
   - This is why t-tests (which assume normality) can be used even when the original data isn't normal, as long as the sample size is large.  

2. **Justifies Standard Error Formula**  
   - CLT confirms that the variability of $\bar{X}$ is $\frac{\sigma}{\sqrt{n}}$ , meaning larger samples give more precise estimates.  

3. **Foundation of Many Statistical Tests**  
   - The normal approximation enables hypothesis testing and confidence intervals.  

---

#### **4.3 How Large Should $n$ Be?**
- **If the population is normal**, CLT applies for any $n$.  
- **If the population is skewed or non-normal**, $n \geq 30$ is usually enough for normal approximation.  
- **If the population is heavily skewed**, $n \geq 50$ or more might be needed.  

---

### 5. Hypothesis Testing  

**Hypothesis testing** is a statistical method used to make inferences about a population based on sample data. It determines whether there is enough evidence to reject a claim about a population parameter.

---

#### **5.1 Steps of Hypothesis Testing**  

1. **Define Null and Alternative Hypotheses**  
   - **Null Hypothesis ($H_0$ )**: Assumes no effect or no difference.  
   - **Alternative Hypothesis ($H_a$ )**: Represents what we want to test (effect or difference exists).  
   - Example:  
     - $H_0$ : The population mean is $\mu_0$ ($\mu = 100$ ).  
     - $H_a$ : The population mean is different from $\mu_0$ ($\mu \neq 100$ ).  

2. **Choose Significance Level ($\alpha$ )**  
   - Common choices: **0.05, 0.01, or 0.10**.  
   - It represents the probability of rejecting $H_0$ when it is actually true (Type I error).  

3. **Select and Compute the Test Statistic**  
   - A **test statistic** measures how far the sample result is from $H_0$.  
   - If the population standard deviation is **unknown** and $n$ is small, use the **t-statistic**:  
     $$
     t = \frac{\bar{X} - \mu_0}{\frac{s}{\sqrt{n}}}
     $$  
   - If the population standard deviation is **known**, use the **z-statistic**.

4. **Find the p-value or Critical Value**  
   - The **p-value** is the probability of obtaining the observed result (or more extreme) if $H_0$ is true.  
   - Compare the **test statistic** with the critical value from the **t-distribution** (or z-distribution for large samples).  

5. **Make a Decision**  
   - If $p \leq \alpha$ , reject $H_0$ (evidence supports $H_a$ ).  
   - If $p > \alpha$ , fail to reject $H_0$ (not enough evidence to support $H_a$ ).  

---

#### **5.2 Types of Hypothesis Tests**  
1. **One-tailed test**: Tests for an increase or decrease (e.g., $H_a: \mu > \mu_0$ or $H_a: \mu < \mu_0$ ).  
2. **Two-tailed test**: Tests for any difference (e.g., $H_a: \mu \neq \mu_0$ ).  

---

#### **5.3 Importance in t-Tests**  
- Hypothesis testing forms the basis of **one-sample t-tests, two-sample t-tests, and paired t-tests**.  
- The t-test uses the sample mean and variance to test if a population mean differs from a known value or another group.  

---

### 6. One-Sample t-Test  

A **one-sample t-test** is used to determine whether the mean of a single sample differs significantly from a known or hypothesized population mean. It is applied when the population standard deviation is **unknown** and the sample size is **small** ($n < 30$ ).

---

#### **6.1 Hypotheses**  
- **Null Hypothesis ($H_0$ )**: The sample mean equals the population mean.  
  $$
  H_0: \mu = \mu_0
  $$  
- **Alternative Hypothesis ($H_a$ )** (depends on the test type):  
  - **Two-tailed**: $H_a: \mu \neq \mu_0$ (mean is different).  
  - **Left-tailed**: $H_a: \mu < \mu_0$ (mean is lower).  
  - **Right-tailed**: $H_a: \mu > \mu_0$ (mean is higher).  

---

#### **6.2 Test Statistic**  
Since the population standard deviation $\sigma$ is unknown, we estimate it using the **sample standard deviation** $s$ and use the **t-statistic**:  

$$
t = \frac{\bar{X} - \mu_0}{\frac{s}{\sqrt{n}}}
$$

where:  
- $\bar{X}$ = sample mean  
- $\mu_0$ = hypothesized population mean  
- $s$ = sample standard deviation  
- $n$ = sample size  

This follows a **t-distribution** with $df = n - 1$.

---

#### **6.3 Decision Rule**  
1. **Find the critical value** $t_{\alpha, df}$ from the t-table based on $\alpha$ and $df$.  
2. Compare the **absolute value** of the test statistic $|t|$ with the critical value:  
   - If $|t| > t_{\alpha, df}$ , **reject $H_0$** (significant difference).  
   - If $|t| \leq t_{\alpha, df}$ , **fail to reject $H_0$** (no significant difference).  
3. Alternatively, use the **p-value approach**:  
   - If $p \leq \alpha$ , reject $H_0$.  
   - If $p > \alpha$ , fail to reject $H_0$.  

---

#### **6.4 Example Calculation**
Suppose we have a sample of **10 students’ test scores**, with:  
- Sample mean $\bar{X} = 78$  
- Sample standard deviation $s = 12$  
- Hypothesized population mean $\mu_0 = 85$  
- Significance level $\alpha = 0.05$  

**Step 1: Compute $t$ -Statistic**  
$$
t = \frac{78 - 85}{\frac{12}{\sqrt{10}}} = \frac{-7}{\frac{12}{3.162}} = \frac{-7}{3.796} = -1.84
$$

**Step 2: Find Critical Value**  
For $df = 10 - 1 = 9$ and $\alpha = 0.05$ (two-tailed), the **critical value** from the t-table is **$t_{0.025,9} = 2.262$**.  

**Step 3: Compare $|t|$ with Critical Value**  
$$
| -1.84 | = 1.84 < 2.262
$$
Since $|t|$ is **less than** the critical value, we **fail to reject $H_0$**.  

**Conclusion:** The sample mean is **not significantly different** from the hypothesized population mean at $\alpha = 0.05$.

---

### 7. Two-Sample t-Test  

A **two-sample t-test** compares the means of **two independent groups** to determine if they are significantly different. It is used when the population standard deviations are **unknown**, and the sample sizes may be small.

---

#### **7.1 Hypotheses**  
- **Null Hypothesis ($H_0$ )**: The two population means are equal.  
  $$
  H_0: \mu_1 = \mu_2
  $$
- **Alternative Hypothesis ($H_a$ )** (depends on the test type):  
  - **Two-tailed**: $H_a: \mu_1 \neq \mu_2$ (means are different).  
  - **Left-tailed**: $H_a: \mu_1 < \mu_2$ (first group has a lower mean).  
  - **Right-tailed**: $H_a: \mu_1 > \mu_2$ (first group has a higher mean).  

---

#### **7.2 Test Statistic**  

The test statistic depends on whether the two groups have **equal variances** or **unequal variances**.

1. **Equal Variance (Pooled t-Test, Assumption: $\sigma_1^2 = \sigma_2^2$ )**  
   When the population variances are assumed equal, we use the **pooled standard deviation**:  

   $$
   s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}
   $$

   The **t-statistic** is:  

   $$
   t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
   $$

   Degrees of freedom:  
   $$
   df = n_1 + n_2 - 2
   $$

2. **Unequal Variance (Welch’s t-Test, No Assumption on $\sigma_1^2$ and $\sigma_2^2$ )**  
   When the two groups have different variances, we use:

   $$
   t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
   $$

   The degrees of freedom are approximated using **Welch-Satterthwaite formula**:

   $$
   df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2}{\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1 - 1} + \frac{\left(\frac{s_2^2}{n_2}\right)^2}{n_2 - 1}}
   $$

---

#### **7.3 Decision Rule**
1. Find the **critical value** $t_{\alpha, df}$ from the t-table.  
2. Compare the test statistic $|t|$ with the critical value:
   - If $|t| > t_{\alpha, df}$ , **reject $H_0$** (significant difference).  
   - If $|t| \leq t_{\alpha, df}$ , **fail to reject $H_0$** (no significant difference).  
3. Alternatively, use the **p-value approach**:
   - If $p \leq \alpha$ , reject $H_0$.
   - If $p > \alpha$ , fail to reject $H_0$.

---

#### **7.4 Example Calculation**
Suppose we have two independent groups with:  
- Group 1: $n_1 = 12$ , $\bar{X}_1 = 80$ , $s_1 = 10$  
- Group 2: $n_2 = 10$ , $\bar{X}_2 = 90$ , $s_2 = 12$  
- Significance level $\alpha = 0.05$  

Assuming **unequal variances**, we compute:  

$$
t = \frac{80 - 90}{\sqrt{\frac{10^2}{12} + \frac{12^2}{10}}}
$$

$$
t = \frac{-10}{\sqrt{\frac{100}{12} + \frac{144}{10}}} = \frac{-10}{\sqrt{8.33 + 14.4}} = \frac{-10}{\sqrt{22.73}} = \frac{-10}{4.77} = -2.10
$$

Using the **Welch-Satterthwaite formula**, we approximate $df$ and find the **critical value** from the t-table for $\alpha = 0.05$.  
- If $|t|$ exceeds the critical value, we reject $H_0$.  
- If $|t|$ is less than the critical value, we fail to reject $H_0$.  

---

### 8. Paired t-Test  

A **paired t-test** is used to compare the means of **two related (dependent) samples**. This test is applied when measurements are taken from the **same subjects** before and after an intervention, or when two related conditions are tested (e.g., performance under two different treatments).  

---

#### **8.1 Hypotheses**  
- **Null Hypothesis ($H_0$ )**: The mean difference between the paired samples is zero.  
  $$
  H_0: \mu_d = 0
  $$
- **Alternative Hypothesis ($H_a$ )**: The mean difference is **not** zero.
  - **Two-tailed**: $H_a: \mu_d \neq 0$ (any difference).  
  - **Left-tailed**: $H_a: \mu_d < 0$ (first condition is lower).  
  - **Right-tailed**: $H_a: \mu_d > 0$ (first condition is higher).  

Here, $\mu_d$ represents the **mean of the differences** between paired observations.

---

#### **8.2 Test Statistic**  

Since the data are dependent, we compute the **differences** $d_i$ between the paired observations first:

$$
d_i = X_{i1} - X_{i2}
$$

where:  
- $X_{i1}$ = first measurement (before treatment)  
- $X_{i2}$ = second measurement (after treatment)  

Then, we compute the **sample mean of the differences** $\bar{d}$ and the **sample standard deviation of the differences** $s_d$ :

$$
\bar{d} = \frac{\sum d_i}{n}
$$

$$
s_d = \sqrt{\frac{\sum (d_i - \bar{d})^2}{n-1}}
$$

The **t-statistic** is:

$$
t = \frac{\bar{d}}{\frac{s_d}{\sqrt{n}}}
$$

where:  
- $\bar{d}$ = mean of the differences  
- $s_d$ = standard deviation of the differences  
- $n$ = number of pairs  

This follows a **t-distribution** with $df = n - 1$.

---

#### **8.3 Decision Rule**  
1. Find the **critical value** $t_{\alpha, df}$ from the t-table.  
2. Compare $|t|$ with the critical value:
   - If $|t| > t_{\alpha, df}$ , **reject $H_0$** (significant difference).  
   - If $|t| \leq t_{\alpha, df}$ , **fail to reject $H_0$** (no significant difference).  
3. Alternatively, use the **p-value approach**:
   - If $p \leq \alpha$ , reject $H_0$.
   - If $p > \alpha$ , fail to reject $H_0$.

---

#### **8.4 Example Calculation**  
Suppose we have a study measuring students' test scores before and after a training program. The data:  

| Student | Before ($X_1$ ) | After ($X_2$ ) | Difference ($d$ ) |
|---------|---------|--------|-------------|
| 1       | 78      | 85     | -7          |
| 2       | 82      | 88     | -6          |
| 3       | 75      | 80     | -5          |
| 4       | 90      | 92     | -2          |
| 5       | 85      | 91     | -6          |

**Step 1: Compute the Mean of Differences**  
$$
\bar{d} = \frac{(-7) + (-6) + (-5) + (-2) + (-6)}{5} = \frac{-26}{5} = -5.2
$$

**Step 2: Compute the Standard Deviation of Differences**  
$$
s_d = \sqrt{\frac{((-7 + 5.2)^2 + (-6 + 5.2)^2 + (-5 + 5.2)^2 + (-2 + 5.2)^2 + (-6 + 5.2)^2)}{5-1}}
$$

$$
s_d = \sqrt{\frac{(3.24 + 0.64 + 0.04 + 10.24 + 0.64)}{4}}
$$

$$
s_d = \sqrt{\frac{14.8}{4}} = \sqrt{3.7} = 1.92
$$

**Step 3: Compute the t-Statistic**  
$$
t = \frac{-5.2}{\frac{1.92}{\sqrt{5}}} = \frac{-5.2}{\frac{1.92}{2.236}} = \frac{-5.2}{0.86} = -6.05
$$

**Step 4: Compare with Critical Value**  
For $df = 5 - 1 = 4$ and $\alpha = 0.05$ (two-tailed), the **critical value** from the t-table is **$t_{0.025,4} = 2.776$**.  

Since $| -6.05 | = 6.05 > 2.776$ , we **reject $H_0$**, meaning the training program had a significant effect on test scores.

---

 ### 9. Assumptions of t-Tests  

t-tests rely on several assumptions that must be **checked** to ensure the validity of results. If these assumptions are violated, the conclusions drawn from the test may be misleading.

---

### **9.1 Assumptions for All t-Tests**  
These apply to both one-sample, two-sample, and paired t-tests.

#### **1. Independence of Observations**  
- For **one-sample** and **paired t-tests**: Each data point must be **independent** of the others.  
- For **two-sample t-tests**: The two groups must be **independent** of each other.  

 **Why is this important?**  
If observations are not independent (e.g., repeated measures without considering correlations), standard errors will be incorrect, leading to **misleading p-values**.

#### **2. Normality of the Population or Sample Size Consideration**  
- The data should come from a **normally distributed population**, especially for small samples ($n < 30$ ).  
- If the sample size is **large** ($n \geq 30$ ), the **Central Limit Theorem (CLT)** ensures approximate normality, even if the original population is not normal.  

 **Why is this important?**  
The t-test is based on the **t-distribution**, which is derived assuming normality. If the data are not normal and the sample is small, the test may not be valid.

**How to check normality?**  
- **Histogram**: Should look roughly symmetric and bell-shaped.  
- **Q-Q Plot**: Should align with the 45-degree line.  
- **Shapiro-Wilk Test**: A formal statistical test for normality.  
  - $H_0$ : Data are normally distributed.  
  - $p < 0.05$ → Data are not normal (consider transformations or non-parametric tests).  

---

### **9.2 Assumptions for Two-Sample t-Tests**  

#### **3. Homogeneity of Variances (Equal Variance Assumption for Pooled t-Test)**  
- This assumption applies **only** to the **pooled (equal variance) t-test**, not Welch’s test.  
- The population variances should be approximately equal:  
  $$
  \sigma_1^2 \approx \sigma_2^2
  $$

 **Why is this important?**  
If the variances are unequal, using the **pooled variance formula** is incorrect and inflates Type I error.

**How to check equal variances?**  
- **Levene’s Test** or **F-test for Equality of Variances**:  
  - $H_0$ : Variances are equal.  
  - $p < 0.05$ → Variances are significantly different → Use Welch’s t-test.  
- **Rule of Thumb**: If **larger sample variance / smaller sample variance < 2**, the variance difference is **not** a major concern.  

---

### **9.3 What If Assumptions Are Violated?**  

| Assumption | What to do if violated? |
|------------|------------------------|
| **Independence** | Use models that account for dependence (e.g., mixed-effects models). |
| **Normality** | Use non-parametric alternatives like **Wilcoxon signed-rank test** (paired) or **Mann-Whitney U test** (two-sample). |
| **Equal Variance** | Use **Welch’s t-test**, which does not assume equal variances. |

---

### 10. Alternatives to t-Tests  

When the assumptions of t-tests are violated—especially **normality** and **homogeneity of variance**—non-parametric or robust methods can be used instead.

---

### **10.1 Non-Parametric Alternatives**  

These tests do **not assume normality** and are suitable for skewed distributions or small sample sizes.

#### **1. One-Sample t-Test Alternative: Wilcoxon Signed-Rank Test**  
- Used when **data are not normally distributed** but are still **symmetrical**.  
- Instead of comparing the mean, it compares the **median**.  
- Test statistic: **Ranks of absolute differences from the median** are summed and compared.  

**When to use?**  
- Data are **skewed** but still **paired or one-sample**.  
- Small sample size ($n < 30$ ).  

#### **2. Paired t-Test Alternative: Wilcoxon Signed-Rank Test**  
- Works the same way as the one-sample case but compares **differences between paired samples**.  

**When to use?**  
- Differences are **not normally distributed**.  

#### **3. Two-Sample t-Test Alternative: Mann-Whitney U Test (Wilcoxon Rank-Sum Test)**  
- Compares **medians** instead of means.  
- Does not assume normality but assumes similar **shapes** of distributions.  
- Test statistic: **Ranks** of all values are compared.  

**When to use?**  
- Two groups have **non-normal distributions**.  
- Sample sizes are **small**.  

---

### **10.2 Robust Alternatives**  

If normality is violated but you still prefer **mean-based** comparisons:

#### **1. Bootstrapping**  
- Resampling method that repeatedly draws samples from the data to estimate **confidence intervals** for the mean difference.  
- No assumption of normality needed.  

**When to use?**  
- Any situation where normality is questionable.  

#### **2. Welch’s t-Test (Alternative for Unequal Variance)**  
- Modified two-sample t-test that does **not assume equal variances**.  
- Uses **separate variance estimates** for each group.  

**When to use?**  
- Variance is **significantly different** between groups.  

---

### **10.3 Summary: When to Use Which Test?**  

| Situation | Use This Test |
|-----------|--------------|
| Normal data, equal variance | **Standard t-test** |
| Normal data, unequal variance | **Welch’s t-test** |
| Non-normal data, paired samples | **Wilcoxon Signed-Rank Test** |
| Non-normal data, independent samples | **Mann-Whitney U Test** |
| Very small sample size | **Wilcoxon tests or Bootstrapping** |

---

### 11. Effect Size in t-Tests  

Effect size measures the **practical significance** of a difference, independent of sample size. While p-values tell us if a difference is statistically significant, **effect size tells us how large that difference is**.

---

### **11.1 Cohen’s d (Most Common Effect Size for t-Tests)**  
Cohen’s $d$ measures the difference between two means in terms of **standard deviations**:

$$
d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}
$$

where:  
- $\bar{X}_1, \bar{X}_2$ = sample means  
- $s_p$ = **pooled standard deviation**, calculated as:

$$
s_p = \sqrt{\frac{(n_1 - 1) s_1^2 + (n_2 - 1) s_2^2}{n_1 + n_2 - 2}}
$$

for two-sample t-tests, or just $s$ (the sample standard deviation) for a one-sample or paired t-test.

**Interpreting Cohen’s d**:  
| $d$ Value | Interpretation |
|-------------|---------------|
| $d < 0.2$ | Negligible effect |
| $0.2 \leq d < 0.5$ | Small effect |
| $0.5 \leq d < 0.8$ | Medium effect |
| $d \geq 0.8$ | Large effect |

---

### **11.2 Glass’s  (Alternative for Unequal Variance)**  
If the two groups have **very different standard deviations**, use Glass’s $\Delta$ :

$$
\Delta = \frac{\bar{X}_1 - \bar{X}_2}{s_1}
$$

where $s_1$ is the standard deviation of the **control group** (not the pooled standard deviation).

**When to use?**  
- When groups have **very different variances**.

---

### **11.3 Hedge’s g (Bias-Corrected Cohen’s d for Small Samples)**  
Cohen’s $d$ slightly **overestimates** effect size when sample sizes are small. Hedge’s $g$ applies a correction:

$$
g = d \times \left(1 - \frac{3}{4(n_1 + n_2) - 9} \right)
$$

**When to use?**  
- When **sample size is small** ($n < 20$ ).  

---

### **11.4 Effect Size for Paired t-Tests**  
For a **paired t-test**, we calculate Cohen’s $d$ using the **mean of the differences**:

$$
d = \frac{\bar{d}}{s_d}
$$

where $s_d$ is the standard deviation of the differences.

---

### **11.5 Why Effect Size Matters**  
- A **statistically significant** result ($p < 0.05$ ) **does not mean** the effect is large or important.  
- A **large effect size** means the difference is **practically meaningful**, even if the sample is small.  

**Example**: A **new teaching method** improves test scores by 2 points with a **large sample** → **statistically significant but small effect size**.  

---

### 12. Confidence Intervals for t-Tests  

A **confidence interval (CI)** provides a range of values that likely contains the true population mean difference. Unlike a p-value, which only tells us whether an effect exists, a confidence interval **quantifies the uncertainty** of our estimate.

---

### **12.1 Confidence Interval Formula for One-Sample and Paired t-Tests**  

For a **one-sample t-test** (or paired t-test, which is mathematically the same), the CI for the mean is:

$$
\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}
$$

where:  
- $\bar{X}$ = sample mean (or mean of differences for paired t-tests)  
- $s$ = sample standard deviation  
- $n$ = sample size  
- $t_{\alpha/2, n-1}$ = critical value from the **t-distribution** with $n-1$ degrees of freedom  
- $\frac{s}{\sqrt{n}}$ = standard error  

**Interpretation**:  
A **95% confidence interval** means:  
- If we repeated the study many times, 95% of those confidence intervals would contain the **true population mean**.  
- If the CI does **not** contain **zero**, the result is statistically significant ($p < 0.05$ ).

---

### **12.2 Confidence Interval for Two-Sample t-Test**  

For an **independent two-sample t-test**, the CI for the difference between two means is:

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, df} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

where:  
- $\bar{X}_1, \bar{X}_2$ = sample means  
- $s_1, s_2$ = sample standard deviations  
- $n_1, n_2$ = sample sizes  
- $df$ = degrees of freedom (depends on whether we assume equal variances)  

**Interpretation**:  
- If the **95% CI** for $\bar{X}_1 - \bar{X}_2$ **does not include 0**, the groups are **significantly different**.  
- A **wide CI** indicates **high uncertainty** due to small sample size or high variability.

---

### **12.3 Confidence Interval for Welch’s t-Test**  

If variances are unequal, we use **Welch’s t-test**, which has a modified CI:

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, df^*} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

where $df^*$ is calculated using **Welch-Satterthwaite equation**:

$$
df^* = \frac{\left( \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}
$$

**Why use this?**  
- Welch’s t-test adjusts for **unequal variances**, leading to more **accurate confidence intervals**.

---

### **12.4 How CI Relates to Hypothesis Testing**  

- If the **95% CI** **excludes zero**, the result is **significant at $p < 0.05$**.  
- If the **CI is very wide**, the sample size might be **too small** or the data too **variable**.

**Example Interpretation**:  
| CI | Conclusion |
|----|------------|
| $(2.1, 5.3)$ | The difference is significant, as 0 is not included. |
| $(-1.2, 3.4)$ | Not significant; the interval contains 0. |
| $(0.5, 1.2)$ | Significant and precise (narrow range). |

---

### 13. Power and Sample Size in t-Tests  

**Power** is the probability of correctly rejecting the null hypothesis when it is false (i.e., detecting a true effect). It depends on four factors:

1. **Effect size ($d$ )** – Larger effects are easier to detect.  
2. **Sample size ($n$ )** – More data reduces variability.  
3. **Significance level ($\alpha$ )** – Lower $\alpha$ (e.g., 0.01) requires stronger evidence.  
4. **Variance ($s^2$ )** – High variance makes effects harder to detect.  

**Target Power**:  
- Typically, studies aim for **80% power** ($1 - \beta = 0.8$ ), meaning an **80% chance of detecting a true effect**.

---

### **13.1 Power Calculation for One-Sample and Paired t-Test**  

For a **one-sample** or **paired t-test**, power is:

$$
1 - \beta = P\left( t > t_{\alpha, n-1} \mid d, n \right)
$$

where $d = \frac{\bar{X} - \mu}{s}$ is **Cohen’s d**, and $t_{\alpha, n-1}$ is the critical t-value.

**Interpretation**:  
- If power = 0.80, there’s a **20% chance ($\beta = 0.2$ ) of a Type II error** (failing to reject $H_0$ when $H_A$ is true).

---

### **13.2 Power Calculation for Two-Sample t-Test**  

For an **independent two-sample t-test**, power is:

$$
1 - \beta = P\left( t > t_{\alpha, df} \mid d, n_1, n_2 \right)
$$

where:

$$
d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}
$$

and $df$ is the degrees of freedom.

**Key Findings**:  
- **Larger $d$** → higher power.  
- **Larger $n$** → higher power.  
- **Higher variance** ($s^2$ ) → lower power.  

---

### **13.3 Sample Size Calculation**  

To determine the **minimum sample size needed for 80% power**, we use:

$$
n = \frac{(t_{\alpha/2} + t_{\beta})^2 \cdot 2s^2}{(\bar{X}_1 - \bar{X}_2)^2}
$$

where:  
- $t_{\beta}$ = critical value for power (from normal distribution).  
- $s^2$ = estimated variance.  
- $\bar{X}_1 - \bar{X}_2$ = expected difference.

**Example**: If we expect $d = 0.5$ and want 80% power at $\alpha = 0.05$ , we need **64 participants per group**.

---

### **13.4 Why Power Matters**  

- **Low power** ($< 80\%$ ) → high risk of **false negatives** (missing a real effect).  
- **Overpowered studies** → unnecessary costs/resources.

**Rule of Thumb**:  
- If $d$ is **small (0.2)** → need **large $n$ (~200 per group)**.  
- If $d$ is **large (0.8)** → smaller $n$ (~25 per group) is sufficient.  

---

### 14. Assumptions of t-Tests and How to Check Them  

t-Tests rely on **several key assumptions**. Violating these can lead to incorrect conclusions.

---

### **14.1 Assumptions for One-Sample and Paired t-Tests**  

1. **Independence of observations**  
   - Each sample must come from independent subjects.  
   - In paired t-tests, differences between pairs must be independent.  
   - **Check:** Ensure study design prevents dependencies (e.g., no repeated measurements on the same subject).  

2. **Normality of data**  
   - The **differences** (not the raw data) should be normally distributed for paired t-tests.  
   - **Check:** Use **Shapiro-Wilk test** or **QQ plot**.  
   - **Fix if violated:** Use **Wilcoxon signed-rank test** instead of t-test if normality is violated.  

---

### **14.2 Assumptions for Independent Two-Sample t-Test**  

1. **Independence of groups**  
   - The two groups should be completely separate (no subject in both groups).  
   - **Check:** Ensure proper randomization and independent sampling.  

2. **Normality within each group**  
   - Each group’s data should follow a normal distribution **if the sample size is small**.  
   - **Check:** Use **Shapiro-Wilk test** for normality.  
   - **Fix if violated:** If $n > 30$ , normality matters less (Central Limit Theorem). If $n < 30$ , use **Mann-Whitney U test** instead.  

3. **Equal variance assumption (Homogeneity of Variance)**  
   - The variances of the two groups should be roughly equal.  
   - **Check:** Use **Levene’s test** or **F-test**.  
   - **Fix if violated:** Use **Welch’s t-test** instead of standard t-test.  

---

### **14.3 Normality Testing Methods**  

1. **Shapiro-Wilk Test**  
   - $H_0$ : Data is normal  
   - $p < 0.05$ → reject normality assumption.  

2. **QQ Plot**  
   - Data should follow a straight diagonal line.  

3. **Histogram & Skewness**  
   - Extreme skewness indicates a potential problem.  

**Rule of Thumb**: If $n > 30$ , normality is less of a concern due to the **Central Limit Theorem**.  

---

### **14.4 Variance Equality Testing Methods**  

1. **Levene’s Test**  
   - $H_0$ : Variances are equal  
   - $p < 0.05$ → variances are unequal → use **Welch’s t-test**.  

2. **Boxplot Comparison**  
   - Visually compare spread (longer boxes suggest higher variance).  

**When to use Welch’s t-test?**  
- If **Levene’s test fails** ($p < 0.05$ ).  
- If sample sizes are **very different**.  

---

### **14.5 What to Do if Assumptions Are Violated?**  

| Violation | Solution |
|------------|------------|
| Normality in small samples | Use **Wilcoxon signed-rank test** (paired) or **Mann-Whitney U test** (independent). |
| Unequal variances | Use **Welch’s t-test** instead of standard t-test. |
| Non-independent observations | Consider **repeated measures ANOVA** or **mixed-effects models**. |

---

