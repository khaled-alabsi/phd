# Dependency types

### 1. **Linear Dependence**  

**Intuition:** A straight-line relationship between variables.  
**Example:**  
$Y = 3X + 5$  
If $X$ increases, $Y$ increases proportionally.  
**Measure:** Pearson correlation.

---

### 2. **Nonlinear Dependence**  

**Intuition:** Relationship exists, but not linear.  
**Example:**  
$Y = X^2$ → as $X$ increases, $Y$ increases faster.  
Linear correlation here might be zero even though dependence exists.

---

### 3. **Monotonic Dependence**  

**Intuition:** One variable always increases or decreases with the other, but rate can vary.  
**Example:**  
$Y = \log(X)$ for $X > 0$  
Still increasing, but slower as $X$ grows.  
**Measure:** Spearman’s or Kendall’s rank correlation.

---

### 4. **Functional Dependence**  

**Intuition:** One variable completely determines the other.  
**Example:**  
$Y = 2X$ → knowing $X$ gives exact $Y$.  
Dependency is deterministic, not probabilistic.

---

### 5. **Stochastic Dependence**  

**Intuition:** Knowing one variable gives information about the probability distribution of another.  
**Example:**  
Height and weight of individuals. Taller people are more likely to weigh more, but not always.

---

### 6. **Conditional Dependence**  

**Intuition:** Variables are dependent given knowledge of a third.  
**Example:**  
Symptoms and diagnosis are dependent *given* disease.  
If you know the disease, symptoms and diagnosis are no longer independent.

---

### 7. **Tail Dependence**  

**Intuition:** Variables show strong dependence in extreme values.  
**Example:**  
Two stock prices may be independent usually, but both crash during a market collapse.  
**Used in:** Copula models, risk management.

---

### 8. **Temporal Dependence**  

**Intuition:** Past values affect future values.  
**Example:**  
Daily temperature – today's value is highly dependent on yesterday’s.  
**Measure:** Autocorrelation, ACF/PACF.

---

### 9. **Spatial Dependence**  

**Intuition:** Observations closer in space tend to be similar.  
**Example:**  
Rainfall measurements at nearby stations.  
**Used in:** Geostatistics, spatial econometrics.

---

### 10. **Joint Dependence**  

**Intuition:** Multiple variables exhibit a dependency structure.  
**Example:**  
Multivariate normal distribution with non-zero off-diagonal covariance terms.  
**Measure:** Mutual information, covariance matrix.

---

Yes, Developer. In addition to the core types, there are more nuanced or domain-specific types of statistical dependency. Here’s an extended list including those:

---

### 11. **Higher-Order Dependence**  

**Intuition:** Dependency appears only when looking at combinations of variables.  
**Example:**  
Three variables $X, Y, Z$ are pairwise independent but jointly dependent.  
**Used in:** Interaction effects in ANOVA, higher-order cumulants.

---

### 12. **Latent Dependence**  

**Intuition:** Apparent independence hides dependence driven by unobserved factors.  
**Example:**  
Two test scores seem independent until we account for a latent "IQ" variable.  
**Used in:** Factor analysis, SEM.

---

### 13. **Instantaneous Dependence**  

**Intuition:** Variables depend on each other at the same time point.  
**Example:**  
GDP and inflation rate in macroeconomic models.  
**Handled in:** VAR models with contemporaneous terms.

---

### 14. **Directional (Causal) Dependence**  

**Intuition:** One variable influences another over time or via a known mechanism.  
**Example:**  
Drug → Recovery.  
**Used in:** Granger causality, DAGs, structural models.

---

### 15. **Nonstationary Dependence**  

**Intuition:** Dependency structure changes over time.  
**Example:**  
Correlation between assets changes pre- and post-crisis.  
**Used in:** Time-varying copulas, regime-switching models.

---

### 16. **Cyclostationary Dependence**  

**Intuition:** Dependence structure repeats periodically.  
**Example:**  
Electricity demand vs. time of day or day of week.  
**Used in:** Signal processing, periodic time series.

---

### 17. **Non-Markovian Dependence**  

**Intuition:** Current state depends on more than just the last one.  
**Example:**  
Long memory processes like ARFIMA models.  
**Opposed to:** Markov processes (only current → next).

---

### 18. **Copula-based Dependence**  

**Intuition:** Separate the marginal distributions and model the dependence structure.  
**Example:**  
Two variables with different distributions but joined by a Gumbel copula.  
**Used in:** Risk, finance, insurance.

---

### 19. **Contextual Dependence**  

**Intuition:** Dependency strength changes with context.  
**Example:**  
Price elasticity of demand may depend on season or customer type.  
**Used in:** Context-aware recommender systems.

---

### 20. **Asymmetric Dependence**  

**Intuition:** Influence is stronger in one direction or in one part of the distribution.  
**Example:**  
Market drops spread faster than gains (leverage effect).  
**Used in:** Tail risk, quantile regression, skewed copulas.

---

Yes, Developer — there *are* even more, especially when you dig into specialized areas. Here's a further extended list capturing additional forms, including theoretical and applied dependencies:

---

### 21. **Pseudo-dependence**  

**Intuition:** Apparent dependence due to data transformation or conditioning.  
**Example:**  
Simpson’s paradox — aggregated data shows dependence that vanishes when split by group.  
**Used in:** Causal inference, confounding analysis.

---

### 22. **Long-range Dependence (LRD)**  

**Intuition:** Correlations decay slowly; distant events remain dependent.  
**Example:**  
Internet traffic, volatility in financial returns.  
**Modelled by:** Fractional Brownian motion, ARFIMA.

---

### 23. **Graphical Dependence**  

**Intuition:** Dependencies represented via edges in a probabilistic graphical model.  
**Example:**  
Bayesian networks: edge from $A \rightarrow B$ means $B$ depends on $A$.  
**Used in:** Probabilistic reasoning, structure learning.

---

### 24. **Marginal Dependence**  

**Intuition:** Dependency exists only when examining marginal distributions.  
**Example:**  
If $P(X, Y) \ne P(X)P(Y)$ → marginal dependence.  
But conditioning on a third variable may remove it.

---

### 25. **Induced Dependence**  

**Intuition:** Dependence introduced by conditioning or sampling.  
**Example:**  
Case-control studies can induce correlation between disease and risk factors.  
**Used in:** Epidemiology, collider bias analysis.

---

### 26. **Mixture-induced Dependence**  

**Intuition:** Combining independent populations introduces dependency.  
**Example:**  
Two independent Gaussians with different means → mixture appears dependent.  
**Used in:** Model-based clustering, hidden Markov models.

---

### 27. **Block Dependence**  

**Intuition:** Dependency is strong within blocks, weak between them.  
**Example:**  
Student scores grouped by class – high within-class correlation, low across.  
**Used in:** Random effects, hierarchical models.

---

### 28. **Nonlinear Granger-type Dependence**  

**Intuition:** Future values of one variable depend nonlinearly on past of another.  
**Example:**  
$Y_t = \sin(X_{t-1}) + \epsilon_t$  
**Tested via:** Kernel Granger causality, neural causality tests.

---

### 29. **Functional Time Series Dependence**  

**Intuition:** Entire curves or functions are dependent over time.  
**Example:**  
Daily electricity load curves.  
**Used in:** Functional data analysis, Hilbert space models.

---

### 30. **Regime-dependent Dependence**  

**Intuition:** Different states of the system have different dependency structures.  
**Example:**  
Low-volatility vs. high-volatility market regimes with different asset correlations.  
**Used in:** Markov switching models.

---
