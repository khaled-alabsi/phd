# Types of statistical dependency grouped by dependency family

## **Temporal Dependencies**

| Type | Testable | Method / Reason | Nonlinear / High-Dimensional |
|------|----------|-----------------|-----------------------------|
| **Temporal Dependence** | Yes | ACF, PACF, Ljung–Box test | Yes (if autoregressive models are nonlinear) |
| **Instantaneous Dependence** | Yes | Structural VAR (SVAR) with contemporaneous terms | No (Linear in structure) |
| **Long-range Dependence (LRD)** | Yes | Hurst exponent, rescaled range (R/S), GPH test | Yes (Nonlinear scaling in long-memory processes) |
| **Cyclostationary Dependence** | Yes | Periodogram, frequency-domain tests | Yes (Nonlinear periodic patterns) |
| **Nonstationary Dependence** | Yes | KPSS, ADF with time-varying models | Yes (Nonlinear time-varying dependencies) |
| **Non-Markovian Dependence** | Yes | AIC/BIC comparison of memory depth, Ljung–Box with high lags | Yes (Nonlinear dependencies in state transitions) |
| **Functional Time Series Dependence** | Yes | Functional autocorrelation, FPCA-based tests | Yes (High-dimensional functional data analysis) |
| **Regime-dependent Dependence** | Yes | Hidden Markov Models, Markov switching likelihood ratio test | Yes (Nonlinear regime-switching models) |

---

## **Spatial Dependencies**

| Type | Testable | Method / Reason | Nonlinear / High-Dimensional |
|------|----------|-----------------|-----------------------------|
| **Spatial Dependence** | Yes | Moran's I, Geary's C, Lagrange Multiplier tests | Yes (Nonlinear spatial models like kernel-based) |
| **Block Dependence** | Yes | Block bootstrap tests, nested random effects models | Yes (High-dimensional random effects models) |

---

## **Causal & Directional Dependencies**

| Type | Testable | Method / Reason | Nonlinear / High-Dimensional |
|------|----------|-----------------|-----------------------------|
| **Directional (Causal) Dependence** | Yes | Granger causality, structural equation modeling (SEM), DAG structure learning | Yes (Nonlinear causal structures like neural networks) |
| **Granger-type Dependence** | Yes | Granger causality test (linear) | Yes (Kernel-based or nonlinear Granger causality) |
| **Nonlinear Granger-type Dependence** | Yes | Kernel Granger test, neural network-based tests | Yes (Nonlinear models with kernel or NN) |
| **Latent Dependence** | Partially | Latent variable models (e.g., factor analysis) | Yes (High-dimensional factor analysis, PCA) |
| **Induced Dependence** | Partially | Requires knowledge of sampling mechanism | Yes (Complex nonlinear interactions) |
| **Pseudo-dependence** | Partially | Requires domain knowledge; not formally testable without counterfactuals | Yes (Could involve high-dimensional causal models) |
| **Mixture-induced Dependence** | Yes | EM algorithm diagnostics, likelihood ratio tests for mixture models | Yes (High-dimensional mixture models) |
| **Contextual Dependence** | Yes | Interaction terms in regression, stratified analysis, causal forests | Yes (High-dimensional interactions in forests) |

---

## **Functional/Mathematical Dependencies**

| Type | Testable | Method / Reason | Nonlinear / High-Dimensional |
|------|----------|-----------------|-----------------------------|
| **Functional Dependence** | Yes | Zero conditional variance test, scatterplots with no residual variance | Yes (Nonlinear functional forms) |
| **Linear Dependence** | Yes | Pearson correlation, linear regression | No (Linear by definition) |
| **Nonlinear Dependence** | Yes | Distance correlation, HSIC, mutual information | Yes (Nonlinear dependency detection) |
| **Monotonic Dependence** | Yes | Spearman’s rho, Kendall’s tau | Yes (Monotonic relationships can be nonlinear) |
| **Tail Dependence** | Yes | Tail dependence coefficient estimation via copulas | Yes (Nonlinear behavior in extreme events) |
| **Asymmetric Dependence** | Yes | Quantile regression, asymmetric copulas | Yes (Nonlinear dependence structures) |
| **Copula-based Dependence** | Yes | Copula goodness-of-fit tests, empirical copula comparison | Yes (High-dimensional copula modeling) |

---

## **Distributional Dependencies**

| Type | Testable | Method / Reason | Nonlinear / High-Dimensional |
|------|----------|-----------------|-----------------------------|
| **Stochastic Dependence** | Yes | Mutual information, joint vs. product-of-marginals tests | Yes (High-dimensional dependencies) |
| **Conditional Dependence** | Yes | Conditional mutual information, partial correlation | Yes (High-dimensional conditional dependence) |
| **Joint Dependence** | Yes | Multivariate dependence tests (e.g., copula models, PCA loading analysis) | Yes (High-dimensional dependencies) |
| **Marginal Dependence** | Yes | χ² test of independence, marginal correlation | No (Linear in nature) |
| **Higher-Order Dependence** | Yes | 3-variable tests: interaction terms, 3-way mutual information | Yes (High-dimensional interaction analysis) |
| **Graphical Dependence** | Yes | Structure learning (PC algorithm, GES, score-based methods) | Yes (Nonlinear graphical models, high-dimensional structure learning) |

---