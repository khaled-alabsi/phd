# Detailed explanation of each **dependency family**

### **Temporal Dependencies**

Temporal dependencies refer to relationships where past or future values influence the current values of a time series or process.

- **Temporal Dependence**: Refers to the influence that past values have on the current value in time series data. For instance, an autoregressive (AR) model predicts future values based on past observations.
- **Instantaneous Dependence**: Occurs when variables are influenced by one another at the same moment in time. This is often modeled in systems where contemporaneous effects are significant, such as in structural vector autoregressions (SVAR).
- **Long-range Dependence (LRD)**: Describes how the dependency between data points decays slowly over time. This can be observed in phenomena such as financial market returns where distant observations remain correlated over long periods.
- **Cyclostationary Dependence**: Involves periodic dependencies that repeat in cycles. For example, seasonal patterns in weather or sales data, where dependencies vary in a predictable manner over time.
- **Nonstationary Dependence**: This occurs when the statistical properties (like mean or variance) of a time series change over time. Time series models like trending or structural break tests capture such dependencies.
- **Non-Markovian Dependence**: Describes scenarios where the future state of a process depends not only on the current state but also on the sequence of past states. This is important in systems with memory effects.
- **Functional Time Series Dependence**: Involves dependencies between functions (curves) over time, such as when studying stock price movements across days, where each day’s price can be viewed as a function in time.
- **Regime-dependent Dependence**: Occurs when the system behaves differently in different regimes (e.g., economic booms vs. recessions). Models like Markov switching allow for capturing this type of dependence.

---

### **Spatial Dependencies**

Spatial dependencies refer to relationships between observations that are geographically or spatially close to each other.

- **Spatial Dependence**: This describes situations where observations located near each other in space are more likely to be similar than those far apart. Examples include weather stations or regional economic data, where neighboring areas exhibit correlated behavior.
- **Block Dependence**: This occurs when groups or blocks of observations exhibit stronger dependencies within their group compared to between groups. For example, data collected from different neighborhoods might show strong internal correlations within each neighborhood.

---

### **Causal & Directional Dependencies**

This family focuses on cause-and-effect relationships or directional flows between variables.

- **Directional (Causal) Dependence**: A direct cause-effect relationship where changes in one variable cause changes in another. In economics, for example, changes in interest rates can affect inflation.
- **Granger-type Dependence**: A form of causal dependency where one variable helps predict the future values of another. In Granger causality tests, if past values of one variable contain information useful for predicting another variable, a directional dependence is assumed.
- **Nonlinear Granger-type Dependence**: A variation of the Granger causality test where nonlinear relationships are modeled. It uses techniques like kernel methods or neural networks to capture more complex dependencies.
- **Latent Dependence**: Occurs when there are hidden (latent) variables that cause dependencies between observed variables. These latent variables may not be directly measurable but still influence the data.
- **Induced Dependence**: Arises due to sampling design or data structure, where dependence is introduced by the way data is collected or by conditioning on certain variables.
- **Pseudo-dependence**: Refers to apparent dependencies that arise due to confounding factors or statistical artifacts, rather than true causal relationships.
- **Mixture-induced Dependence**: Occurs when data consists of several subgroups or mixtures, each with its own underlying distribution, and the dependencies arise from the combination of these subgroups.
- **Contextual Dependence**: Describes how the strength or nature of dependencies changes depending on the context or conditions, such as in stratified data or when interactions between variables are influenced by other variables.

---

### **Functional/Mathematical Dependencies**

This family involves dependencies driven by deterministic or functional relationships between variables.

- **Functional Dependence**: This occurs when one variable is a deterministic function of another. For example, in physics, the position of an object may be a function of time, such as in Newton's laws of motion.
- **Linear Dependence**: Involves relationships where a variable can be expressed as a linear combination of others. For instance, in linear regression, the dependent variable is modeled as a linear function of the independent variables.
- **Nonlinear Dependence**: Occurs when the relationship between variables is not linear. Examples include exponential growth models, quadratic relationships, and more complex forms of interaction between variables.
- **Monotonic Dependence**: This occurs when one variable either increases or decreases in a consistent direction as the other variable changes. For instance, as temperature increases, ice melts, which is a monotonic relationship.
- **Tail Dependence**: Refers to dependencies observed in the extremes (tail ends) of distributions. In finance, for example, tail dependence could be seen in the correlation between stock returns during market crashes.
- **Asymmetric Dependence**: Describes relationships where the strength of dependence varies depending on the direction or magnitude of the relationship. This is common in financial markets, where positive and negative shocks might have asymmetric impacts.
- **Copula-based Dependence**: Copulas are used to model and measure dependencies between random variables, especially when the relationships are not well captured by traditional correlation measures. They can model both linear and nonlinear dependencies across multiple dimensions.

---

### **Distributional Dependencies**

This family describes dependencies related to joint and conditional distributions of variables.

- **Stochastic Dependence**: Refers to probabilistic dependencies, where the joint distribution of random variables is different from the product of their marginal distributions. This kind of dependence is common in multivariate analysis.
- **Conditional Dependence**: Occurs when the relationship between two variables depends on a third. For example, two variables may be independent when conditioned on a third, but dependent when unconditioned.
- **Joint Dependence**: Refers to the overall relationship between multiple variables, such as when multiple variables exhibit some common structure or pattern in their joint distribution.
- **Marginal Dependence**: This focuses on the dependency between the marginal distributions of variables. While it is often simpler, it can reveal important insights into univariate relationships.
- **Higher-Order Dependence**: Refers to complex dependencies involving more than two variables. It’s important for understanding interactions and non-linear relationships in multivariate data.
- **Graphical Dependence**: Involves the representation of dependencies through graphical models, where variables are nodes, and edges represent direct dependencies. These models help visualize complex relationships between variables and are used in machine learning (e.g., Bayesian networks, Markov networks).
