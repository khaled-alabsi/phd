# Type of Causality

### **1. Temporal Causality**

- **Granger Causality**  
  A variable $X$ Granger-causes $Y$ if past values of $X$ help predict future values of $Y$ beyond what past $Y$ alone can.  
  *Testable*: Yes. *Method*: Granger causality test (F-test or Wald test).

- **Instantaneous Causality**  
  $X_t$ and $Y_t$ influence each other within the same time period (no lag).  
  *Testable*: Yes. *Method*: Instantaneous causality tests using structural VAR or contemporaneous correlation.

- **Lagged (Delayed) Causality**  
  Effects of $X_t$ are not seen immediately but at $Y_{t+h}$ , $h > 0$.  
  *Testable*: Yes. *Method*: Lag-augmented causality tests or impulse response analysis.

---

### **2. Structural Causality**

- **Mechanistic Causality**  
  Based on physical or biological mechanisms; e.g., pushing a switch causes a light to turn on.  
  *Testable*: Sometimes. *Method*: Requires domain-specific experimental or physical validation.

- **Structural Equation Modeling (SEM)**  
  Uses systems of equations to represent cause-effect relationships.  
  *Testable*: Yes. *Method*: Model fitting tests such as chi-squared, RMSEA, CFI.

- **Causal Graphs / DAGs (Directed Acyclic Graphs)**  
  Represent cause-effect structures using graph theory.  
  *Testable*: Yes. *Method*: PC, FCI, or GES algorithms using conditional independence tests.

---

### **3. Counterfactual Causality**

- **Potential Outcomes Framework (Rubin Causal Model)**  
  A treatment causes an effect if the outcome under treatment differs from what would have happened without it.  
  *Testable*: Yes. *Method*: Randomized experiments, matching, or propensity score methods.

- **Do-Calculus (Pearl’s Causal Inference)**  
  Defines causal effects using do-operators: $P(Y \mid \text{do}(X=x))$.  
  *Testable*: Yes. *Method*: Do-calculus rules, back-door/front-door adjustment, identification algorithms.

---

### **4. Statistical & Data-Driven Causality**

- **Conditional Independence-Based Causality**  
  Algorithms learn causal structure from observed conditional independencies.  
  *Testable*: Yes. *Method*: CI tests (e.g., kernel-based, partial correlation, G-test).

- **Information-Theoretic Causality**  
  Measures like transfer entropy quantify directional dependence.  
  *Testable*: Yes. *Method*: Transfer entropy, conditional mutual information.

- **Additive Noise Models (ANMs)**  
  Assume that $Y = f(X) + \varepsilon$ , with $\varepsilon \perp X$.  
  *Testable*: Yes. *Method*: Regression followed by independence testing (e.g., HSIC).

- **Invariant Causal Prediction (ICP)**  
  Uses stability of causal relationships across different environments.  
  *Testable*: Yes. *Method*: ICP testing procedures based on invariance across data partitions.

---

### **5. Philosophical / Conceptual Causality**

- **Necessary Cause**  
  Without the cause, the effect cannot happen.  
  *Testable*: Partially. *Method*: Elimination experiments or counterfactual logic.

- **Sufficient Cause**  
  The presence of the cause guarantees the effect.  
  *Testable*: Partially. *Method*: Controlled experiments or logical reasoning.

- **INUS Conditions**  
  Insufficient but Necessary part of an Unnecessary but Sufficient condition.  
  *Testable*: No (direct). *Reason*: Complex logic-based framework not easily empirically testable.

- **Probabilistic Causality**  
  The cause increases the probability of the effect.  
  *Testable*: Yes. *Method*: Check if $P(Y \mid X) > P(Y)$ using statistical tests.