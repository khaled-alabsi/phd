The PhD work results should **demonstrate novelty, rigor, and impact**. Here’s what they **must** include:

---

### 1. **Theoretical Contributions**
- A **novel or extended method** for monitoring multivariate time series or control charts
- Mathematical formulation (e.g., new test statistic, new limit function, non-normal assumptions)
- Proofs of properties: e.g., control limit derivation, false alarm rate under certain assumptions

---

### 2. **Algorithmic Implementation**
- A **working Python implementation** (ideally a small open-source package)
- Efficient code with modular design for adaptation in real-world setups

---

### 3. **Quantitative Evaluation**
- Evaluation on **synthetic datasets** with controlled scenarios:
  - Known changes, variable correlations, non-Gaussian structures
  - Metrics: ARL, detection delay, TPR/FPR, variable attribution accuracy
- Evaluation on **real-world industrial time series**, ideally with labeled disruptions

---

### 4. **Interpretability Layer**
- A method or tool for **explaining which variable(s)** caused the control chart signal
- Clear visualization or explanation of attribution (e.g., projection methods, SHAP analogues)

---

### 5. **Scientific Communication**
- At least **2 high-quality publications**:
  - 1 on method/theory (e.g., control chart or detection strategy)
  - 1 on application (e.g., manufacturing, anomaly contribution)
- Poster/talk at a relevant conference (e.g., QSR, IFAC, COMPSTAT)

---

### 6. **Dissertation Artifacts**
- Structured and reproducible experiments (configs, seeds, logs)
- Figures: control charts, contribution plots, synthetic case studies
- A clearly written thesis that connects SPC with AI and anomaly reasoning

---

### 7. **Optional but Strong Add-ons**
- Python/R package release with docs + examples
- Industrial partner use-case (validates relevance)
- Robustness study: non-normality, missing data, seasonality

---

If you want to show academic rigor **and** practical relevance, then your results must include:
- A new method (statistical/machine learning based)
- A validated implementation
- A framework to understand **why a process went out of control**
- Proof that it works better than current state-of-the-art in at least one realistic setting

---

The exposé does **not require** you to work with one specific method or distribution. It **suggests directions**, such as:

- **Exploring alternative multivariate distributions** (instead of always using the multivariate normal)
- **Trying different smoothing and anomaly detection techniques**
- **Using AI/ML (e.g., neural networks)** to enhance control charts
- **Designing control limits optimally**, possibly based on new criteria

These are **open research questions**, not fixed requirements. The exposé gives you **flexibility** to choose the most appropriate methods as long as:
- They fit the scope of multivariate SPC and anomaly detection
- They address robustness, interpretability, or performance
- They lead to scientific contributions

You’re expected to justify your choices based on empirical evidence or theoretical insights.