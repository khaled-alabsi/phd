Based on your exposé, the structure of your PhD work should reflect a **method-driven and application-focused** research design. Here's a structure aligned with your goals, contribution claims, and methodological flexibility:

---

## **1. Introduction**
- Motivation (e.g. criticality of early detection in production)
- Research gap (limitations of classical SPC + lack of robust multivariate anomaly tools)
- Research questions (directly map to exposé list)
- Contributions
- Outline of thesis structure

---

## **2. Background & Literature Review**
- Statistical Process Control (Shewhart, CUSUM, EWMA – uni- & multivariate)
- Assumptions: distributional (normality, independence)
- Modern anomaly detection (in time series & multivariate streams)
- Review of smoothing techniques
- Related software and implementations (R, Python)

---

## **3. Methodological Framework**
- Data assumptions: multivariate time series, centered variables
- Problem formalization: detection, attribution, robustness
- Evaluation setup: simulation and real-world requirements
- Control chart design goals (per exposé: detection, attribution, customization)

---

## **4. Alternative Multivariate Modeling**
- Analysis of the multivariate normality assumption
- Proposal of alternative distributions (skewed, t, mixture)
- Estimation procedures and parameter adaptation
- Simulation studies showing impact on chart performance

---

## **5. Control Chart Design & Optimization**
- New design approaches (based on new distributions or AI methods)
- Smoothing/integration: EWMA extensions, autoencoder smoothing, etc.
- Dynamic or adaptive control limits
- Optimization: grid search, learning-based tuning
- Visualization and interpretability tools

---

## **6. Implementation and Experimental Studies**
- Implementation details (Python modules)
- Simulated benchmark datasets
- Real-world or synthetic data with realistic disruptions
- Performance metrics (ARL, detection delay, attribution accuracy)
- Comparisons with baselines (Shewhart, Hotelling $T^2$ , etc.)

---

## **7. Case Studies / Industrial Application**
- Apply proposed methods to industrial-like data
- Show interpretability and adaptability in real use cases
- Evaluation of user-driven control chart customization

---

## **8. Discussion**
- Interpretation of results
- Sensitivity to assumptions (e.g. normality, shift size)
- Generalizability and scalability
- Limitations and trade-offs

---

## **9. Conclusion & Future Work**
- Recap of research questions and findings
- Theoretical and practical contributions
- Directions for future research (e.g. online learning, federated SPC)

---

