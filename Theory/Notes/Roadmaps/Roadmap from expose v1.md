# PhD Roadmap: Monitoring Disruptions in Production Processes via Multivariate Control Charts and AI

## Year 1 – Foundation & Orientation

- **Literature Review**
  - Classical SPC methods (Shewhart, CUSUM, EWMA, Hotelling's T²)
  - Multivariate distribution assumptions in control charts
  - State-of-the-art in anomaly detection (with/without control charts)
  - AI approaches in SPC context

- **Tooling**
  - Deep dive into Python packages: `pyspc`, `qcc`, `scikit-multiflow`, etc.
  - Start with synthetic datasets for understanding sensitivity/specificity

- **Research Questions Refinement**
  - Formalize definitions of "disruption", "anomaly", and "contribution of variables"
  - Identify initial gaps in literature

- **Experiments**
  - Reproduce baseline methods on benchmark datasets
  - Explore behavior under misspecification of multivariate normality

---

## Year 2 – Method Development Phase I

- **Statistical Methodology**
  - Propose alternative distributions (e.g., elliptical, skew-normal, copulas)
  - Theoretical modeling and derivation of control limits

- **Algorithm Prototyping**
  - Implement baseline + extended methods in Python
  - Design adaptive control charts with flexible thresholds

- **Simulation Framework**
  - Develop modular simulation for performance evaluation (ARL, detection delay)
  - Simulate multivariate data with dependencies and different noise regimes

- **Preliminary Results**
  - Identify design bottlenecks
  - Validate against synthetic and public datasets

---

## Year 3 – Method Development Phase II & Dissemination

- **Advanced Techniques**
  - Integrate smoothing techniques (e.g., wavelets, LSTMs, EWMA hybrids)
  - Embed AI methods (e.g., VAEs, graph-based detectors)

- **Anomaly Attribution**
  - Develop metrics to quantify component-wise contributions
  - Explore SHAP-like techniques for interpretability in multivariate SPC

- **Conferences**
  - Present at SPC, quality engineering, or AI in manufacturing events
  - Network with Prof. Knoth’s group and similar labs

- **Publication Drafting**
  - Target methodological innovation paper (e.g., new control chart design)

---

## Year 4 – Maturation, Industrial Relevance, and Scaling

- **System Refinement**
  - Improve robustness under real-world noise, non-stationarity, and missing data
  - Apply methods on real industrial datasets (via collaboration if possible)

- **Software Contribution**
  - Release research code as Python packages (documented, tested)

- **Publications**
  - Publish applied case studies
  - Submit journal paper on anomaly contribution measures

---

## Year 5 – Consolidation & Dissertation

- **Final Evaluations**
  - Compare all proposed models on standard and industrial datasets
  - Create comprehensive evaluation matrix (ARL, FDR, delay, etc.)

- **Writing**
  - Dissertation chapters: methodology, evaluation, application, impact
  - Final publications and journal submissions

- **Dissemination**
  - Final talk at Promotionszentrum
  - Conference presentations
  - Package finalization and documentation
