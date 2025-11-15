# Introduction

## Fault Diagnosis Method Categories — with TEP Examples

### (1) **Data-Driven Methods**

**Definition**: Use historical process data to learn patterns or anomalies without explicit system modeling.

**TEP Example**:

* Apply **Principal Component Analysis (PCA)** on TEP sensor data (e.g., reactor pressure, component concentrations).
* Train PCA on fault-free data to capture normal behavior.
* When a new data point shows a large deviation from the PCA subspace, it’s flagged as a fault.

**Pros**:

* No need to model the chemical reactions or mass balances.
* Can detect unknown faults if enough training data is available.

**Cons**:

* Cannot explain *why* the fault occurred.
* Requires labeled or at least clean data.

---

### (2) **Analytical Methods**

**Definition**: Use a mathematical model of the physical/chemical process to infer faults based on model-data consistency.

**TEP Example**:

* Build a state-space model using known reaction kinetics and mass/energy balances.
* Design an **observer** (e.g., Kalman filter) to estimate internal states like reactor temperature.
* Compute **residuals**: difference between measured and estimated values.
* A persistent large residual → model inconsistency → possible fault (e.g., sensor bias or valve leak).

**Pros**:

* Can isolate faults more precisely.
* Better interpretability: residuals linked to physical subsystems.

**Cons**:

* Requires accurate models of the nonlinear reaction system in TEP.
* Sensitive to modeling errors and parameter uncertainty.

---

### (3) **Knowledge-Based Methods**

**Definition**: Use expert rules, heuristics, or operator knowledge to detect or isolate faults.

**TEP Example**:

* IF reactor pressure rises AND cooling water flow drops, THEN suspect heat exchanger fouling.
* Use a rule-based system encoded by plant operators who know the fault signatures.

**Pros**:

* Easy to interpret.
* Can detect faults even when data-driven or analytical models fail.

**Cons**:

* Not scalable to all 21 faults in TEP.
* Depends heavily on expert availability and correct knowledge encoding.

---

### Mapping Summary (for TEP)

| **Category**    | **TEP Use Case**                                     | **Core Input**              | **Output**                 |
| --------------- | ---------------------------------------------------- | --------------------------- | -------------------------- |
| Data-Driven     | PCA detects fault #3 (reflux failure)                | Historical sensor data      | Anomaly flags              |
| Analytical      | Observer detects fault #1 (A/C feed loss)            | System model + measurements | Residuals, fault isolation |
| Knowledge-Based | Rules detect fault #10 (cooling water flow decrease) | Expert knowledge + signals  | Fault classification       |

---
