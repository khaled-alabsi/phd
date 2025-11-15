# Classification Metrics for MSPC

This document maps traditional classification metrics to their usage in Multivariate Statistical Process Control (MSPC) for fault detection and diagnosis.

## MSPC Context Mapping

### Process States
- **In-Control (IC)**: Process operating normally → **Negative Class**
- **Out-of-Control (OC)**: Process experiencing fault → **Positive Class**

### Chart Signals
- **Signal: Yes** → **Predicted Positive**
- **Signal: No** → **Predicted Negative**

### Outcomes
| MSPC Outcome | Classification Equivalent |
|--------------|--------------------------|
| False Alarm (IC, Signal Yes) | False Positive (FP) |
| Missed Alarm (OC, Signal No) | False Negative (FN) |
| Correct Rejection (IC, Signal No) | True Negative (TN) |
| Correct Detection (OC, Signal Yes) | True Positive (TP) |

## Confusion Matrix

|                     | **Predicted Normal (0)** | **Predicted Fault (1)** |
|---------------------|-------------------------|-------------------------|
| **True Normal (0)** | TN (correct)            | FP (false alarm)        |
| **True Fault (1)**  | FN (missed fault)       | TP (correct detection)  |

## Core Metrics

### 1. Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Interpretation**: Overall correctness

**Limitation in MSPC**:
- Misleading with rare faults (class imbalance)
- High accuracy possible even with poor fault detection
- Example: Always predicting "normal" gives 95% accuracy if only 5% are faults

**Use case**: Only when classes are balanced

---

### 2. Precision (Positive Predictive Value)
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interpretation**: Of all alarms raised, how many were actual faults?

**MSPC Context**:
- Low precision → many false alarms → alarm fatigue
- Important when false alarms are costly (shutdowns, investigations)
- **High precision = trustworthy alarms**

**Example**: 100 alarms raised, 80 were real faults → Precision = 0.80

---

### 3. Recall / Sensitivity / TPR (True Positive Rate)
$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interpretation**: Proportion of actual faults correctly detected

**MSPC Context**:
- High recall critical in safety-sensitive environments
- Low recall means faults go undetected → damage/hazards
- **Also called Detection Rate or Power in MSPC**

**Example**: 100 faults occurred, 85 detected → Recall = 0.85

**Equivalent MSPC Metrics**:
- Detection Rate
- Power (1 - Type II Error)
- PAR (Positive Alarm Rate)

---

### 4. Specificity / TNR (True Negative Rate)
$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Interpretation**: Proportion of normal conditions correctly identified

**MSPC Context**:
- High specificity → fewer false alarms
- Preserves operator trust
- Avoids unnecessary shutdowns

**Relationship**: Specificity = 1 - FPR

---

### 5. False Positive Rate (FPR)
$$\text{FPR} = \frac{FP}{FP + TN}$$

**Interpretation**: Rate of false alarms during normal operation

**MSPC Context**:
- **Directly equivalent to FAR in MSPC**
- High FPR → alarm fatigue, wasted investigations
- Target: minimize FPR while maintaining detection capability

**Relationship**: FPR = 1 - Specificity

**MSPC Equivalent**: False Alarm Rate (FAR), Type I Error

---

### 6. False Negative Rate (FNR)
$$\text{FNR} = \frac{FN}{FN + TP}$$

**Interpretation**: Rate of missed faults

**MSPC Context**:
- **Most dangerous metric** in safety-critical monitoring
- High FNR → faults silently ignored
- Directly impacts risk and quality

**Relationship**: FNR = 1 - Recall

**MSPC Equivalent**: Type II Error, Missed Detection Rate

---

### 7. F1 Score
$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Interpretation**: Harmonic mean of Precision and Recall

**MSPC Context**:
- Balances false alarms (Precision) and missed detections (Recall)
- Best for evaluating fault classifiers where both errors are costly
- Drops sharply if either Precision or Recall is poor

**When to use**: Need single metric balancing both error types

---

### 8. Balanced Accuracy
$$\text{BACC} = \frac{1}{2}(\text{Recall} + \text{Specificity})$$

**Interpretation**: Average of detection rate (Recall) and correct rejection rate (Specificity)

**MSPC Context**:
- Fair performance on imbalanced data (few faults)
- Shows overall balance between detecting faults and avoiding false alarms
- Hides individual performance details

---

### 9. NPV (Negative Predictive Value)
$$\text{NPV} = \frac{TN}{TN + FN}$$

**Interpretation**: When system says "normal," how often is it correct?

**MSPC Context**:
- Important for trusting "all clear" signals
- Low NPV → can't rely on normal predictions
- Critical when faults are rare

---

### 10. FDR (False Discovery Rate)
$$\text{FDR} = \frac{FP}{FP + TP}$$

**Interpretation**: Among detected anomalies, how many were false?

**MSPC Context**:
- High FDR undermines system credibility
- Operators may start ignoring outputs
- Complement of Precision: FDR = 1 - Precision

---

### 11. AUC (Area Under ROC Curve)
$$\text{AUC} = \int_0^1 \text{TPR} \, d(\text{FPR})$$

**Interpretation**: Ability to distinguish between IC and OC states across all thresholds

**MSPC Context**:
- **Direct equivalent from classification**
- Threshold-independent measure
- AUC = 1.0: perfect separation
- AUC = 0.5: no better than random
- Used to compare detection statistics (T², SPE, etc.)

**Use case**: Evaluating inherent discriminatory power before parameter tuning

---

## Metric Relationships and Redundancies

### Complementary Pairs

| Metric | Complement | Relationship |
|--------|-----------|--------------|
| Recall | FNR | Recall = 1 - FNR |
| Specificity | FPR | Specificity = 1 - FPR |
| Precision | FDR | Precision = 1 - FDR |

### When to Drop Redundant Metrics

| Drop This | If You Have | Reason |
|-----------|------------|--------|
| FNR | Recall | Opposite views of same performance |
| FPR | Specificity (TNR) | TNR shows success, FPR shows failure |
| FDR | Precision | Precision more intuitive |
| Balanced Accuracy | Recall + TNR | Can compute directly |
| F1 Score | Precision + Recall | Use only for summary |

## Use-Case Driven Selection

| Goal | Recommended Metric | Reason |
|------|-------------------|--------|
| Detect all faults | **Recall (TPR)** | Missing faults could cause damage |
| Avoid false alarms | **Precision, FPR** | False positives cause alarm fatigue |
| Trust "normal" output | **NPV** | Operators rely on normal predictions |
| Compare on imbalanced data | **Balanced Accuracy, F1** | Raw accuracy biased when faults rare |
| Threshold-independent comparison | **AUC** | Evaluates across all possible thresholds |

## Perfect Performance

If both **Recall = 1.0** and **Precision = 1.0**:

| Metric | Value | Meaning |
|--------|-------|---------|
| Precision | 1.0 | No false positives (FP = 0) |
| Recall | 1.0 | No false negatives (FN = 0) |
| Specificity | 1.0 | All normals correctly ignored |
| FPR | 0.0 | No false alarms |
| FNR | 0.0 | No missed faults |
| FDR | 0.0 | All alarms are true |
| NPV | 1.0 | All predicted-normal truly normal |
| Accuracy | 1.0 | All predictions correct |
| F1 Score | 1.0 | Perfect balance |

**Reality check**: Very rare unless faults are highly distinct or data is overfit.

## Comparison: MSPC Metrics vs Classification Metrics

| MSPC Metric | Definition | Classification Equivalent |
|-------------|-----------|--------------------------|
| FAR (False Alarm Rate) | FP / Total IC samples | **FPR** |
| Detection Rate / Power | TP / Total OC samples | **Recall / TPR** |
| Type I Error | P(Signal \| IC) | **FPR / α** |
| Type II Error | P(No Signal \| OC) | **FNR / β** |
| Misclassification Rate | (FP + FN) / Total | **1 - Accuracy** |
| ARL₀ | 1 / FAR (approx.) | Related to **FPR** |
| ARL₁ | Detection delay | Related to **Recall** and time |
| TTD (Time to Detection) | Samples from shift to signal | No direct equivalent (temporal) |

## Multi-Class Fault Detection

For multi-class scenarios (different fault types):

### Per-Class Metrics
- **FDR per fault**: Among predictions of fault k, how many were wrong?
- **FAR per fault**: How often fault k predicted during normal operation?
- **ARL₁ per fault**: Average delay to detect specific fault k

### Aggregate Metrics
- **Macro-averaged**: Average metric across all fault classes (treats all equally)
- **Micro-averaged**: Pool all TP, FP, FN, TN then calculate (weighted by class frequency)
- **Weighted average**: Weight by class importance or frequency

## Key Takeaways

1. **Classification metrics apply directly to MSPC** with proper context mapping
2. **Choose metrics based on costs**: false alarms vs missed detections
3. **Watch for redundancy**: don't report both FPR and Specificity
4. **Consider imbalance**: use F1, Balanced Accuracy, or AUC for rare faults
5. **Precision + Recall** together give complete picture of performance
6. **AUC is threshold-independent**, good for comparing methods
7. **Context matters**: semiconductor vs pharma may prioritize different metrics
