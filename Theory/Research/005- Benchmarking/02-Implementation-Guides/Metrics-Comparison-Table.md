# Complete MSPC Metrics Comparison Table

Comprehensive comparison of all metrics used in Multivariate Statistical Process Control benchmarking.

## Main Comparison Matrix

| MSPC Metric | Definition | Use Case | Advantages | Limitations | Classification Equivalent |
|-------------|-----------|----------|------------|-------------|--------------------------|
| **ARL₀** | Expected samples until false alarm (IC) | Primary Phase II performance metric | Simple, fundamental measure | Can be skewed, doesn't show variability | Related to FPR |
| **ARL₁** | Expected samples until detection (OC) | Detection speed for specific shifts | Allows direct comparison | Shift-specific, requires simulation | Related to TPR/Recall |
| **SDRL** | Std dev of run lengths | Quantify consistency of timing | Shows predictability | Not robust to heavy tails | No direct equivalent |
| **MDRL** | Median of run lengths | Robust central tendency | Less affected by outliers | Less mathematically convenient | No direct equivalent |
| **FAR** | FP / Total IC samples | Cost of false alarms | Directly quantifiable risk | Threshold-dependent | **FPR** |
| **Detection Rate** | TP / Total OC samples | Chart sensitivity | Direct measure of ability | Requires shift window definition | **Recall / TPR** |
| **Power** | 1 - Type II Error | Detection probability | Linked to statistical tests | β specific to OC condition | **1 - FNR** |
| **Type I Error** | P(Signal \| IC) | Theoretical false alarm risk | Hypothesis testing basis | Less intuitive for sequential | **FPR** |
| **Type II Error** | P(No Signal \| OC) | Theoretical missed detection risk | Power calculation | Doesn't capture delay | **FNR** |
| **TTD** | Samples from shift to signal | Responsiveness measure | Practical and intuitive | Requires known shift point | Related to TP latency |
| **Precision** | TP / (TP + FP) | Alarm trustworthiness | Important for acceptance | Can be high with low recall | **Precision** |
| **Recall** | TP / (TP + FN) | Fault coverage | Critical for safety | May allow high FPR | **Recall / TPR** |
| **F1 Score** | Harmonic mean(Prec, Rec) | Balanced performance | Single summary metric | Hides individual trade-offs | **F1** |
| **Specificity** | TN / (TN + FP) | Correct rejection rate | Complements sensitivity | Inverse of FPR | **TNR** |
| **AUC** | ∫TPR d(FPR) | Discriminatory power | Threshold-independent | Doesn't give operating point | **AUC-ROC** |
| **Balanced Accuracy** | (TPR + TNR) / 2 | Overall imbalanced performance | Fair for rare faults | Masks individual failure modes | **BACC** |
| **Misclassification Rate** | (FP + FN) / Total | Phase I setup quality | Simple overall check | Doesn't distinguish error types | **1 - Accuracy** |

## Metric Relationships

### Direct Equivalents

| MSPC | Classification | Relationship |
|------|---------------|--------------|
| FAR | FPR | FAR ≈ FPR for MSPC context |
| Detection Rate | Recall/TPR | Identical concept |
| Type I Error | α level | α = P(False Alarm) |
| Type II Error | β level | β = P(Missed Detection) |
| Power | 1 - FNR | Power = 1 - β |

### Complementary Pairs

| Metric A | Metric B | Relationship |
|----------|----------|--------------|
| Recall | FNR | Recall = 1 - FNR |
| Specificity | FPR | Specificity = 1 - FPR |
| Precision | FDR | Precision = 1 - FDR |
| Detection Rate | Type II Error | Detection = 1 - Type II |

### Approximations

| MSPC | Approximation | Valid When |
|------|--------------|------------|
| ARL₀ | 1 / FAR | Memoryless charts (Shewhart) |
| ARL₀ | 1 / α | Independent observations |
| ARL₁ | 1 / (1 - β) | Simple shift scenarios |

## When to Use Each Metric

### Design Phase

| Goal | Primary Metrics | Secondary Metrics |
|------|----------------|-------------------|
| Compare chart types | ARL₀, ARL₁ | SDRL, MDRL |
| Set control limits | Type I Error (α) | ARL₀ target |
| Evaluate sensitivity | Power curves | Detection Rate vs shift |
| Assess consistency | SDRL | Run length distribution |

### Validation Phase

| Goal | Primary Metrics | Secondary Metrics |
|------|----------------|-------------------|
| Classifier performance | Precision, Recall, F1 | AUC, Balanced Accuracy |
| Phase I quality | Misclassification Rate | Type I/II Errors |
| Real-time readiness | ARL₀, ARL₁ | TTD, Signal Rate |

### Operational Monitoring

| Goal | Primary Metrics | Secondary Metrics |
|------|----------------|-------------------|
| Track false alarms | Signal Rate, FAR | ARL₀ |
| Evaluate detection | TTD, Detection Rate | ARL₁ |
| Assess reliability | Precision | NPV |
| Overall performance | F1 Score | Balanced Accuracy |

## Trade-offs Between Metrics

### ARL₀ vs ARL₁

| Adjustment | Effect on ARL₀ | Effect on ARL₁ | When to Use |
|-----------|---------------|---------------|-------------|
| Widen limits | ↑ (fewer false alarms) | ↑ (slower detection) | High false alarm cost |
| Narrow limits | ↓ (more false alarms) | ↓ (faster detection) | Critical fault detection |
| Increase chart memory | Variable | ↓ (better small shifts) | Persistent small shifts |

### Precision vs Recall

| Adjustment | Effect on Precision | Effect on Recall | When to Use |
|-----------|-------------------|-----------------|-------------|
| Raise threshold | ↑ (fewer FP) | ↓ (more FN) | False alarm cost > missed detection |
| Lower threshold | ↓ (more FP) | ↑ (fewer FN) | Missed detection cost > false alarm |
| Improve features | ↑ | ↑ | Always desirable |

## Metric Selection by Industry

### Semiconductor Manufacturing

| Priority | Metric | Reason |
|----------|--------|--------|
| 1 | TTD, ARL₁ | Fast detection critical for yield |
| 2 | Precision | Reduce costly false stops |
| 3 | AUC | Tool/recipe comparison |

### Pharmaceutical

| Priority | Metric | Reason |
|----------|--------|--------|
| 1 | Recall, Detection Rate | Regulatory compliance |
| 2 | Misclassification Rate | Batch validation |
| 3 | FAR, Type I Error | Process interruption cost |

### Automotive

| Priority | Metric | Reason |
|----------|--------|--------|
| 1 | F1 Score | Balance speed and reliability |
| 2 | Signal Rate | Line efficiency |
| 3 | SDRL | Consistency across shifts |

## Common Pitfalls

### Using Accuracy Alone
❌ **Problem**: Misleading with imbalanced data
✓ **Solution**: Use F1, Balanced Accuracy, or separate Precision/Recall

### Reporting Only ARL
❌ **Problem**: Ignores variability and distribution shape
✓ **Solution**: Report ARL, SDRL, and MDRL together

### Ignoring Shift Magnitude
❌ **Problem**: ARL₁ meaningless without shift context
✓ **Solution**: Report ARL curves for multiple shift sizes

### Comparing Different ARL₀
❌ **Problem**: Unfair comparison of methods
✓ **Solution**: Fix ARL₀, then compare ARL₁

### Confusing FAR and Precision
❌ **Problem**: FAR measures IC behavior, Precision measures alarm quality
✓ **Solution**: Use FAR for stability, Precision for alarm trustworthiness

## Minimum Reporting Requirements

### Academic Research

**Must report:**
- ARL₀ and ARL₁ for all methods
- SDRL or confidence intervals
- ARL curves for multiple shift magnitudes
- Comparison at fixed ARL₀

**Should report:**
- MDRL for skewed distributions
- TTD statistics
- Phase I misclassification rates

### Industrial Application

**Must report:**
- Detection Rate and FAR
- Precision and Recall
- Signal Rate per unit time

**Should report:**
- TTD for critical faults
- F1 Score for overall assessment
- AUC for method comparison

## Summary Decision Tree

```
Is your goal comparison of charts?
├─ Yes: Use ARL₀ (fixed), ARL₁, SDRL
└─ No: Is it classification evaluation?
    ├─ Yes: Use Precision, Recall, F1, AUC
    └─ No: Is it operational monitoring?
        ├─ Yes: Use Signal Rate, TTD, FAR
        └─ No: Use context-specific metrics
```

## Key Takeaways

1. **No single metric** tells the complete story
2. **Always report multiple metrics** from different perspectives
3. **Match metrics to stakeholder needs**: engineers vs. managers vs. researchers
4. **Consider costs**: balance false alarm cost vs missed detection cost
5. **Context matters**: semiconductor ≠ pharma ≠ automotive priorities
6. **Simulation needed**: for complex charts and shift scenarios
7. **Fixed comparison basis**: always fix ARL₀ when comparing ARL₁
