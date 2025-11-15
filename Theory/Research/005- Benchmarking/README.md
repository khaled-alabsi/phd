# MSPC Benchmarking Metrics

This folder contains comprehensive documentation on benchmarking metrics for Multivariate Statistical Process Control (MSPC) methods, particularly for fault detection and diagnosis.

## 📂 Folder Structure

### 01-Core-Metrics/
Detailed definitions of all key metrics:
- **ARL (Average Run Length)**: Expected samples until signal
- **SDRL (Standard Deviation of Run Length)**: Variability in detection timing
- **MDRL (Median Run Length)**: Robust central tendency measure
- **FAR (False Alarm Rate)**: Rate of false positives
- **Detection Rate & Power**: Ability to detect true faults
- **Classification Metrics**: Precision, Recall, F1, AUC, etc.

### 02-Implementation-Guides/
Practical implementation approaches:
- ARL calculation methods (analytical vs. simulation)
- Monte Carlo simulation for complex scenarios
- Comprehensive metrics comparison tables

### 03-Examples-and-QA/
Learning resources:
- Worked examples with numerical calculations
- Industry-specific practice questions (semiconductor, automotive, pharma)
- Step-by-step solutions

### 04-Context/
Background and applications:
- MSPC benchmarking overview and motivation
- Standard datasets (Tennessee Eastman Process, etc.)
- Industry-specific applications

## 🎯 Quick Reference

### When to Use Each Metric

| Goal | Recommended Metrics |
|------|-------------------|
| Compare false alarm rates | ARL₀, FAR, Type I Error |
| Measure detection speed | ARL₁, TTD, Detection Rate |
| Assess consistency | SDRL, MDRL |
| Evaluate classification | Precision, Recall, F1, AUC |
| Handle imbalanced data | Balanced Accuracy, F1 |

### Common Metric Relationships

- **ARL₀ = 1 / FAR** (for memoryless charts)
- **Detection Rate = 1 - Type II Error**
- **Specificity = 1 - FPR**
- **F1 = 2 × (Precision × Recall) / (Precision + Recall)**

## 🚀 Getting Started

1. Start with **04-Context/MSPC-Benchmarking-Overview.md** for background
2. Review **01-Core-Metrics/** for metric definitions
3. Check **02-Implementation-Guides/** for calculation methods
4. Practice with **03-Examples-and-QA/** scenarios

## 📚 Key Concepts

### Phase I vs Phase II
- **Phase I**: Historical data analysis, model building, control limit establishment
- **Phase II**: Real-time monitoring, online fault detection

### In-Control vs Out-of-Control
- **In-Control (IC)**: Process operating normally (ARL₀, FAR relevant)
- **Out-of-Control (OC)**: Process experiencing faults (ARL₁, Detection Rate relevant)

### Trade-offs
- Sensitivity vs False Alarms: Higher sensitivity → more false alarms
- ARL₀ vs ARL₁: Tightening limits reduces ARL₀ but may increase ARL₁
- Precision vs Recall: Depends on cost of false positives vs false negatives
