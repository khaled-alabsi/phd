# SDRL and MDRL: Run Length Variability Metrics

These metrics complement ARL by describing the **distribution** of run lengths, not just the average.

## SDRL (Standard Deviation of Run Length)

### Definition
$$\mathrm{SDRL} = \sqrt{\mathrm{Var}(\mathrm{RL})}$$

**Interpretation**: Measures the spread/variability in detection timing

### Why It Matters
- **Lower SDRL** → more consistent, predictable performance
- **Higher SDRL** → erratic behavior, uncertain timing
- Critical when false alarms or late detection have significant costs

### Example Comparison

| Chart | ARL (in-control) | SDRL | Interpretation |
|-------|-----------------|------|----------------|
| Chart A | 200 | 50 | Predictable, stable |
| Chart B | 200 | 180 | Same average but highly variable |

**Conclusion**: Chart A is preferable despite identical ARL due to lower variability.

### Calculation
- For **memoryless charts** (Shewhart): $\text{SDRL} = \sqrt{(1-p)}/p$ where p = signal probability
- For **CUSUM/EWMA**: Requires simulation or numerical integration

### Use Cases in MSPC

1. **Comparing Control Charts**
   - When ARLs are similar, choose the one with **lower SDRL** for reliability
   - Example: MCUSUM might have same ARL but lower SDRL than Shewhart for small shifts

2. **Shift-Specific Evaluation**
   - For small mean shifts, CUSUM and EWMA have **both lower ARL and SDRL**

3. **Risk-Sensitive Applications**
   - In high-cost environments (semiconductor, pharma)
   - Lower SDRL ensures consistent detection, reducing cost from random false alarms

---

## MDRL (Median Run Length)

### Definition
$$\text{MDRL} = \text{median}(\{RL_1, RL_2, ..., RL_N\})$$

**Interpretation**: The middle value of run length distribution; 50% of runs signal by this time

### Why It Matters
- **Less sensitive to outliers** than ARL
- **More robust** for skewed distributions
- Gives better sense of "typical" waiting time

### When to Use MDRL

| Use MDRL When | Use ARL When |
|--------------|--------------|
| Run lengths are right-skewed | Run lengths are symmetric |
| Heavy-tailed distributions | Normal-like distributions |
| Robust measure needed | Theoretical calculation preferred |
| Reporting to non-statisticians | Academic/theoretical work |

### Example

Simulated 1000 runs with sorted run lengths:
```
[4, 5, 5, 6, 6, 6, 7, 8, ..., 65, 80, 100]
```

- **MDRL** = 12 (500th value)
- **Interpretation**: 50% of the time, chart detects shift in ≤12 samples

### Comparison: MDRL vs ARL

| Metric | Sensitive to outliers? | Interpretability | Best when... |
|--------|----------------------|------------------|-------------|
| ARL | Yes (mean) | General trend | RLs symmetric |
| **MDRL** | No (median) | Robust & central | RLs skewed or heavy-tailed |

**Example scenario**:
- Run lengths: [1, 2, 3, 100, 200]
- **ARL** = 61.2 (inflated by large values)
- **MDRL** = 3 (typical run length)

### Calculation

1. Simulate or observe large number of run lengths
2. Sort the run lengths
3. Take the middle value (or average of two middle values if even)

```python
import numpy as np

def calculate_mdrl(run_lengths):
    """Calculate median run length."""
    return np.median(run_lengths)

# Example
run_lengths = [4, 5, 6, 7, 8, 10, 12, 15, 20, 50]
mdrl = calculate_mdrl(run_lengths)  # 9.0
```

---

## Related Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **ARL** | Mean time to detection | Standard comparison metric |
| **SDRL** | Variability in timing | Assessing consistency |
| **MDRL** | Median time to detection | Robust central tendency |
| **QR** | Interquartile range | Additional spread measure |

---

## Summary Comparison

| Aspect | ARL | SDRL | MDRL |
|--------|-----|------|------|
| **What it measures** | Average | Spread | Typical (median) |
| **Good if...** | Low (for OOC detection) | Low | Low (for OOC) |
| **Robust to outliers?** | No | No | Yes |
| **Used for** | Primary comparison | Consistency check | Skewed distributions |

---

## Practical Example: Semiconductor Etching

Monitoring plasma etching with Hotelling's T²:

### Before Parameter Tuning
- ARL₀ = 80
- SDRL = 16
- MDRL = 60

**Analysis**: High variability (SDRL=16) and discrepancy between ARL and MDRL suggest unstable detection.

### After Tuning (Kernel PCA)
- ARL₀ = 200
- SDRL = 8
- MDRL = 180

**Analysis**: Much more consistent performance, MDRL closer to ARL, lower SDRL indicates predictable behavior.

---

## Key Takeaways

1. **Always assess SDRL** alongside ARL for real-world deployment
2. **Low SDRL** critical in regulated/cost-sensitive industries
3. **MDRL provides robustness** when run length distributions are skewed
4. **Both metrics complement ARL**, providing complete performance picture
5. **SDRL reveals reliability**, MDRL reveals typicality
6. **In MSPC reports, include all three**: ARL (average), SDRL (consistency), MDRL (robust center)
