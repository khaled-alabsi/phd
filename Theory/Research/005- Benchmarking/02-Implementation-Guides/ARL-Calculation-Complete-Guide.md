# Complete Guide to ARL Calculation for Multi-Fault Classification

This guide provides implementation details for calculating ARL₀ and ARL₁ in multi-class fault detection scenarios, such as Tennessee Eastman Process (TEP) benchmarking.

## Problem Statement

**Given:**
- True fault label sequence: $y_{\text{true}} = [y_1, y_2, ..., y_n]$ where $y_t \in \{0, 1, 2, ..., F\}$
  - 0 = no fault (normal operation)
  - 1..F = different fault types
- Predicted fault label sequence: $y_{\text{pred}} = [\hat{y}_1, \hat{y}_2, ..., \hat{y}_n]$

**Objective:**
1. Compute **ARL₁**: Average detection delay from fault start to first correct detection
2. Compute **ARL₀**: Average time between false alarms during normal operation

## ARL₁: Detection Delay Calculation

### Method

1. **Segment Faults**: Identify continuous fault segments in $y_{\text{true}}$
2. **Find Detection**: For each segment, locate first correct detection
3. **Calculate Delay**: Measure time from segment start to detection
4. **Average**: Mean delay across all fault segments

### Fault Segmentation

**Definition**: A fault segment is a maximal contiguous run of the same fault type.

**Example:**
```python
y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0]
```
- Segment 1: Fault 1 at indices [3, 4, 5]
- Segment 2: Fault 2 at indices [6, 7, 8]

### Detection Delay per Segment

For each segment $[t_s, t_e]$ of fault $f$:

$$\text{Delay}_f =
\begin{cases}
t_d - t_s, & \text{if detected (first } y_{\text{pred}}[t_d] = f \text{ for } t_d \geq t_s\text{)} \\
t_e - t_s + 1, & \text{if not detected (penalty)}
\end{cases}$$

**Penalty**: Undetected faults count full segment length to avoid underestimating ARL.

### Aggregation

$$ARL_1 = \frac{1}{N} \sum_{i=1}^N \text{Delay}_{f_i}$$

where $N$ = number of fault segments.

### Numerical Example

| t | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $y_{\text{true}}$ | 0 | 0 | 0 | 1 | 1 | 1 | 2 | 2 | 2 | 0 | 0 |
| $y_{\text{pred}}$ | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 2 | 0 | 0 |

**Fault 1**: $t_s=3, t_e=5$ → never detected → Delay = $5 - 3 + 1 = 3$
**Fault 2**: $t_s=6, t_e=8$ → detected at $t=6$ → Delay = $6 - 6 = 0$

$$ARL_1 = \frac{3 + 0}{2} = 1.5$$

## ARL₀: False Alarm Rate Calculation

### Method

1. **Identify Normal Periods**: Find all $t$ where $y_{\text{true}}[t] = 0$
2. **Detect False Alarms**: Find $t$ where $y_{\text{pred}}[t] \neq 0$ (during normal)
3. **Measure Intervals**: Calculate gaps between consecutive false alarms
4. **Average**: Mean interval length

### Formula

For false alarms at times $t_1, t_2, ..., t_k$ during normal operation:

$$\text{Intervals} = [t_2 - t_1, t_3 - t_2, ..., t_k - t_{k-1}]$$

$$ARL_0 = \frac{1}{k-1} \sum_{i=1}^{k-1} (t_{i+1} - t_i)$$

**Special cases:**
- 0 false alarms → $ARL_0 = \infty$
- 1 false alarm → $ARL_0 = \text{total normal samples}$ or $\infty$ (based on convention)

### Numerical Example

Normal intervals: $t=0..2, 9..10$ (6 total normal samples)

| Scenario | False Alarms | ARL₀ Calculation |
|----------|--------------|-----------------|
| None | - | $\infty$ |
| 1 at t=1 | [1] | $\infty$ (or 6) |
| 2 at t=1, 9 | [1, 9] | $(9-1) = 8$ |

## Python Implementation

### Complete Multi-Class ARL Calculation

```python
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

def segment_faults(y_true: List[int]) -> List[Tuple[int, int, int]]:
    """
    Identify continuous fault segments.

    Returns:
        List of (fault_label, start_index, end_index)
    """
    segments = []
    n = len(y_true)
    start = None
    current_fault = 0

    for i in range(n):
        if y_true[i] != 0:
            if current_fault != y_true[i]:
                if current_fault != 0:
                    segments.append((current_fault, start, i - 1))
                current_fault = y_true[i]
                start = i
        else:
            if current_fault != 0:
                segments.append((current_fault, start, i - 1))
                current_fault = 0
                start = None

    if current_fault != 0:
        segments.append((current_fault, start, n - 1))

    return segments


def compute_arl_per_fault(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Calculate ARL₀ and ARL₁ for each fault type.

    Returns:
        DataFrame with columns: Fault, ARL0, ARL1
    """
    faults = np.unique(y_true[y_true > 0])
    results = []

    def find_segments(arr: np.ndarray, fault: int):
        """Find continuous segments of a specific fault."""
        segments = []
        start = None
        for i, val in enumerate(arr):
            if val == fault and start is None:
                start = i
            elif val != fault and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(arr)))
        return segments

    for fault in faults:
        # ARL₀: False alarms during normal operation
        false_alarm_idxs = np.where((y_true == 0) & (y_pred == fault))[0]

        if len(false_alarm_idxs) < 2:
            arl0 = float('inf')
        else:
            gaps = np.diff(false_alarm_idxs) - 1
            arl0 = float(np.mean(gaps))

        # ARL₁: Detection delay in fault segments
        segments = find_segments(y_true, fault)
        delays = []

        for start, end in segments:
            segment_preds = y_pred[start:end]
            correct_indices = np.where(segment_preds == fault)[0]

            if len(correct_indices) == 0:
                # Not detected: penalty = full segment length
                delay = end - start
            else:
                # Detected: delay from segment start to first detection
                delay = correct_indices[0]

            delays.append(delay)

        arl1 = float(np.mean(delays)) if delays else float('inf')

        results.append({"Fault": int(fault), "ARL0": arl0, "ARL1": arl1})

    return pd.DataFrame(results).sort_values("Fault").reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    # Example data
    y_true = np.array([
        0, 0, 2, 2, 2, 0,  # Fault 2: indices 2-4
        1, 1,              # Fault 1: indices 6-7
        0, 0,
        3, 3, 3, 3,        # Fault 3: indices 10-13
        0, 0, 0, 0, 0,
        2, 2,              # Fault 2: indices 19-20
    ])

    y_pred = np.array([
        0, 2, 0, 2, 2, 0,  # False alarm at 1, detection at 3-4
        0, 0,              # Missed fault 1
        1, 0,              # False alarm at 8
        0, 3, 0, 3,        # Detection at 11, 13
        2, 0, 3, 0, 0,     # False alarms at 14, 16
        2, 2,              # Detection at 19-20
    ])

    results = compute_arl_per_fault(y_true, y_pred)
    print(results)
```

### Output Interpretation

```
   Fault  ARL0  ARL1
0      1   inf   2.0
1      2  12.0   0.5
2      3   inf   1.0
```

**Fault 1**:
- ARL₀ = ∞ (only one false alarm, can't compute interval)
- ARL₁ = 2.0 (average delay: never detected → penalty of 2)

**Fault 2**:
- ARL₀ = 12.0 (gap between false alarms at t=1 and t=14)
- ARL₁ = 0.5 (average of delays: 1 and 0)

**Fault 3**:
- ARL₀ = ∞ (only one false alarm)
- ARL₁ = 1.0 (detected at first point)

## Monte Carlo Simulation for CUSUM/EWMA

For charts with memory (CUSUM, EWMA), use Monte Carlo:

```python
def simulate_arl1_with_history(
    n_simulations: int,
    fault_start: int,
    detection_method,
    data_generator
) -> Tuple[float, float]:
    """
    Simulate ARL₁ with proper history.

    Args:
        n_simulations: Number of Monte Carlo runs
        fault_start: Time point where fault occurs
        detection_method: Function that returns detection index
        data_generator: Function that generates process data

    Returns:
        (mean_arl1, std_arl1)
    """
    delays = []

    for _ in range(n_simulations):
        # Generate sequence with fault at known time
        data = data_generator(fault_start=fault_start)

        # Run detection method
        detection_time = detection_method(data)

        if detection_time is None:
            delay = len(data) - fault_start  # Penalty
        else:
            delay = detection_time - fault_start

        delays.append(delay)

    return np.mean(delays), np.std(delays)
```

## Key Implementation Notes

1. **Always reset** chart state between simulation runs
2. **Penalize undetected faults** with full segment length
3. **Per-fault metrics** more informative than aggregate
4. **Handle edge cases**: no false alarms, no detections
5. **Use sufficient simulations** (≥1000) for stable estimates

## Literature-Based Method

According to Shams et al. (2010) for TEP:

> *"The out-of-control ARL is estimated by the average run length over multiple simulation runs with fault introduced at a known time."*

> *"If no detection occurs during the simulation run, the ARL for that run is set to the maximum length of the monitoring period."*

This implementation follows these established practices for fair evaluation in process monitoring research.
