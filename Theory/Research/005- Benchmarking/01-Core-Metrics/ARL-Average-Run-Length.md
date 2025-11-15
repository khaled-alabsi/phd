# ARL (Average Run Length)

**ARL (Average Run Length)** is a key performance metric used to evaluate control charts in Statistical Process Control (SPC). It measures the average number of samples taken before a signal is triggered, indicating a potential out-of-control condition.

## Key Concepts

### ARL₀ (In-Control ARL)
- **Definition**: Average number of samples until a false alarm occurs when the process is actually in control
- **Goal**: Higher is better (fewer false alarms)
- **Typical target**: ARL₀ = 370 for standard Shewhart 3-sigma limits
- **Relationship**: ARL₀ ≈ 1 / α (for memoryless charts)

### ARL₁ (Out-of-Control ARL)
- **Definition**: Average number of samples until a signal when the process is out of control
- **Goal**: Lower is better (faster detection)
- **Interpretation**: Detection delay after fault onset
- **Context-dependent**: Value depends on shift magnitude

## Usage in Chart Design

### Design Phase (Before Monitoring)

ARL is primarily a **design and evaluation tool** used before Phase II monitoring begins:

1. **Define Performance Goals**
   - Set target ARL₀ based on cost of false alarms (e.g., 370, 500, 1000)
   - Set target ARL₁ for different shift magnitudes
   - Balance between sensitivity and stability

2. **Compare Methods**
   - Fix ARL₀ to the same value for all methods
   - Calculate ARL₁ for various shift sizes
   - Plot ARL curves to visualize performance

3. **Select Optimal Chart**
   - Choose chart with best ARL₁ for expected shift types
   - Consider trade-offs between small and large shift detection

### Example Comparison

Monitoring bottle fill volume with target ARL₀ = 500:

| Chart Type | ARL₁ (1σ shift) | ARL₁ (2σ shift) | Best For |
|-----------|----------------|----------------|----------|
| Shewhart  | 8              | 2              | Large shifts |
| CUSUM     | 4              | 3              | Small shifts |
| EWMA      | 5              | 2.5            | Balanced |

**Decision**: CUSUM detects 1σ shift fastest, making it optimal for this scenario.

## Theoretical vs Empirical ARL

### Theoretical (Formula-based)
For probability p of signaling:
$$\text{ARL} = \frac{1}{p}$$

**Example**: Shewhart 3σ chart with p = 0.0027
$$\text{ARL}_0 = \frac{1}{0.0027} \approx 370$$

### Empirical (Simulation-based)
1. Run many independent simulations
2. Record run length for each (samples until first alarm)
3. Average all run lengths

**Example**: 1,000 simulations with run lengths [200, 150, 300, ...]
$$\text{Empirical ARL}_0 = \frac{\sum \text{run lengths}}{1000}$$

## Why Use Simulations?

**Required when:**
- Complex charts (CUSUM, EWMA) with memory
- Multivariate methods (Hotelling T², MEWMA)
- Non-standard distributions
- Correlated data

**Proper simulation procedure:**
1. Generate many **independent runs** (e.g., 10,000)
2. For each run, **start fresh** (reset chart state)
3. Record sample number of **first alarm**
4. Average to get empirical ARL

```python
import numpy as np

def simulate_arl0_shewhart(n_simulations=10_000, ucl=3, lcl=-3):
    """Simulate ARL₀ for Shewhart chart."""
    arl_list = []

    for _ in range(n_simulations):
        data = np.random.normal(0, 1, 1000)  # In-control process
        alarm_indices = np.where((data < lcl) | (data > ucl))[0]

        if len(alarm_indices) > 0:
            arl_list.append(alarm_indices[0] + 1)  # First alarm
        else:
            arl_list.append(1000)  # No alarm in sequence

    return np.mean(arl_list)

# Example
empirical_arl0 = simulate_arl0_shewhart()
print(f"Empirical ARL₀: {empirical_arl0:.1f}")  # ~370
```

## Common Pitfalls

### ❌ Don't: Count alarms in one long dataset
```python
# Wrong approach
total_samples = 10_000
total_alarms = 27
arl0 = total_samples / total_alarms  # Not accounting for independence
```

**Problem**: Assumes process resets after each alarm, which doesn't reflect reality.

### ✓ Do: Run independent simulations
```python
# Correct approach
for each simulation:
    reset chart
    run until first alarm
    record run length
arl0 = average of all run lengths
```

## Performance Optimization

### Reducing High ARL₁ (Slow Detection)

| Chart | Problem | Fix | Result |
|-------|---------|-----|--------|
| Shewhart | High ARL₁ for small shifts | Use narrower limits (2σ) | Faster detection, more false alarms |
| CUSUM | Slow for specific shifts | Adjust reference value k | Tuned sensitivity |
| EWMA | Late detection | Increase λ (e.g., 0.3) | More responsive |

### Reducing Low ARL₀ (Too Many False Alarms)

| Chart | Problem | Fix | Result |
|-------|---------|-----|--------|
| Any | Low ARL₀ | Widen control limits | Fewer false alarms, slower detection |
| CUSUM | Too sensitive | Increase threshold h | More stable |
| EWMA | Excessive alarms | Decrease λ | Less reactive |

## Chart-Specific ARL Characteristics

### Shewhart Chart
- **ARL₀**: High (370 for 3σ)
- **ARL₁**: High for small shifts, low for large shifts
- **Best for**: Detecting large, sudden shifts

### CUSUM Chart
- **ARL₀**: Can be set equal to Shewhart
- **ARL₁**: Low for small persistent shifts
- **Best for**: Detecting small, sustained changes

### EWMA Chart
- **ARL₀**: Moderate (tunable via λ and limits)
- **ARL₁**: Low for small to moderate shifts
- **Best for**: Trends and gradual changes

## Conditional Metrics (Simulation-Based)

When simulations have finite length, some runs may not trigger alarms:

### Conditional ARL₀
- Average run length **only over runs that had false alarms**
- Excludes runs that "ran out" without alarming
- More accurate than including runs with no alarms

### Non-FA Fraction (Non-False Alarm Fraction)
- Proportion of runs with **no false alarms**
- High value (>0.5) → many runs exhausted without alarming
- Indicates conditional ARL₀ may be based on small sample

### Example
10 simulation runs, sequence length = 500:
- 7 runs: first alarm at [200, 150, 300, 250, 180, 220, 190]
- 3 runs: no alarm (None)

**Conditional ARL₀** = (200+150+300+250+180+220+190)/7 ≈ **212**
**Non-FA Fraction** = 3/10 = **0.3** (30%)

If target ARL₀ = 370, the 30% non-FA fraction is expected: exp(-500/370) ≈ 0.26

## Key Takeaways

1. **ARL is a design metric**, not a real-time monitoring metric
2. **Always specify ARL₀** when comparing methods
3. **ARL₁ is shift-specific** – report for relevant shift magnitudes
4. **Simulation is essential** for complex/multivariate charts
5. **Balance is critical**: high ARL₀, low ARL₁
6. **Context matters**: ARL values depend on shift type, magnitude, and direction
