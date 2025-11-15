# Multivariate Relationship Discovery System - Implementation Prompt

## Research Context

### PhD Project Overview

This project is part of PhD research in **multivariate control charts** for industrial process monitoring, specifically focusing on the **Tennessee Eastman Process** - a benchmark chemical process simulation widely used in fault detection and diagnosis research.

### Problem Statement

Traditional multivariate control charts (PCA-based, PLS-based, Hotelling's T², MEWMA) are effective at detecting when a process goes out of control, but they:
- Often fail to explain **which variables are involved** in the fault
- Don't reveal the **temporal relationships** between variables
- Require complex statistical assumptions
- Are difficult to interpret for process engineers

### Research Contribution

This work proposes a **novel, interpretable approach** to discover relationships between process variables by:
1. Using **distributional boundaries** instead of complex statistical models
2. Tracking **co-occurrence of boundary violations** (when variables simultaneously exceed normal ranges)
3. Detecting **time-lagged dependencies** (when one variable's change precedes another's)
4. Applying **hierarchical boundary refinement** to capture relationships at multiple scales
5. Enabling **online anomaly detection** by identifying when learned relationships break down

### Key Innovation

Unlike existing methods (Granger causality, mutual information, correlation-based approaches), this method is:
- **Simple and interpretable**: Based on intuitive boundary violations
- **Scale-aware**: Detects both large and subtle coordinated changes
- **Lag-explicit**: Reports relationships organized by time delay
- **Actionable**: Directly applicable to real-time fault diagnosis

### Application Domain

**Tennessee Eastman Process:**
- 52 variables (temperatures, pressures, flow rates, levels, compositions, compressor power, valve positions)
- Complex chemical process with strong variable interactions
- Contains 20+ different fault types
- Sampled every 3 minutes over 25-48 hour periods

**Goal:** Develop a system that not only detects faults but explains them through relationship analysis.

---

## Core Methodology

### Phase 1: Define Normal Operating Boundaries

**For each variable:**
1. Calculate mean and standard deviation
2. Define boundaries at multiple levels (hierarchical refinement):
   - **Level 1 (Wide):** mean ± 2.0σ  
   - **Level 2 (Medium):** mean ± 1.5σ  
   - **Level 3 (Narrow):** mean ± 1.0σ

**Purpose:** Progressively tighten boundaries to detect both large and subtle changes.

---

### Phase 2: Detect Boundary Violations

**For each timestep and variable:**
- Mark as **"in excursion"** (1) if value is outside boundaries
- Mark as **"normal"** (0) if value is inside boundaries

**Result:** Binary timeline for each variable showing when it violates boundaries.

---

### Phase 3: Co-occurrence Analysis

**Simultaneous Changes (0-lag):**
- Count how many times two variables are BOTH in excursion at the same timestep
- Build co-occurrence matrix: `C[i,j]` = count of simultaneous boundary violations

**Lagged Changes (time-delayed):**
- For each variable pair and each lag τ (0 to max_lag):
  - Count times when variable i violates boundary at time t
  - AND variable j violates boundary at time t+τ
- Find dominant lag: the time offset with maximum co-occurrence

**Purpose:** Discover which variables change together (simultaneously or with delay).

---

### Phase 4: Hierarchical Refinement

**Process:**
1. Run Phase 2 & 3 with Level 1 boundaries (wide) → Results_L1
2. Run Phase 2 & 3 with Level 2 boundaries (medium) → Results_L2  
3. Run Phase 2 & 3 with Level 3 boundaries (narrow) → Results_L3
4. Compare results across levels to see how relationships emerge at different scales

**Purpose:** Some relationships only appear when looking at subtle changes (narrow boundaries), while others are obvious at all scales.

---

### Phase 5: Filter and Report Relationships

1. Keep only variable pairs with co-occurrence count > minimum threshold (e.g., 5)
2. Identify isolated variables (no relationships to any other variable)
3. Build relationship network showing:
   - Which variables are connected
   - Strength of connection (co-occurrence count)
   - Time lag (if lagged relationship)

---

## Configuration System

All analysis parameters should be configurable:

**Data Processing:**
- Normalization method (zscore, minmax, or robust)

**Boundary Levels:**
- Sigma values for each level (default: 2.0, 1.5, 1.0)
- Can add more levels if needed

**Time Lag Analysis:**
- Minimum lag (default: 0)
- Maximum lag (default: 10)
- Lag step (default: 1) - or specify exact lag points to check

**Filtering Thresholds:**
- Minimum co-occurrence count (default: 5)
- Minimum relationship strength (default: 0.3)

---

## Lag-Based Relationship Report

After analysis completes, generate a comprehensive report organized by time lag:

### Report Section 1: Relationships by Lag

**For each lag value (0 to max_lag), list:**
- All variable pairs found with that lag
- Co-occurrence count
- Relationship strength score
- Which boundary level(s) detected it

**Example structure:**

```
LAG 0 (Simultaneous):
- Temperature ↔ Pressure: count=45, strength=0.89
- Pressure ↔ CompPower: count=42, strength=0.85
Total: 2 relationships

LAG 2 (2-timestep delay):
- Temperature → ReactorTemp: count=38, strength=0.76
Total: 1 relationship

LAG 5 (5-timestep delay):
- FlowRate → Level: count=51, strength=0.91
Total: 1 relationship
```

### Report Section 2: Summary Statistics by Lag

**Create a summary table showing:**
- For each lag: number of relationships found
- Average strength score at that lag
- Strongest relationship at that lag

**Purpose:** Identify which time lags are most common in your process.

### Report Section 3: Variable-Centric Report

**For each variable, report:**
- Which variables it influences (outgoing relationships)
- Which variables influence it (incoming relationships)
- At what lags these relationships occur
- Most common output lag for that variable

### Report Section 4: Hierarchical Comparison

**Show how relationship counts change across boundary levels:**
- Which relationships appear at all levels (robust, strong relationships)
- Which only appear at narrow boundaries (subtle relationships)

---

## Key Visualizations

### 1. Individual Variable Timelines
**For each variable:**
- Timeline showing when it violates boundaries (spikes/bars)
- Separate plots for each boundary level

### 2. Combined Multi-Variable Timeline
**Stack all variables vertically:**
- Shared time axis
- Each row shows one variable's boundary violations
- Color-coded: Red = excursion, Green = normal
- **Purpose:** Visually identify simultaneous spikes across variables

### 3. Co-occurrence Heatmap
**Variables × Variables matrix:**
- Color intensity = co-occurrence count
- One heatmap per boundary level

### 4. Relationship Network Graph
**Network diagram:**
- Nodes = variables (size by connectivity)
- Edges = relationships (thickness by strength)
- Edge labels = dominant lag
- Highlight isolated variables

### 5. Lag Distribution Charts
- Bar chart: relationships found at each lag value
- For specific pairs: co-occurrence across all tested lags

---

## Online Anomaly Detection

### Training Phase (Offline Analysis)

**Use fault-free historical data to:**
1. Learn normal boundaries for each variable
2. Discover expected relationships between variables
3. Record baseline co-occurrence counts for each relationship
4. Identify dominant lags for each variable pair
5. Save this as the "normal behavior model"

---

### Monitoring Phase (Online Detection)

**Sliding window approach:**

**At each new timestep:**
1. Keep a window of recent data (e.g., last 50 timesteps)
2. Run the relationship analysis on this window
3. Compare discovered relationships to baseline

**Detect anomalies when:**

**Type 1: Relationship Breakdown**
- Expected strong relationship has much lower co-occurrence than baseline
- Indicates one variable behaving abnormally

**Type 2: New Unexpected Relationships**
- Variable pairs that were isolated during training now show co-occurrence
- May indicate fault propagation

**Type 3: Lag Changes**
- Relationship exists but dominant lag has shifted
- Indicates process dynamics have changed

**Type 4: Boundary Violations**
- Variable exits boundaries more frequently than during training

---

### Anomaly Scoring System

**Calculate an anomaly score combining:**
1. **Relationship deviation score:** How much do current co-occurrence counts differ from baseline?
2. **Network structure score:** Are expected connections present?
3. **Lag deviation score:** Are relationships happening at expected time delays?
4. **Excursion frequency score:** Are variables violating boundaries more than normal?

**Alert levels:**
- Score < 0.3: Normal operation
- Score 0.3-0.6: Minor deviation - monitor
- Score 0.6-0.8: Moderate anomaly - investigate
- Score > 0.8: Severe anomaly - immediate action

---

## Synthetic Test Data

Generate synthetic data (10 variables, 300 timesteps) with these **known relationships**:

1. **Temperature ↔ Pressure** (simultaneous, strong)
2. **Temperature → ReactorTemp** (2 timestep lag)
3. **Pressure ↔ CompressorPower** (simultaneous, strong)
4. **FlowRate → Level** (5 timestep lag)
5. **ReactorTemp → CoolingFlow** (3 timestep lag)
6. **ReactorTemp ↔ AgitatorSpeed** (simultaneous, weak)
7. **Independent, Concentration** (no relationships - isolated)

Add cyclical patterns, noise, and step changes (faults at timestep 100-200).

**For online testing:** Generate sequences with normal operation and fault scenarios.

---

## Validation Criteria

**Offline Analysis:**
1. ✅ Detects all 6 known relationships in synthetic data
2. ✅ Correctly identifies isolated variables
3. ✅ Finds correct time lags
4. ✅ Shows different results at different boundary levels
5. ✅ Generates clear lag-based report
6. ✅ Visualizations show variables spiking together

**Online Monitoring:**
1. ✅ Low anomaly scores during normal operation
2. ✅ Triggers alerts when faults occur
3. ✅ Identifies which relationship broke down
4. ✅ Real-time performance (< 1 second per timestep)

---

## Expected Deliverables

**Offline Phase:**
- All visualizations (timelines, heatmaps, network graphs)
- Lag-based relationship report
- Trained model for online phase

**Online Phase:**
- Real-time anomaly scoring
- Alert system with diagnostic information
- Updated visualizations showing current vs. expected behavior

---

**This system aims to bridge the gap between detecting that something is wrong (traditional control charts) and understanding what is wrong (relationship-based diagnosis), making it a valuable contribution to multivariate process monitoring research.**