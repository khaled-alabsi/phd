# Multivariate Relationship Discovery System

## Research Context

### PhD Project Overview

This project is part of PhD research in **multivariate control charts** for industrial process monitoring, specifically focusing on the **Tennessee Eastman Process** - a benchmark chemical process simulation widely used in fault detection and diagnosis research.

### Problem Statement

Traditional multivariate control charts (PCA-based, PLS-based, Hotelling’s T², MEWMA) are effective at detecting when a process goes out of control, but they:

- Often fail to explain **which variables are involved** in the fault
- Don’t reveal the **temporal relationships** between variables
- Require complex statistical assumptions
- Are difficult to interpret for process engineers

### Used data: The Tennessee Eastman Process - Compact Data Summary

#### Process Overview
The **Tennessee Eastman Process (TEP)** is a benchmark industrial chemical process widely used for testing fault detection and diagnosis methods. It simulates a realistic chemical plant with multiple unit operations.

#### Dataset Structure

##### Features (52 total)
- **Process Measurements (xmeas_1 to xmeas_41)**: 41 sensor measurements
  - Flow rates, temperatures, pressures, compositions, etc.
- **Manipulated Variables (xmv_1 to xmv_11)**: 11 control inputs
  - Valve positions, setpoints, flow controls

##### Labels (Fault Numbers)
- **0**: Fault-free operation (normal/in-control)
- **1-20**: Different fault types representing various process abnormalities
  - Examples: Step changes, random variations, slow drifts, sticking valves, etc.

#### Data Splits

| Dataset | Description | Samples | Purpose |
|---------|-------------|---------|---------|
| **Training (In-Control)** | Normal operation only | 500 per simulation | Learn normal behavior |
| **Training (Out-of-Control)** | Faulty operation | 500 per simulation | Learn fault patterns |
| **Testing (In-Control)** | Normal operation only | 960 per simulation | Validate normal detection |
| **Testing (Out-of-Control)** | Faulty operation | 960 per simulation | Validate fault detection |

#### Key Characteristics
- **Multivariate**: 52 correlated variables
- **Dynamic**: Time-series data with temporal dependencies
- **Realistic**: Based on actual industrial process
- **Challenging**: Complex fault signatures, variable interactions
- **Preprocessed**: All features standardized (zero mean, unit variance)

#### Common Research Tasks
1. **Fault Detection**: Identify when faults occur (binary classification)
2. **Fault Diagnosis**: Classify which specific fault occurred (multi-class)
3. **Anomaly Detection**: Detect abnormal behavior without fault labels
4. **Root Cause Analysis**: Identify which variables caused the fault
5. **Predictive Maintenance**: Predict faults before they occur



### Research Contribution

This work proposes a **novel, interpretable approach** to discover relationships between process variables by:

1. Using **distributional boundaries** instead of complex statistical models
1. Tracking **co-occurrence of boundary violations** (when variables simultaneously exceed normal ranges)
1. Detecting **time-lagged dependencies** (when one variable’s change precedes another’s)
1. Applying **hierarchical boundary refinement** to capture relationships at multiple scales
1. Enabling **online anomaly detection** by identifying when learned relationships break down

### Key Innovation

Unlike existing methods (Granger causality, mutual information, correlation-based approaches), this method is:

- **Simple and interpretable**: Based on intuitive boundary violations
- **Sensitivity-aware**: Analyzes ALL variables at multiple sensitivity levels (large, medium, sensitive) without elimination
- **Comprehensive**: Preserves weak/subtle relationships that might be early warning signals
- **Lag-explicit**: Reports relationships organized by sensitivity level AND time delay
- **Actionable**: Directly applicable to real-time fault diagnosis with risk stratification

### Application Domain

**Tennessee Eastman Process:**

- 52 variables (temperatures, pressures, flow rates, levels, compositions, compressor power, valve positions)
- Complex chemical process with strong variable interactions
- Contains 20+ different fault types
- Sampled every 3 minutes over 25-48 hour periods

**Goal:** Develop a system that not only detects faults but explains them through relationship analysis.

-----

## Core Methodology

### Phase 1: Define Normal Operating Boundaries

**For each variable:**

1. **Multiple Boundaries per Sensitivity Level** (NEW APPROACH):
   - Each sensitivity level (Sensitive, Medium, Large) has **N boundary regions** (default N=3)
   - Supports **multimodal distributions** and **multiple operating regimes**
   - Example: A valve variable with 3 common positions gets 3 boundaries around each position

2. **Quantile-Based Boundary Widths**:
   - **Level 1 (Sensitive/Narrow):** 25th percentile per boundary (captures subtle variations)
   - **Level 2 (Medium):** 40th percentile per boundary (captures moderate variations)
   - **Level 3 (Large/Wide):** 60th percentile per boundary (captures major variations)

3. **Clustering Methods for Boundary Locations**:
   - **K-Means Clustering:** Detects natural clusters in data (best for true multi-modal distributions)
   - **Histogram Peaks:** Finds peaks in data distribution (best for clear modes)
   - **Quantile Positions:** Evenly spaced boundaries (simple, always works)
   - All methods are implemented and can be compared via visualization

4. **Boundary Calculation Architecture:**
   - Extract boundary calculation logic into **separate, dedicated functions** for each sensitivity level
   - Each function calculates N boundary regions with lower/upper bounds based on quantile
   - Around each cluster center, calculate local quantile-based boundaries
   - Functions include TODO comments for future enhancements:
     - Dynamic boundary calculation based on current timestamp
     - Adaptive boundaries using smoothing functions (e.g., moving average, exponential smoothing)
     - Context-aware boundaries that adjust based on process state
     - Variable boundary widths per cluster based on local density
     - Automatic merging/splitting of boundaries based on data drift

5. **Violation Detection Logic** (CRITICAL):
   - A datapoint is marked as **violation ONLY if it's outside ALL N boundaries**
   - If datapoint is inside at least ONE boundary → Normal (0)
   - If outside ALL boundaries and above mean → Violation above (+1)
   - If outside ALL boundaries and below mean → Violation below (-1)
   - This conservative approach reduces false alarms for multimodal data

6. **Visualizations**:
   - **Distribution view**: Histogram with all N boundaries overlaid
   - **Timeline view**: Time series with all N boundary regions shown
   - **Clustering view (NEW)**: Scatter plot with:
     - Data points colored by cluster membership
     - Circles/bands around each boundary region
     - Red 'X' markers for violations (outside all boundaries)
     - Method comparison view (K-means vs Histogram Peaks vs Quantile Positions)

7. User adjustments: Widths and locations adjustable per variable via configuration

**Purpose:** Handle multimodal distributions and multiple operating regimes. Only flag violations when datapoint doesn't belong to ANY normal cluster. Quantile-based boundaries are robust to outliers and provide consistent coverage.

-----

### Phase 2: Detect Boundary Violations

**For each timestep and variable:**

1. **Multi-Boundary Violation Logic**:
   - Check if value is inside ANY of the N boundaries
   - Mark as **"normal" (0)** if inside at least one boundary
   - Mark as **"violation"** ONLY if outside ALL N boundaries:
     - **+1** if outside all boundaries AND value > mean
     - **-1** if outside all boundaries AND value < mean
   - TODO: Consider alternative classification methods (median, distance-based, cluster-specific)

2. **Modes**:
   - **Ternary mode** (-1, 0, +1): Directional violations for co-occurrence analysis
   - **Binary mode** (0, 1): Simple violation detection

3. **Visualizations**:
   - **Excursion timeline**: Shows violations over time with all N boundaries displayed
   - **Multi-variable heatmap**: Matrix view of violations across variables
   - **Clustering view**: NEW - Shows which cluster each point belongs to, highlights violations

4. **Interactive widgets**: Select variable(s), boundary level, and visualization type

**Result:** Violation matrix for each variable showing when it's outside ALL boundary regions.

**Key Insight:** With N=3 boundaries, a variable must be far from ALL 3 normal clusters to be flagged as a violation. This is more conservative and reduces false alarms for multimodal distributions.

-----

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

-----

### Phase 4: Hierarchical Refinement

**Process:**

1. Run Phase 2 & 3 with Level 1 boundaries (large) → Results_L1
2. Run Phase 2 & 3 with Level 2 boundaries (medium)  → Results_L2
3. Run Phase 2 & 3 with Level 3 boundaries (sensitive) → Results_L3

**CRITICAL: Analyze ALL variables at ALL levels - no elimination.**

**Purpose:**

- Different sensitivity levels reveal different types of relationships
- **Large boundaries (2σ):** Detect robust relationships visible during major disturbances
- **Medium boundaries (1.5σ):** Detect moderate relationships
- **Sensitive boundaries (1σ):** Detect subtle relationships and early warning signals
- Preserves complete relationship landscape without information loss

-----

### Phase 5: Comprehensive Reporting (No Filtering)

**DO NOT filter or eliminate variables.** Report relationships for ALL variables at ALL sensitivity levels.

1. **Organize results by Sensitivity Level × Time Lag**
- For each sensitivity level (Large, Medium, Sensitive)
- For each time lag (0 to max_lag)
- List all variable pairs with their co-occurrence counts
1. **Characterize relationship types:**
- **Robust relationships:** High co-occurrence at large boundary level (visible even during major disturbances)
- **Moderate relationships:** Visible at medium level but not large
- **Sensitive relationships:** Only detected at sensitive level (subtle coordinated changes)
1. **Build relationship network showing:**
- Which variables are connected at each sensitivity level
- Strength of connection (co-occurrence count) at each level
- Time lag (if lagged relationship)
- How relationship strength changes across sensitivity levels

-----

## Configuration System

All analysis parameters should be configurable:

**Data Processing:**

- Normalization method (zscore, minmax, or robust)

**Sensitivity Levels (Quantile-Based with Multiple Boundaries):**

- **Sensitive boundary quantile (default: 0.25)** - Narrow boundaries capturing 25% of data range per cluster
- **Medium boundary quantile (default: 0.40)** - Medium boundaries capturing 40% of data range per cluster
- **Large boundary quantile (default: 0.60)** - Wide boundaries capturing 60% of data range per cluster
- **Number of boundaries per level (default: 3)** - N boundary regions to handle multimodal distributions
- Can add more levels or adjust N per level if needed
- Quantile approach ensures consistent coverage across variables with different distributions

**Boundary Calculation:**

- Each sensitivity level has a dedicated calculation function
- Each function calculates **N boundary regions** (not just one)
- Three clustering methods available:
  - **K-means** - Detects natural clusters
  - **Histogram peaks** - Finds distribution peaks
  - **Quantile positions** - Evenly distributed
- Functions are extensible for future dynamic boundary logic
- Support for user adjustments via offset parameters
- Can compare methods via visualization

**Time Lag Analysis:**

- Minimum lag (default: 0)
- Maximum lag (default: 10)
- Lag step (default: 1) - or specify exact lag points to check

**Reporting Thresholds (Optional):**

- Minimum co-occurrence for inclusion in report (default: 1 - include everything)
- Set to higher value only if dealing with very large datasets and need to reduce report size
- **NEVER eliminate variables** - may set minimum count for highlighting in visualizations

**Important:** The philosophy is comprehensive analysis - analyze ALL variables at ALL sensitivity levels.

-----

## Lag-Based Relationship Report

After analysis completes, generate a comprehensive report organized by sensitivity level and time lag:

### Report Section 1: Relationships by Sensitivity Level and Lag

**For each sensitivity level (Sensitive, Medium, Large):**
**For each lag value (0 to max_lag), list:**

- All variable pairs at that sensitivity and lag
- Co-occurrence count
- Relationship strength score

**Example structure:**

```
SENSITIVITY LEVEL: SENSITIVE (1σ - Narrow Boundaries)
══════════════════════════════════════════════════════

LAG 0 (Simultaneous):
- Temperature ↔ Pressure: count=68, strength=0.92
- Pressure ↔ CompPower: count=65, strength=0.89
- ReactorTemp ↔ AgitatorSpeed: count=35, strength=0.61
Total: 15 relationships

LAG 2 (2-timestep delay):
- Temperature → ReactorTemp: count=52, strength=0.84
Total: 3 relationships

LAG 5 (5-timestep delay):
- FlowRate → Level: count=58, strength=0.93
Total: 2 relationships


SENSITIVITY LEVEL: MEDIUM (1.5σ)
═════════════════════════════════

LAG 0 (Simultaneous):
- Temperature ↔ Pressure: count=52, strength=0.89
- Pressure ↔ CompPower: count=48, strength=0.85
Total: 8 relationships

LAG 5 (5-timestep delay):
- FlowRate → Level: count=45, strength=0.89
Total: 1 relationship


SENSITIVITY LEVEL: LARGE (2σ - Wide Boundaries)
═══════════════════════════════════════════════

LAG 0 (Simultaneous):
- Temperature ↔ Pressure: count=28, strength=0.82
- Pressure ↔ CompPower: count=25, strength=0.78
Total: 3 relationships

LAG 5 (5-timestep delay):
- FlowRate → Level: count=32, strength=0.85
Total: 1 relationship
```

### Report Section 2: Summary Statistics

**Create a summary table:**

|Sensitivity Level|Total Relationships|Avg Strength Score|Dominant Lag Pattern|
|-----------------|-------------------|------------------|--------------------|
|Sensitive (1σ)   |47                 |0.68              |Lag 0 most common   |
|Medium (1.5σ)    |18                 |0.76              |Lag 0 most common   |
|Large (2σ)       |8                  |0.84              |Lag 0, 5 equally    |

**Interpretation guide:**

- Relationships appearing at ALL levels = robust (reliable for critical alarms)
- Relationships only at sensitive level = early warning signals
- More relationships at sensitive level = more subtle coordinated behaviors

### Report Section 3: Variable-Centric Report

**For each variable, report at EACH sensitivity level:**

- Which variables it influences (outgoing relationships)
- Which variables influence it (incoming relationships)
- At what lags these relationships occur
- How relationship counts change across sensitivity levels

**Example:**

```
VARIABLE: Temperature
─────────────────────

Sensitivity: SENSITIVE (1σ)
  Influences (outgoing):
    → Pressure (lag 0, strength 0.92)
    → Pressure (lag 1, strength 0.38)
    → ReactorTemp (lag 2, strength 0.84)
  Influenced By (incoming):
    ← Pressure (lag 0, strength 0.92)
  
Sensitivity: MEDIUM (1.5σ)
  Influences (outgoing):
    → Pressure (lag 0, strength 0.89)
    → ReactorTemp (lag 2, strength 0.76)
  Influenced By (incoming):
    ← Pressure (lag 0, strength 0.89)

Sensitivity: LARGE (2σ)
  Influences (outgoing):
    → Pressure (lag 0, strength 0.82)
  Influenced By (incoming):
    ← Pressure (lag 0, strength 0.82)

Relationship Robustness: HIGH (appears at all sensitivity levels)
```

### Report Section 4: Relationship Classification

**Categorize each discovered relationship:**

1. **Robust Relationships** (appear at all 3 sensitivity levels)
- These are reliable for critical alarm triggers
- Example: Temperature ↔ Pressure
1. **Moderate Relationships** (appear at sensitive + medium, but not large)
- Useful for intermediate monitoring
1. **Sensitive Relationships** (only appear at sensitive level)
- Early warning signals
- May indicate incipient faults before major disturbances
- Example: Subtle coordinated drifts

**NO relationships are filtered out or eliminated.**

-----

## Key Visualizations

### 1. Individual Variable Timelines

**For each variable:**

- Timeline showing when it violates boundaries (spikes/bars)
- Separate plots for each boundary level

### 2. Combined Multi-Variable Timeline

**Stack all variables vertically:**

- Shared time axis
- Each row shows one variable’s boundary violations
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

-----

## Online Anomaly Detection

### Training Phase (Offline Analysis)

**Use fault-free historical data to:**

1. Learn normal boundaries for each variable
1. Discover expected relationships between variables
1. Record baseline co-occurrence counts for each relationship
1. Identify dominant lags for each variable pair
1. Save this as the “normal behavior model”

-----

### Monitoring Phase (Online Detection)

**Sliding window approach:**

**At each new timestep:**

1. Keep a window of recent data (e.g., last 50 timesteps)
1. Run the relationship analysis on this window
1. Compare discovered relationships to baseline

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

-----

### Anomaly Scoring System

**Calculate an anomaly score combining:**

1. **Relationship deviation score:** How much do current co-occurrence counts differ from baseline?
1. **Network structure score:** Are expected connections present?
1. **Lag deviation score:** Are relationships happening at expected time delays?
1. **Excursion frequency score:** Are variables violating boundaries more than normal?

**Alert levels:**

- Score < 0.3: Normal operation
- Score 0.3-0.6: Minor deviation - monitor
- Score 0.6-0.8: Moderate anomaly - investigate
- Score > 0.8: Severe anomaly - immediate action

-----

## Synthetic Test Data

Generate synthetic data (10 variables, 300 timesteps) with these **known relationships** embedded:

1. **Temperature ↔ Pressure** (simultaneous, strong - should appear at ALL sensitivity levels)
1. **Temperature → ReactorTemp** (2 timestep lag, strong - should appear at all levels)
1. **Pressure ↔ CompressorPower** (simultaneous, strong - should appear at all levels)
1. **FlowRate → Level** (5 timestep lag, strong - should appear at all levels)
1. **ReactorTemp → CoolingFlow** (3 timestep lag, moderate - appears at sensitive & medium)
1. **ReactorTemp ↔ AgitatorSpeed** (simultaneous, weak - appears only at sensitive level)
1. **Independent, Concentration** (minimal coordinated behavior - weak relationships only)

Add cyclical patterns, noise, and step changes (faults at timestep 100-200).

**Key testing points:**

- Strong relationships should be detected at large, medium, AND sensitive boundaries
- Moderate relationships should appear at medium and sensitive, but not large
- Weak relationships should only appear at sensitive boundaries
- Verify relationship counts decrease as sensitivity decreases (most at sensitive, least at large)

**For online testing:** Generate sequences with normal operation and fault scenarios affecting relationships at different sensitivities.

-----

## Validation Criteria

**Offline Analysis:**

1. ✅ Detects all 6 known relationships in synthetic data at appropriate sensitivity levels
1. ✅ Strong relationships (Temperature ↔ Pressure) appear at ALL sensitivity levels
1. ✅ Weak relationships (ReactorTemp ↔ AgitatorSpeed) appear only at sensitive level
1. ✅ Finds correct time lags at each sensitivity level
1. ✅ Shows decreasing relationship counts as sensitivity decreases (sensitive→medium→large)
1. ✅ Generates comprehensive lag-based report organized by sensitivity level
1. ✅ Visualizations show variables spiking together at different sensitivity thresholds
1. ✅ NO variables are eliminated - all appear in final report

**Online Monitoring:**

1. ✅ Low anomaly scores during normal operation at all sensitivity levels
1. ✅ Triggers alerts when faults occur, with sensitivity-appropriate thresholds
1. ✅ Identifies which relationships broke down and at which sensitivity level
1. ✅ Provides early warnings through sensitive-level relationship changes
1. ✅ Real-time performance (< 1 second per timestep)

-----

## Expected Deliverables

**Offline Phase:**

- All visualizations (timelines at each sensitivity, heatmaps, network graphs)
- Comprehensive lag-based relationship report organized by sensitivity level
- Relationship classification (robust, moderate, sensitive)
- Trained model for online phase with multi-sensitivity baselines

**Online Phase:**

- Multi-level anomaly scoring (separate scores for sensitive, medium, large)
- Tiered alert system:
  - Sensitive level alerts for early warnings
  - Large level alerts for critical situations
- Diagnostic information showing which relationships changed at which sensitivity
- Updated visualizations showing current vs. expected behavior at each level

-----

**This system provides a complete multi-sensitivity relationship analysis framework: comprehensive offline discovery preserving all relationships, then sensitivity-aware online monitoring for both early warning and critical fault detection.**

-----

## Key Design Philosophy

**Comprehensive Analysis Without Filtering:**

Unlike traditional approaches that start with wide boundaries and filter out weak relationships, this method:

- ✅ Analyzes ALL variables at ALL three sensitivity levels (sensitive, medium, large)
- ✅ Preserves the complete relationship landscape
- ✅ Reports everything - no elimination or filtering
- ✅ Enables distinction between robust (visible at all levels) and subtle (only at sensitive level) relationships
- ✅ Provides risk stratification: use robust relationships for critical alarms, sensitive relationships for early warnings

**Goal:** Capture the full spectrum of relationships from major coordinated disturbances to subtle synchronized variations, organized by sensitivity level and time lag.