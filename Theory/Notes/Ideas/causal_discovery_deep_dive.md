# Causal Discovery for Process Monitoring: Complete Guide

## Why Causal Discovery is Critical

**You are absolutely correct:** If causal discovery fails, the entire hybrid approach collapses. This is the **foundation** that determines whether the GNN learns meaningful relationships or just noise.

---

## What Causal Discovery Actually Does

### Input
- Time series data from your 52 Tennessee Eastman sensors
- Historical normal operation data (e.g., 480 samples × 52 variables)

### Output
- **Causal graph** showing directed edges: X → Y
- **Time delays** (lags): "X affects Y after τ samples"
- **Confidence scores** for each edge

### Goal
Distinguish **true causation** from **spurious correlation**

**Example:**
- ❌ Correlation: "Ice cream sales and drowning both increase in summer"
- ✅ Causation: "Hot weather CAUSES both ice cream sales AND more swimming"

---

## Key Algorithms Behind Causal Discovery

### 1. PCMCI+ (Recommended for Your Work)

**Full name:** PC algorithm with Momentary Conditional Independence

**How it works:**
1. **Step 1 - Condition-Selection:** For each variable pair (X, Y), find relevant conditioning set Z
2. **Step 2 - Momentary CI tests:** Test if X(t-τ) ⊥ Y(t) | Z(t-τ') for various time lags τ
3. **Step 3 - Remove spurious edges:** If conditional independence holds, no causal link exists

**Mathematical foundation:**
```
If X(t-τ) ⊥ Y(t) | Z(t-τ'), then X does not cause Y at lag τ
Where:
  ⊥ = statistical independence
  Z = all other relevant variables (parents of X and Y)
  τ = time lag being tested
```

**Why PCMCI+ for Tennessee Eastman:**
- ✅ **Designed for multivariate time series** (most algorithms aren't)
- ✅ **Handles time delays automatically** (crucial for process dynamics)
- ✅ **Computationally feasible:** O(N²p²T) complexity
  - N = number of variables (52)
  - p = maximum time lag (typically 5-10)
  - T = number of samples (480)
- ✅ **Available in Tigramite library** (production-ready Python code)

**Example output:**
```
Reactor_Temperature(t-2) → Separator_Pressure(t)  [confidence: 0.87]
Feed_Rate(t-1) → Reactor_Level(t)                 [confidence: 0.92]
Compressor_Power(t-3) → Product_Flow(t)           [confidence: 0.78]
```

---

### 2. PC Algorithm (Classical Constraint-Based)

**How it works:**
1. Start with **fully connected graph** (assume everything causes everything)
2. Test pairwise independence: Remove edge if X ⊥ Y
3. Test conditional independence: Remove edge if X ⊥ Y | Z for some Z
4. Orient remaining edges using collider rules

**Advantages:**
- ✅ Theoretically sound (Spirtes-Glymour-Scheines framework)
- ✅ Works well with sufficient data

**Disadvantages:**
- ❌ Assumes **no hidden confounders** (unrealistic for industrial processes)
- ❌ Not designed for time series (treats temporal order as regular conditioning)
- ❌ Requires large sample sizes for reliable independence tests

---

### 3. GES (Greedy Equivalence Search) - Score-Based

**How it works:**
1. Start with empty graph or random structure
2. **Forward phase:** Add edges that improve Bayesian Information Criterion (BIC) score
3. **Backward phase:** Remove edges that improve BIC score
4. Return graph with best score

**Score function:**
```
Score(G) = log P(Data | G) - (k/2) log(n)
Where:
  G = candidate graph structure
  k = number of parameters
  n = sample size
```

**Advantages:**
- ✅ Can find optimal structure (under assumptions)
- ✅ Handles equivalence classes (multiple graphs with same independence structure)

**Disadvantages:**
- ❌ **Computationally expensive** for 52 variables (exponential search space)
- ❌ Assumes causal sufficiency (no hidden variables)
- ❌ Score functions assume specific parametric forms

---

### 4. VAR-LiNGAM (Vector AutoRegressive Linear Non-Gaussian Acyclic Model)

**How it works:**
- Exploits **non-Gaussianity** of data to identify causal direction
- Uses Independent Component Analysis (ICA)
- Specifically for time series data

**Key insight:** 
If X → Y and both are non-Gaussian, the causal direction can be identified from data alone!

**Advantages:**
- ✅ Can determine causal direction without experiments
- ✅ Handles time series naturally

**Disadvantages:**
- ❌ Requires **non-Gaussian distributions** (Tennessee Eastman may be partially Gaussian)
- ❌ Assumes **linear relationships** (processes are often nonlinear)

---

## Comparison Table

| Algorithm | Time Series | Nonlinear | Hidden Variables | Complexity | Implementation |
|-----------|-------------|-----------|------------------|------------|----------------|
| **PCMCI+** | ✅ Excellent | △ Partial | △ Some robustness | O(N²p²T) | Tigramite |
| PC | ❌ Poor | ❌ No | ❌ No | O(N^k) | causal-learn |
| GES | ❌ Poor | ❌ No | ❌ No | Exponential | causal-learn |
| VAR-LiNGAM | ✅ Good | ❌ No | ❌ No | O(N³) | causal-learn |

**Recommendation:** **PCMCI+** is clearly the best choice for Tennessee Eastman Process monitoring.

---

## Major Challenges in Causal Discovery

### Challenge 1: Hidden (Confounding) Variables

**Problem:** Unmeasured variables that affect multiple observed variables

**Example:**
```
True structure:
  Catalyst_Activity (unmeasured) → Reactor_Temp
  Catalyst_Activity (unmeasured) → Product_Yield

Observed structure:
  Reactor_Temp ↔ Product_Yield (spurious correlation!)
```

**Mitigation strategies:**
- Use **FCI algorithm** (Fast Causal Inference) which handles latent confounders
- Include as many relevant process variables as possible
- Use process knowledge to identify potential hidden variables

---

### Challenge 2: Nonlinear Relationships

**Problem:** Most algorithms assume linear causation, but chemical processes are inherently nonlinear

**Example:** Arrhenius equation for reaction rate
```
k = A × exp(-Ea / RT)
Reaction rate is exponentially related to temperature!
```

**Mitigation strategies:**
- Use **kernel-based independence tests** (Gaussian Process regression)
- **Neural network-based** conditional independence testing
- Transform variables (log, sqrt) to linearize relationships

**Advanced option:** Use **PCMCI+ with neural network CI tests** instead of partial correlation

---

### Challenge 3: Validation - How Do We Know It's Correct?

**The fundamental problem:** We can never observe the true causal structure directly!

**Validation approaches:**

#### 1. Process Knowledge Validation
- Compare discovered graph with **engineering knowledge**
- Check if edges make physical sense
- **Example:** If algorithm finds "Product_Flow → Reactor_Temperature" (backward!), it's wrong

#### 2. Intervention Testing (Gold Standard)
- Deliberately change one variable, observe effects
- **Example:** Increase feed rate by 10%, measure which variables respond
- **Problem:** Expensive and risky in industrial settings

#### 3. Cross-Validation with Prediction
- Use discovered graph to build **predictive model**
- Test on held-out data
- If graph is wrong, predictions will be poor

#### 4. Stability Across Different Operating Conditions
- Run causal discovery on multiple datasets
- Check if discovered relationships are **consistent**
- Inconsistency suggests spurious findings

#### 5. Synthetic Data with Known Ground Truth
- Create simulated process with known causal structure
- Test if algorithm recovers true structure
- Build confidence before applying to real data

---

## Practical Implementation Steps

### Step 1: Data Preparation (Critical!)

```python
import pandas as pd
import numpy as np

# Load Tennessee Eastman data
data = pd.read_csv('TEP_normal_operation.csv')  # 480 samples × 52 variables

# Preprocessing pipeline
def preprocess_for_causal_discovery(data):
    # 1. Handle missing values
    data = data.fillna(method='ffill')  # Forward fill
    
    # 2. Remove constant variables (no causal info)
    data = data.loc[:, data.std() > 1e-6]
    
    # 3. Standardize (zero mean, unit variance)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns
    )
    
    # 4. Remove linear trends (detrend)
    from scipy.signal import detrend
    data_detrended = pd.DataFrame(
        detrend(data_scaled, axis=0),
        columns=data.columns
    )
    
    return data_detrended

data_clean = preprocess_for_causal_discovery(data)
```

**Why each step matters:**
- Missing values → bias in independence tests
- Constant variables → cause numerical instability
- Different scales → some variables dominate
- Trends → spurious correlations

---

### Step 2: Run PCMCI+ with Tigramite

```python
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# Convert to Tigramite format
dataframe = pp.DataFrame(
    data_clean.values,
    var_names=data_clean.columns.tolist()
)

# Initialize PCMCI with partial correlation test
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=1
)

# Run causal discovery
results = pcmci.run_pcmciplus(
    tau_min=0,           # Minimum time lag
    tau_max=5,           # Maximum time lag (tune this!)
    pc_alpha=0.01,       # Significance level (lower = stricter)
    contemp_collider_rule='majority',
    conflict_resolution=True
)

# Extract causal graph
causal_graph = results['graph']  # Shape: (52, 52, 6) - [var, var, lag]
p_values = results['p_matrix']   # Statistical significance
```

**Key parameters to tune:**
- `tau_max`: Maximum time lag (5-10 for most processes)
- `pc_alpha`: Significance threshold (0.01 = strict, 0.05 = permissive)
- Independence test: ParCorr (linear) vs. GPDC (nonlinear)

---

### Step 3: Validate Results

```python
def validate_causal_graph(causal_graph, var_names):
    """
    Multiple validation checks
    """
    # 1. Sparsity check (too many edges = likely spurious)
    num_edges = np.sum(causal_graph != '')
    max_possible = len(var_names) ** 2 * 6  # All possible edges
    sparsity = 1 - (num_edges / max_possible)
    print(f"Graph sparsity: {sparsity:.2%}")  # Should be > 90%
    
    # 2. Physical plausibility
    suspicious_edges = []
    for i, var1 in enumerate(var_names):
        for j, var2 in enumerate(var_names):
            if causal_graph[i, j, 0] != '':  # Edge exists
                # Check if edge makes physical sense
                if not is_physically_plausible(var1, var2):
                    suspicious_edges.append((var1, var2))
    
    if suspicious_edges:
        print(f"Warning: {len(suspicious_edges)} suspicious edges found!")
    
    # 3. Cross-validation with prediction
    # Build VAR model using discovered graph
    # Test prediction accuracy on held-out data
    
    return sparsity, suspicious_edges
```

---

### Step 4: Convert to GNN Graph Structure

```python
import torch
from torch_geometric.data import Data

def causal_graph_to_pyg(causal_graph, tau_max=5):
    """
    Convert Tigramite output to PyTorch Geometric format
    """
    num_vars = causal_graph.shape[0]
    edge_list = []
    edge_attrs = []  # Store time lags as edge attributes
    
    for lag in range(tau_max + 1):
        for i in range(num_vars):
            for j in range(num_vars):
                # Check if edge exists at this lag
                if causal_graph[i, j, lag] != '':
                    edge_type = causal_graph[i, j, lag]
                    if '-->' in edge_type or '<--' in edge_type:
                        edge_list.append([i, j])
                        edge_attrs.append(lag)
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Create PyG Data object
    graph_data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_vars
    )
    
    return graph_data
```

---

## Advanced Topics for Your PhD

### 1. Online Causal Discovery

**Challenge:** Causal relationships change over time in industrial processes

**Solution:** Incremental causal discovery
- Use **sliding window approach** (update graph every M samples)
- Employ **change point detection** to identify when to update
- Apply **experience replay** to prevent catastrophic forgetting

```python
def online_causal_discovery(data_stream, window_size=500, update_freq=100):
    """
    Incrementally update causal graph
    """
    causal_graph = None
    buffer = []
    
    for t, new_sample in enumerate(data_stream):
        buffer.append(new_sample)
        
        # Update graph every update_freq samples
        if t % update_freq == 0 and len(buffer) >= window_size:
            # Take most recent window_size samples
            window_data = buffer[-window_size:]
            
            # Run PCMCI+ on window
            new_graph = run_pcmci(window_data)
            
            # Smooth update (blend old and new)
            if causal_graph is not None:
                causal_graph = 0.7 * causal_graph + 0.3 * new_graph
            else:
                causal_graph = new_graph
    
    return causal_graph
```

---

### 2. Handling Nonlinearity

**Use neural network-based conditional independence testing:**

```python
from tigramite.independence_tests.gpdc import GPDC

# Gaussian Process based test (handles nonlinearity)
gpdc = GPDC(significance='analytic', gp_params=None)
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=gpdc,  # Use GPDC instead of ParCorr
    verbosity=1
)
```

**Trade-off:** 
- ParCorr: Fast, assumes linearity
- GPDC: Slow, handles nonlinearity

**Recommendation:** Start with ParCorr for initial exploration, then use GPDC for final graph.

---

### 3. Incorporating Domain Knowledge

**Hybrid approach:** Combine data-driven discovery with engineering knowledge

```python
def constrained_causal_discovery(data, forbidden_edges, required_edges):
    """
    Causal discovery with constraints
    """
    # Run standard PCMCI+
    results = pcmci.run_pcmciplus(...)
    causal_graph = results['graph']
    
    # Apply domain knowledge constraints
    for (i, j) in forbidden_edges:
        # Remove edges that violate physical laws
        causal_graph[i, j, :] = ''
    
    for (i, j, lag) in required_edges:
        # Force edges known from engineering knowledge
        if causal_graph[i, j, lag] == '':
            causal_graph[i, j, lag] = '-->'
    
    return causal_graph

# Example: Engineering knowledge for Tennessee Eastman
forbidden_edges = [
    (product_flow_idx, reactor_temp_idx),  # Product can't affect reactor!
]

required_edges = [
    (reactor_temp_idx, separator_pressure_idx, 2),  # Known 2-sample delay
]
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Not Enough Data
**Problem:** PCMCI+ requires sufficient samples for reliable independence tests  
**Solution:** Minimum 200-500 samples recommended; more is better

### Pitfall 2: Wrong Time Lag Range
**Problem:** Missing important lagged relationships  
**Solution:** Use autocorrelation analysis to determine appropriate `tau_max`

### Pitfall 3: Ignoring Preprocessing
**Problem:** Spurious causations from trends, seasonality  
**Solution:** Always detrend, standardize, and check stationarity

### Pitfall 4: Over-Interpreting Weak Edges
**Problem:** Edges with low confidence may be spurious  
**Solution:** Filter edges by both p-value AND effect size

### Pitfall 5: Assuming Stationarity
**Problem:** Causal structure changes over time  
**Solution:** Use online/adaptive causal discovery methods

---

## Your PhD Contribution Opportunities

### Contribution 1: First Online Causal-GNN for Process Monitoring
**Gap:** No existing work combines incremental causal discovery with GNN-based anomaly detection

### Contribution 2: Uncertainty-Aware Causal Discovery
**Gap:** Causal discovery doesn't quantify uncertainty in discovered edges  
**Your innovation:** Bayesian causal discovery → propagate uncertainty to GNN

### Contribution 3: Multi-Resolution Causal Graphs
**Gap:** Single time scale may miss fast and slow dynamics  
**Your innovation:** Discover causal graphs at multiple time scales (seconds, minutes, hours)

---

## Recommended Next Steps

1. **Week 1:** Install Tigramite, run basic PCMCI+ on synthetic data
2. **Week 2:** Apply to Tennessee Eastman normal operation data
3. **Week 3:** Validate discovered graphs against process knowledge
4. **Week 4:** Implement online causal discovery with sliding windows
5. **Month 2:** Integrate discovered graph with GNN architecture

---

## Key Takeaways

✅ **PCMCI+ is the best algorithm** for Tennessee Eastman time series  
✅ **Validation is critical** - use multiple methods  
✅ **Preprocessing matters** - detrend, standardize, check stationarity  
✅ **Online discovery is essential** - processes change over time  
✅ **Combine with domain knowledge** - don't trust data blindly  

**Bottom line:** Causal discovery is hard, but with PCMCI+, Tigramite library, and careful validation, it's absolutely feasible for your PhD work. This will be the foundation that makes your entire hybrid approach successful!
