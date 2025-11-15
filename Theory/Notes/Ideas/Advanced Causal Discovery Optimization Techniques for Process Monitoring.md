# Advanced Causal Discovery Optimization Techniques for Process Monitoring

## The Optimization Landscape (2024-2025 Breakthrough Results)

**Critical insight from recent research:** Standard PCMCI suffers from three key bottlenecks that have been solved by cutting-edge optimization methods:

1. **Computational complexity** with large variable sets (Tennessee Eastman: 52 variables)
2. **Spurious relationships** from irrelevant/noisy variables
3. **Static assumptions** that ignore time-varying causal structures

## 🚀 Tier 1 Optimizations (Immediate Implementation)

### 1. F-PCMCI: Feature Selection + Transfer Entropy

**Breakthrough**: Castri et al. (2023, CLeaR) - **Faster AND more accurate** than standard PCMCI

**How it works:**
```
STAGE 1: Feature Selection (Transfer Entropy)
- Compute Transfer Entropy between all variable pairs
- Rank variables by causal influence strength  
- Remove variables below threshold (noise reduction)

STAGE 2: Standard PCMCI on Selected Features
- Run PCMCI+ only on meaningful variables
- Dramatically reduced computational cost
- Higher accuracy due to noise removal
```

**Key advantages:**
- ✅ **Removes spurious variables** (X₆ noise variables automatically eliminated)
- ✅ **Faster execution** (fewer variables = polynomial speedup)
- ✅ **Higher accuracy** (no interference from irrelevant variables)
- ✅ **Production ready** (GitHub: lcastri/fpcmci, Python package available)

**Tennessee Eastman application:**
```python
from fpcmci import FPCMCI
from tigramite.independence_tests.parcorr import ParCorr

# Initialize F-PCMCI
fpcmci = FPCMCI(
    dataframe=dataframe,
    cond_ind_test=ParCorr(),
    f_alpha=0.05,        # Feature selection threshold
    pcmci_alpha=0.01,    # PCMCI significance level
    verbosity=1
)

# Run optimized causal discovery
results = fpcmci.run_fpcmci(
    tau_min=0,
    tau_max=5,
    max_cond_px=None
)

# Extract results
selected_features = results['selected_variables']  # Reduced variable set
causal_graph = results['causal_matrix']           # Optimized graph
feature_scores = results['feature_scores']        # Variable importance
```

**Expected speedup**: 2-5x faster with 10-30% accuracy improvement

---

### 2. Mapped-PCMCI: Dimensionality Reduction

**Innovation**: Tibau et al. (2021) - **Orders of magnitude speedup** for large systems

**Core idea:** 
```
Instead of: 52 variables → 52² × τ_max = 13,520 possible edges
Use: 52 variables → 10 components → 100 × τ_max = 500 possible edges
```

**Implementation strategy:**
```python
def mapped_pcmci_optimization(data, n_components=10):
    """
    Dimensionality reduction before causal discovery
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import UMAP
    
    # Step 1: Reduce to lower-dimensional space
    reducer = PCA(n_components=n_components)  # or UMAP for nonlinear
    data_reduced = reducer.fit_transform(data)
    
    # Step 2: Run PCMCI in reduced space
    pcmci_results = run_pcmci_on_reduced_data(data_reduced)
    
    # Step 3: Map causal relationships back to original variables
    original_graph = map_back_to_original_space(
        pcmci_results, reducer.components_
    )
    
    return original_graph, reducer
```

**Trade-offs:**
- ✅ **Massive speedup** (O(n³) → O(k³) where k << n)
- ✅ **Handles collinearity** (PCA removes redundant variables)
- ❌ **Interpretability loss** (components vs. individual sensors)
- ❌ **May miss sparse relationships** (dimensionality reduction artifacts)

**Best for:** Initial exploration phase, then refine with full-space analysis

---

### 3. Bagged-PCMCI+: Stability & Confidence

**Enhancement**: Debeire et al. (2023) - **Robust causal discovery with uncertainty**

**Method:**
```python
def bagged_pcmci_plus(data, n_bootstrap=100, sample_ratio=0.8):
    """
    Bootstrap aggregate