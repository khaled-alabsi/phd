# Model Caching & Grid Search Design Pattern (Condensed)

**Purpose:** Train models with grid search, auto-save results, and load from cache on subsequent runs.

---

## Core Components

### 1. Artifacts Dataclass
```python
@dataclass
class ModelArtifacts:
    config_path: Path      # Hyperparameters JSON
    model_path: Path       # Trained model .keras
    metrics_path: Path     # Performance metrics JSON
    scaler_path: Path      # Optional: normalization stats JSON
```

### 2. Directory Structure
```
models/
├── {prefix}_config.json
├── {prefix}_model.keras
├── {prefix}_metrics.json
└── {prefix}_scaler.json (optional)
```

### 3. Main Method Signature
```python
def fit(
    X_train, X_val,
    param_grid=None,           # Grid search (mutually exclusive with model_config)
    model_config=None,         # Single config (mutually exclusive with param_grid)
    reset_experiment=False     # Force retrain flag
) -> Model
```

---

## Implementation Checklist

### Required Methods

- [ ] `__init__(model_dir, cache_prefix, verbose)`
- [ ] `fit()` - Main entry with cache check → grid search → save
- [ ] `_get_artifacts()` - Return ModelArtifacts with file paths
- [ ] `_cache_exists(artifacts)` - Check if all required files exist
- [ ] `_load_model(artifacts)` - Load config, model, metrics, scaler from disk
- [ ] `_save_model(artifacts)` - Save config, model, metrics, scaler to disk
- [ ] `_run_grid_search(X_train, X_val, param_grid)` - Grid search with best model selection
- [ ] `_train_single(X_train, X_val, config)` - Train one configuration
- [ ] `_iter_param_grid(grid)` - Generate all param combinations using `itertools.product`
- [ ] `_normalize_config(config)` - Type coercion and validation

### Class Variables

```python
DEFAULT_CONFIG: Dict[str, Any] = {...}           # Single default config
DEFAULT_PARAM_GRID: Dict[str, Sequence[Any]] = {...}  # Default grid for search
```

---

## Implementation Pattern

### 1. fit() Flow
```
1. Check cache (unless reset_experiment=True)
   → If exists: _load_model() and return
2. Validate params (param_grid XOR model_config)
3. If param_grid: _run_grid_search()
   If model_config: _train_single()
4. Store best_config, best_model, best_metrics
5. _save_model()
```

### 2. Grid Search Flow
```
1. Generate all configs from param_grid
2. For each config:
   - Clear TF session (tf.keras.backend.clear_session())
   - Build model
   - Train
   - Evaluate (track best val_loss)
   - Save weights if best
3. Rebuild best model from saved weights
4. Return (best_config, best_model, best_metrics)
```

### 3. Persistence
**Save:** JSON for config/metrics/scaler, `.keras` for model
**Load:** Parse JSON, load model with `tf.keras.models.load_model()`
**Important:** Convert tuples → lists for JSON serialization

---

## Critical Details

### Memory Management
**Always clear Keras session between grid search iterations:**
```python
tf.keras.backend.clear_session()  # Prevents memory leaks
```

### JSON Serialization
Tuples not JSON-serializable → convert to lists before saving:
```python
if isinstance(config["layers"], tuple):
    config["layers"] = list(config["layers"])
```

### Weight Preservation
Save weights before rebuilding best model:
```python
best_weights = model.get_weights()
# ... later ...
best_model.set_weights(best_weights)
```

### Parameter Grid Iterator
```python
from itertools import product
def _iter_param_grid(grid):
    keys = sorted(grid.keys())
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))
```

---

## Usage Patterns

| Scenario | Code |
|----------|------|
| **First run (grid search)** | `model.fit(X, param_grid=grid, reset_experiment=False)` |
| **Load cached** | `model.fit(X, reset_experiment=False)` |
| **Force retrain** | `model.fit(X, reset_experiment=True)` |
| **Single config** | `model.fit(X, model_config=cfg)` |
| **Use default grid** | `model.fit(X, param_grid=None)` |

---

## Common Pitfalls

| Problem | Solution |
|---------|----------|
| Memory leak during grid search | Call `tf.keras.backend.clear_session()` |
| Tuple serialization error | Convert to list before `json.dump()` |
| Missing scaler crashes load | Check `if scaler_path.exists()` before loading |
| Cache exists but incomplete | Check ALL artifact paths in `_cache_exists()` |
| Wrong model loaded | Ensure unique `cache_prefix` per experiment |

---

## Reference Implementation

**File:** `src/autoencoder_residual_regressor.py` (lines 29-714)

**Key Methods:**
- `fit()` (141-212)
- `_run_grid_search()` (413-460)
- `_save_model()` (680-699)
- `_load_model()` (662-678)
