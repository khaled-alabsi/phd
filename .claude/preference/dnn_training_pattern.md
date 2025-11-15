# DNN Training Pattern Preference

## Standard Training Pattern

All DNN-based models should follow this pattern:

### Constructor
```python
model = ModelClass(
    pretrained_model=<encoder/autoencoder>,  # Pre-trained model passed as argument
    verbose=1,                                # 0=silent, 1=progress messages
    cache_prefix="model_name"                 # Unique prefix for caching
)
```

### Training Method
```python
model.fit(
    X_train,
    X_val,
    param_grid=PARAM_GRID,  # None to use DEFAULT_PARAM_GRID
    reset=False,             # True to force retrain
)
```

## Required Features

### 1. Grid Search
- Support `param_grid` parameter (dict with lists of values)
- If `param_grid=None`, use class DEFAULT_PARAM_GRID
- Save all grid search results to file

### 2. Automatic Caching
- Check if cached model exists before training
- If cached and `reset=False`, load from cache
- If cached and `reset=True`, force retrain
- Save artifacts:
  - `{cache_prefix}_config.json` - Best config
  - `{cache_prefix}_model.keras` - Trained model
  - `{cache_prefix}_metrics.json` - Performance metrics
  - `{cache_prefix}_grid_results.json` - All grid search results

### 3. Cache Management
- Use `model_dir` for all saved files (default: code/models/)
- Use `cache_prefix` to distinguish different experiments
- Check cache with `_cache_exists()`
- Load cache with `_load()`
- Save cache with `_save()`

### 4. Verbose Mode
- `verbose=0`: Silent training
- `verbose=1`: Print progress messages (grid search iterations, loading, saving)

## Example Implementation Pattern

```python
class MyDNNModel:
    DEFAULT_PARAM_GRID = {
        "hidden_layers": [(64, 32), (128, 64)],
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [128, 256],
        "epochs": [100],
        "patience": [15],
    }

    def __init__(self, encoder, verbose=0, cache_prefix="my_model"):
        self.encoder = encoder
        self.verbose = verbose
        self.cache_prefix = cache_prefix
        self.model_dir = Path("code/models")
        self.model = None
        self.config = None
        self.metrics = None
        self.grid_results = []

    def fit(self, X_train, X_val, param_grid=None, reset=False):
        artifacts = self._get_artifacts()

        # Check cache
        if not reset and self._cache_exists(artifacts):
            self._load(artifacts)
            if self.verbose:
                print("[Model] Loaded cached model.")
            return self.model

        # Grid search
        grid = param_grid or self.DEFAULT_PARAM_GRID
        cfg, model, metrics = self._run_grid_search(X_train, X_val, grid)

        self.model = model
        self.config = cfg
        self.metrics = metrics

        # Save
        self._save(artifacts)
        return model
```

## Key Points
- **Never retrain** if cache exists and reset=False
- **Always save** best model from grid search
- **Support both** single config and grid search modes
