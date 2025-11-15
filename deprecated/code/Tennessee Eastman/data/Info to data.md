# Tennessee Eastman Process Simulation Dataset

## Dataset Overview

The Tennessee Eastman Process (TEP) dataset is a well-known benchmark used for testing fault detection and process monitoring techniques. It simulates a chemical production process under normal and faulty conditions.

---

## Data Files

### 1. Fault-Free Training Data

* **Runs:** 500
* **Samples per run:** 500 (25 hours at 3-minute intervals)
* **Label:** `faultNumber = 0` (no fault)
* **Columns:**

  * `xmeas_1` to `xmeas_41`: Measured variables (sensor readings)
  * `xmv_1` to `xmv_11`: Manipulated variables (control settings)
  * `simulationRun`: Run identifier
  * `sample`: Timestep within a run

### 2. Faulty Test Data

* **Runs:** 1 per fault (20 faults)
* **Samples per run:** 960 (48 hours at 3-minute intervals)
* **Label:** `faultNumber ∈ [1, 20]` (fault introduced at sample 160)
* **Columns:** Same as training data
* **Structure:**

  * First 160 samples: in-control (normal operation)
  * Remaining 800 samples: out-of-control (faulty)
  * `faultNumber` remains constant for entire run

### 3. Fault Definitions (faultNumber)

Each `faultNumber` corresponds to a predefined process disturbance:

* Types: step change, drift, random variation, valve sticking, unknown disturbances
* Example: feed composition change, cooling water failure, sensor bias, etc.

---

## Variable Descriptions

### Measured Variables (`xmeas_1` to `xmeas_41`)

Process sensors (flows, pressures, levels, compositions). Examples:

* `xmeas_1`: A Feed (kscmh)
* `xmeas_2`: D Feed (kscmh)
* ...
* `xmeas_39`: Product G Composition (%)
* `xmeas_41`: Product D Composition (%)

### Manipulated Variables (`xmv_1` to `xmv_11`)

Control inputs (valves, speed, flow rates). Examples:

* `xmv_1`: D Feed Valve
* `xmv_2`: E Feed Valve
* `xmv_3`: A Feed Valve
* `xmv_11`: Reactor Cooling Water Valve

### Metadata Columns

* `faultNumber`: Integer label for fault type (0 = no fault)
* `simulationRun`: Integer identifier for the run
* `sample`: Time index (1 to 500 or 960)

---

## Key Observations

* **Training data** contains only fault-free data
* **Test data** contains normal + fault phases, but `faultNumber` does not reflect timing
* Use `sample >= 160` to detect when the fault starts

So:
The first 160 samples ≈ in-control baseline
Samples after 160 ≈ faulty (out-of-control) region
But the label faultNumber stays constant for the whole run — even before the fault starts

### Example Labeling Code:

```python
import pandas as pd

df = pd.read_csv("test_data.csv")
df["fault_active"] = df["sample"] >= 160
```

This allows proper separation of in-control vs out-of-control behavior within each fault run.

---

## Summary Table

| Data File           | Runs | Samples | Variables                 | Fault Range        |
| ------------------- | ---- | ------- | ------------------------- | ------------------ |
| Fault-Free Training | 500  | 500     | 41 measured + 11 controls | faultNumber = 0    |
| Faulty/Test Data    | 20   | 960     | same as training          | faultNumber ∈ 1–20 |

---

## Recommended Next Steps

* Add `fault_active` flag to enable time-aware labeling
* Use consistent preprocessing for both training and test runs
* Consider dimensionality reduction (e.g., PCA) to visualize variable shifts across faults
