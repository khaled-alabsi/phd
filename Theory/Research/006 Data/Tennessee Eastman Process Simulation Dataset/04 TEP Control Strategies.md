# Control Strategies in the Tennessee Eastman Process (TEP)

## 1. **Closed-Loop Control**

### Definition:

Closed-loop control (also called **feedback control**) is a control strategy where the system continuously measures outputs (process variables) and adjusts inputs (manipulated variables) based on deviations from a desired setpoint.

### Core Principle:

**Error = Setpoint – Measured Value**

Controllers act to minimize the error by adjusting process inputs.

### Key Elements:

* **Controller** (e.g., PID)
* **Sensor** (measures output)
* **Actuator** (applies control action)

### Example in TEP:

* A temperature controller maintains reactor temperature by adjusting the coolant flow rate.
* If temperature rises above setpoint, controller increases coolant flow.

---

## 2. **Hierarchical Control Strategy**

### Definition:

A **hierarchical control strategy** uses multiple control layers, each responsible for a different scope or timescale. Lower levels handle fast dynamics and stability, higher levels manage optimization and constraint handling.

### Levels in TEP (from bottom to top):

| Level                   | Purpose                                          | Example in TEP                                 |
| ----------------------- | ------------------------------------------------ | ---------------------------------------------- |
| **Regulatory control**  | Basic PID loops to keep variables near setpoints | Maintain pressure, flow, level                 |
| **Supervisory control** | Setpoint updates, coordination among controllers | Adjusts feed ratios based on product specs     |
| **Optimization layer**  | Economic or efficiency optimization              | Maximize throughput or yield under constraints |

### Benefits:

* **Stability at lower layers** ensures robustness.
* **Flexibility at higher layers** allows for optimization and adaptation.
* Each layer has **different update frequencies** and **decision scopes**.

### Example Hierarchy:

```plaintext
Level 3: Economic Optimization (slow loop)
Level 2: Supervisory Coordination (medium loop)
Level 1: Regulatory PID Control (fast loop)
```
