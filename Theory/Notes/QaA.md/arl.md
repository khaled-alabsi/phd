
**Question:**

I am running Monte Carlo simulations for fault detection in a multivariate process using MCUSUM. For each simulation run and each fault, I calculate:

* **ARL0**: first false alarm index
* **ARL1**: first detection delay

I am facing two challenges:

1. In some runs, no false alarm or no detection occurs, so ARL values are `None`.
2. Currently, I record only the first detection or false alarm in each run, which ignores subsequent behavior—i.e., after the first detection, the model might fail to detect further out-of-control points.

I am considering several approaches to address these issues:

* **Handling `None` values**:

  1. Exclude `None` values when calculating average ARL.
  2. Replace `None` with the maximum sequence length.
  3. Report both the mean over detected runs and the number/fraction of runs with no detection.

* **Accounting for detection beyond the first alarm**:

  1. Keep ARL as is (time until first signal) and treat subsequent detections separately.
  2. Compute additional metrics, such as:

     * Fraction of out-of-control points detected across the sequence.
     * Average delay for all detected points, not just the first one.
     * Number or fraction of missed detections after the first alarm.

My goal is to provide a **comprehensive evaluation of detection performance** that reflects both the speed of initial detection and the reliability of continued monitoring.

Which approach is recommended or considered best practice in SPC or Monte Carlo-based process monitoring for summarizing detection performance under these conditions?

