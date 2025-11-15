## **Structured Research Note – Dynamic Weighting in Multivariate Process Anomaly Detection**

### 1. Context and Motivation

In process monitoring, the **Exponentially Weighted Moving Average (EWMA)** is a widely used method for detecting deviations from a stable, in-control process.
Unlike static control charts, EWMA assigns **exponentially decaying weights** to past observations, giving recent data more influence than older samples.
This allows the method to be both **responsive to sudden changes** and **robust to short-term noise**.

In a **multivariate** context, EWMA can be extended to track vector-valued process statistics, enabling detection of:

* **Transient process shifts** — short-lived anomalies in the latest samples.
* **Long-horizon process drift** — gradual or sustained departures from nominal conditions.



### 2. The Role of Weighting in Anomaly Detection

In multivariate EWMA-based monitoring, two distinct weighting mechanisms influence detection performance:

1. **EWMA Smoothing Parameter ($\lambda_t$)** – Governs the relative weight of current versus historical deviations in the smoothed statistic.

   * Higher $\lambda_t$ → more sensitivity to sudden changes.
   * Lower $\lambda_t$ → smoother trend tracking, better for detecting slow drifts.
   * Example: A process with seasonal variation may require higher $\lambda_t$ to remain responsive to real faults without overreacting to seasonal patterns.

2. **Adaptive Decision Limit Scaling Factor ($\beta_t$)** – Adjusts the control threshold based on evolving process statistics.

   * Dynamically raises or lowers the decision boundary depending on short-term variability.
   * Example: If variance in the last 100 samples increases due to a raw material change, $\beta_t$ may increase to reduce false alarms.



### 3. The Core Problem – Determining Optimal Weights

The challenge is **how to estimate $\lambda_t$ and $\beta_t$ automatically**.
Two main approaches exist:

* **Data-driven estimation** – Parameters inferred from rolling process statistics.
* **Expert-informed estimation** – Parameters set using domain knowledge, operating rules, or engineering constraints.

This leads to a natural question:

> If we can compute process statistics robust enough to determine $\lambda_t$ and $\beta_t$, could these same statistics be used directly for anomaly detection without EWMA smoothing?

This highlights a trade-off between:

* **Indirect detection** — Using EWMA to smooth statistics and detect cumulative deviations.
* **Direct detection** — Using raw or robust statistics as primary indicators.



### 4. The Statistical Indifference Region

Certain general-purpose statistics — such as the mean or variance — are:

* **Easy to compute** from any distribution.
* **Interpretable** across multiple process types.
* **Insufficiently discriminative** for complex anomaly scenarios.

These measures often occupy what can be called the **Statistical Indifference Region** — a range where the statistic changes but does not provide enough evidence to distinguish between normal and abnormal operation.

Example:

* A spike in variance may be acceptable in a high-mix production environment but would be abnormal in a precision-controlled chemical process.



### 5. Requirements for Universally Applicable Weighting Statistics

To design a dynamic weighting system that works across different processes and distributions, the selected metrics should be:

1. **Distribution-agnostic** — Robust to skewness, kurtosis, and heavy tails.
2. **Multivariate-capable** — Able to incorporate correlation structures among process variables.
3. **Stable-yet-responsive** — Minimal false alarms under stability, rapid reaction during changes.

Potential candidates:

* **Robust central tendency estimators** — Median, trimmed mean.
* **Robust dispersion estimators** — Median Absolute Deviation (MAD), Interquartile Range (IQR).
* **Multivariate robust measures** — Robust covariance estimators, shrinkage-based Mahalanobis distance.



### 6. Illustrative Example – Multivariate EWMA with Sliding-Window Threshold Adaptation

Consider a 3-variable process (temperature, pressure, flow rate):

* **EWMA update**:

  $$
  Z_t = \lambda_t X_t + (1 - \lambda_t) Z_{t-1}
  $$

  with $\lambda_t = 0.7$ (current data) and $1 - \lambda_t = 0.3$ (historical data).

* **Adaptive thresholding**:

  $$
  L_t = \beta_t L_0
  $$

  where $L_0$ is the baseline limit and $\beta_t$ is determined by rolling MAD over the last 50 samples:

  * If MAD > $1.5 \times$ baseline → $\beta_t$ increases by 20%.
  * If MAD < $0.8 \times$ baseline → $\beta_t$ decreases by 10%.

Outcome:

* **High sensitivity** to sudden multivariate shifts.
* **Robustness** to gradual, expected variations.





## Literature comparison — works on adaptive / multivariate EWMA and related adaptive charts

### 1. Lowry, Woodall, Champ & Rigdon (1992) — MEWMA (foundational)

* **What they do:** Introduce the Multivariate EWMA (MEWMA) chart and give design guidance (statistic, covariance of the EWMA vector, control limits based on χ² approximations or empirical calibration).
* **How they treat adaptation:** *No* adaptive smoothing or adaptive limits — λ is fixed and limits are set for desired in-control ARL. MEWMA provides temporal smoothing and strong baseline performance for small-to-moderate shifts. ([JSTOR][1], [ResearchGate][2])

**Takeaway:** MEWMA is the canonical starting point for any multivariate EWMA work; your method builds on this by letting λ and limits vary with real-time statistics.



### 2. Capizzi & Masarotto (series) and AEWMA lineage — adaptive EWMA families

* **What they do:** Propose adaptive EWMA (AEWMA) schemes that adjust chart behavior based on the current observation (often combining Shewhart and EWMA ideas or using conditional rules). Later work and follow-ups refine AEWMA design and CFAR-style thresholding. ([ResearchGate][3], [Wiley Online Library][4])
* **How they treat adaptation:** Emphasis is typically on **adaptive thresholds** or on combining statistics to change sensitivity; many AEWMA variants adapt the decision rule, sometimes by weighting observations based on the magnitude of the current standardized residual. Most classical AEWMA work focuses on univariate cases or scalar adaptations extended to multivariate by aggregation. ([ای ترجمه][5], [ResearchGate][6])

**Takeaway:** AEWMA literature directly supports adaptive decision limits and observation-weighted schemes; however, many AEWMA proposals either (a) adapt thresholds only or (b) are univariate or scalar extensions — your dual adaptation in a multivariate MEWMA framework is an extension of this line.



### 3. Adaptive Multivariate EWMA / AMEWMA papers (recent)

* **What they do:** Several recent works propose multivariate adaptive EWMA charts (AMEWMA), combining MEWMA statistics with adaptive features (e.g., data-driven λ or limit scaling), sometimes targeting specific distributional setups (e.g., monitoring covariance structure). Examples include dedicated AMEWMA proposals and function-based AEWMA for multivariate variance monitoring. ([ScienceDirect][7], [Nature][8], [ResearchGate][9])
* **How they treat adaptation:** These works commonly (a) propose specific adaptation rules for λ or for the limit, (b) evaluate performance (ARL) over a range of shift sizes, and (c) sometimes combine EWMA with Shewhart components to cover both small and large shifts. Robust dispersion measures (MAD/IQR) are less common than variance-based measures but appear in more recent robust adaptations. ([ScienceDirect][10], [PMC][11])

**Takeaway:** There is emerging, active work on adaptive multivariate EWMA charts; your proposal should be framed as contributing to this trend by (1) using robust dispersion (MAD/IQR) for βₜ and (2) using an SNR-style rule for λₜ — both together in one multivariate MEWMA is less explored.



### 4. Sparks (2000) and related adaptive CUSUM work

* **What they do:** Propose adaptive CUSUM (and variants) that estimate the unknown shift magnitude online (sometimes via EWMA estimators) and adapt the CUSUM reference parameter accordingly to improve performance across a range of shift sizes. ([ResearchGate][12], [ScienceDirect][13])
* **How they treat adaptation:** Sparks and follow-ups adapt the detection *reference* or *drift* used in CUSUM, often using an EWMA estimator inside the CUSUM scheme. These are conceptually “dual” approaches (estimating shifts and plugging into an accumulator), but they are CUSUM-centric rather than pure EWMA.

**Takeaway:** Sparks shows a pattern: adaptive detection can be implemented either by adapting the accumulator (CUSUM) or by using an adaptive-smoothed estimator inside an accumulator. Your work chooses the more direct EWMA smoothing + adaptive limits path rather than adapting a CUSUM accumulator.



### 5. Robust and modern adaptive chart papers (2020–2024)

* **What they do:** Newer papers develop AEWMA/MEWMA variations addressing non-normality, heavy tails, high dimensionality, and robust dispersion monitoring (some apply MAD, robust covariance, shrinkage). There are also ML-assisted adaptive EWMA proposals that learn adaptation rules from data. ([Nature][14], [Wiley Online Library][15], [PMC][11])
* **How they treat adaptation:** Approaches include (a) robust scale estimators to adjust limits, (b) shrinkage estimators for covariance in high-dimensional MEWMA, and (c) learned or optimized λ profiles across operating modes. These papers suggest practical techniques you can borrow (robust covariance, empirical calibration of adaptive limits, ML for λ scheduling). ([Nature][8], [PMC][11])

**Takeaway:** These modern methods align with your use of robust statistics (MAD, shrinkage covariance) for βₜ and with data-driven λ scheduling; they show the field is moving toward exactly the kinds of adaptations you propose, but few combine both robust βₜ and SNR λₜ in one multivariate MEWMA.



## Summary of differences — your approach vs. prior work

* **Dual adaptation (λₜ + βₜ) in a multivariate MEWMA:** many prior works adapt either λ or h (or adapt CUSUM reference). Papers on AEWMA/AMEWMA address parts of this, but **few explicitly combine SNR-based λₜ with robust-dispersion βₜ inside a multivariate MEWMA**. ([ResearchGate][3], [ScienceDirect][7])
* **Robust dispersion for threshold scaling:** variance-based scaling is common; using MAD/IQR for βₜ provides distribution-independence and is less prevalent in older AEWMA literature (but increasingly present in recent robust adaptive papers). ([Nature][8], [PMC][11])
* **Practical calibration and ARL analysis:** the canonical MEWMA work gives ARL design tools (Lowry et al.); adaptive charts often require simulation to control ARL under non-stationarity — your framework should include empirical ARL calibration under baseline and perturbed regimes. ([JSTOR][1], [ScienceDirect][10])



[1]: https://www.jstor.org/stable/1269551?utm_source=chatgpt.com "A Multivariate Exponentially Weighted Moving Average Control Chart"
[2]: https://www.researchgate.net/profile/William-Woodall/publication/261191178_A_Multivariate_Exponentially_Weighted_Moving_Average_Control_Chart/links/5c952b5ea6fdccd4603366df/A-Multivariate-Exponentially-Weighted-Moving-Average-Control-Chart.pdf?utm_source=chatgpt.com "[PDF] A Multivariate Exponentially Weighted Moving Average Control ..."
[3]: https://www.researchgate.net/publication/323334353_An_efficient_adaptive_EWMA_control_chart_for_monitoring_the_process_mean?utm_source=chatgpt.com "An efficient adaptive EWMA control chart for monitoring the process ..."
[4]: https://onlinelibrary.wiley.com/doi/full/10.1002/qre.3324?utm_source=chatgpt.com "Design of adaptive EWMA control charts using the conditional false ..."
[5]: https://e-tarjome.com/storage/panel/fileuploads/2019-09-21/1569051996_E13528-e-tarjome.pdf?utm_source=chatgpt.com "[PDF] An adaptive exponentially weighted moving average-type control ..."
[6]: https://www.researchgate.net/publication/273351117_Optimal_Design_of_the_Adaptive_Exponentially_Weighted_Moving_Average_Control_Chart_over_a_Range_of_Mean_Shifts?utm_source=chatgpt.com "Optimal Design of the Adaptive Exponentially Weighted Moving ..."
[7]: https://www.sciencedirect.com/science/article/abs/pii/S0360835218305205?utm_source=chatgpt.com "An adaptive multivariate EWMA chart - ScienceDirect.com"
[8]: https://www.nature.com/articles/s41598-023-45399-3?utm_source=chatgpt.com "Adaptive multivariate dispersion control chart with application to ..."
[9]: https://www.researchgate.net/publication/255728013_A_Multivariate_Adaptive_Exponentially_Weighted_Moving_Average_Control_Chart?utm_source=chatgpt.com "A Multivariate Adaptive Exponentially Weighted Moving Average ..."
[10]: https://www.sciencedirect.com/science/article/abs/pii/S037722171930565X?utm_source=chatgpt.com "An adaptive exponentially weighted moving average-type control ..."
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11109341/?utm_source=chatgpt.com "New adaptive EWMA CV control chart with application to the ..."
[12]: https://www.researchgate.net/publication/259190508_SPARKS_R_CUSUM_Charts_for_Signalling_Varying_Location_Shifts_Journal_of_Quality_Technology_32_no_2_2000_157-71?utm_source=chatgpt.com "SPARKS, R. \"CUSUM Charts for Signalling Varying Location Shifts ..."
[13]: https://www.sciencedirect.com/science/article/abs/pii/S016794730800090X?utm_source=chatgpt.com "Adaptive CUSUM procedures with Markovian mean estimation"
[14]: https://www.nature.com/articles/s41598-024-83780-y?utm_source=chatgpt.com "A novel EWMA-based adaptive control chart for industrial ... - Nature"
[15]: https://onlinelibrary.wiley.com/doi/10.1002/qre.3797?af=R&utm_source=chatgpt.com "A Machine Learning Approach to Adaptive EWMA Control Charts"






# How prior work computes adaptive parameters (λₜ and/or βₜ) — detailed, technical comparison

> Note: I group similar methods and give the typical formulas/pseudocode used in the papers. The strongest / most load-bearing claims are cited.



## 1) Lowry, Woodall, Champ & Rigdon (1992) — MEWMA (canonical baseline)

* **What they do (baseline):** Introduce Multivariate EWMA (MEWMA) with a *fixed* smoothing parameter λ and *fixed* control limits (often based on a χ² approximation or empirical calibration). They provide closed-form variance of the MEWMA vector and ARL design guidelines. This paper is the canonical, non-adaptive starting point. ([conservancy.umn.edu][1], [digitalcommons.georgiasouthern.edu][2])
* **Adaptive values:** none (λ and limits fixed).
* **Problems fixed / remaining:** MEWMA gives good detection of small-to-moderate shifts but suffers from inertia with large shifts and can be suboptimal under non-stationary variance; this motivates adaptive variants. ([conservancy.umn.edu][1])



## 2) Capizzi & Masarotto (2003) — Adaptive EWMA (AEWMA) using score functions (Huber-style)

* **Core idea (how λₜ is produced):** AEWMA adapts the chart by making the effective weight a smooth function of the *current standardized residual* (the magnitude of the difference between the new observation and the current plot statistic). The adaptation is implemented via a **score function** (Capizzi & Masarotto propose Huber-type score functions). Concretely, the update uses a function ψ of the current error to modulate the contribution of the new observation. ([Tandfonline][3], [halweb.uc3m.es][4])
* **Typical implementation pattern (pseudocode / formula):**

  * Let residual rₜ = (xₜ − Zₜ−1)/s (standardized error).
  * Define a bounded score ψ(rₜ) (Huber or similar).
  * Update the smoothed statistic as something like:

    $$
    Z_t = (1 - k_t)\,Z_{t-1} + k_t\,x_t,\quad\text{with } k_t = g\big(\lvert r_t\rvert\big),
    $$

    or equivalently use ψ directly inside the update. Capizzi & Masarotto choose g/ψ so that small residuals keep small λ (smooth long-memory), whereas large residuals increase λ (short-memory, Shewhart-like reaction). ([Tandfonline][3], [ای ترجمه][5])
* **How they choose the score g / parameters:** They recommend Huber-type functions and propose a two-step optimization to set parameters (optimize ARL performance over a range of shift magnitudes). The Huber function performed consistently well in their experiments. ([Tandfonline][3], [ای ترجمه][5])
* **Problems this fixes:**

  * Reduces *inertia* of EWMA in the presence of large shifts (by increasing the effective weight on large residuals).
  * Retains EWMA sensitivity to small shifts (by keeping weights small when residuals are small).
* **Limitations / implementation notes:** Original AEWMA papers are mainly univariate; multivariate extensions typically apply the same idea to scalarized residuals (e.g., Hotelling T or profile errors) or extend the score function to multivariate norms. ([halweb.uc3m.es][4])



## 3) Adaptive multivariate EWMA (AMEWMA and variants) — recent literature

* **What they do (typical patterns):** Extend AEWMA ideas to multivariate monitoring. Two main adaptation families appear:

  1. **Adaptive λₜ via estimated shift magnitude** (SNR-like or estimated |shift|): λₜ increases when an estimated shift magnitude crosses thresholds.
  2. **Adaptive limits (βₜ)** via local dispersion measures (variance, MAD, or CFAR principles) or via online covariance shrinkage. ([ScienceDirect][6], [arXiv][7])
* **Representative implementation motifs and formulas:**

  * **SNR-derived λₜ:** compute recent variance (or energy) vs baseline and set

    $$
    \mathrm{SNR}_t = \frac{\mathrm{Var}_{\text{recent}}(\text{score})}{\mathrm{Var}_{\text{baseline}}(\text{score})},\qquad
    \lambda_t=\frac{\mathrm{SNR}_t}{1+\mathrm{SNR}_t}.
    $$

    This raises λ when the signal-to-noise ratio increases. (Several applied AMEWMA papers use this or closely related rules.) ([ScienceDirect][8])
  * **CFAR / conditional false alarm rate for βₜ:** choose βₜ so that the conditional false-alarm probability (given recent dispersion) remains at target; equivalently scale L₀ by a function of recent dispersion to keep conditional ARL stable. Papers using CFAR-style design show how to pick scaling to maintain stable false alarm probability across non-stationary variance regimes. ([Wiley Online Library][9])
  * **Robust dispersion-based limits:** use sliding-window MAD or IQR:

    $$
    \beta_t = \min\!\left(1,\; \frac{\operatorname{MAD}_t}{\operatorname{MAD}_0}\right),\quad L_t = L_0\big(1 + c\cdot(\beta_t-1)\big).
    $$

    Recent robust AMEWMA work explicitly recommends MAD/IQR when non-normality or heavy tails are expected. ([ScienceDirect][8], [arXiv][7])
  * **Shrinkage covariance for high-dimension:** when p is large, papers use Ledoit–Wolf or other shrinkage covariance estimators inside the MEWMA statistic so the adaptive chart remains stable. ([ScienceDirect][8])
* **Problems this fixes:**

  * Retains ARL control under changing dispersion (CFAR-style scaling).
  * Improves sensitivity across a range of shift sizes (SNR/λₜ adaptation).
  * Handles non-normal/heavy-tailed data (robust dispersion).
  * Addresses high-dimensional instability (shrinkage covariance). ([ScienceDirect][6], [arXiv][7])



## 4) Sparks (2000) and adaptive CUSUM families — EWMA used as an internal estimator

* **How they implement adaptation:** Sparks and related adaptive CUSUM methods often **estimate the unknown shift online** using an EWMA (or generalized EWMA) and then plug that estimate into the CUSUM reference value, or use a weighting function to blend estimators. Practically:

  * Compute EWMA estimate μ̂ₜ of the current process mean.
  * Update the CUSUM reference kₜ (or the CUSUM recursion) using μ̂ₜ so the accumulator becomes adaptive to the estimated shift. ([ResearchGate][10], [users.iems.northwestern.edu][11])
* **Typical formulas:** EWMA estimator μ̂ₜ = α xₜ + (1−α) μ̂ₜ₋₁ used to set kₜ or to modulate the CUSUM increment.
* **Problems this fixes:** Makes CUSUM less dependent on a single assumed shift magnitude; improves detection over a broader range of shifts. But it mixes EWMA and CUSUM ideas rather than remaining a pure EWMA chart. ([users.iems.northwestern.edu][11])



## 5) Recent robust / ML-assisted adaptive charts (2018–2024)

* **What they do:** Use data-driven or learned rules to set λₜ and/or thresholds:

  * **ML / ANN / SVR-based λ schedulers:** learn a mapping from recent feature vectors (residual magnitude, recent variance, kurtosis) to an optimal λₜ in simulation or from historical labeled events. This yields nonparametric, non-linear adaptation functions. ([Nature][12], [ResearchGate][13])
  * **Robust scale + shrinkage:** use MAD/IQR for βₜ and Ledoit–Wolf shrinkage for Σ estimate in MEWMA to stabilize multivariate statistics. ([arXiv][7], [ScienceDirect][8])
* **Problems these fix:** adaptivity in complex, non-normal, high-dimensional settings; automatic tuning learned from data rather than heuristics. They usually require offline training / simulations for calibration. ([Nature][12], [arXiv][7])



# Practical recipe patterns you can reuse (concrete formulas you can plug into code)

1. **SNR-based λₜ (simple, interpretable):**

   * Compute score dₜ (e.g., Hotelling T² or Mahalanobis distance of xₜ).
   * Compute recent variance v\_recent (window w) and baseline variance v\_0.
   * SNR = v\_recent / v\_0.
   * λₜ = SNR / (1 + SNR)  (maps SNR ∈ (0,∞) → λₜ ∈ (0,1)).
   * Rationale / citation: used by several AMEWMA variants and matches the intuition in recent adaptive MEWMA literature. ([ScienceDirect][8])

2. **Residual-score / Huber-based λₜ (Capizzi & Masarotto style):**

   * rₜ = standardized residual = (scalarized error) / s.
   * kₜ = g(|rₜ|) where g is Huber-like or sigmoid: small |rₜ| → small kₜ; large |rₜ| → kₜ ≈ 1.
   * Update Zₜ = (1 − kₜ) Zₜ₋₁ + kₜ xₜ.
   * Rationale / citation: Capizzi & Masarotto (AEWMA); good for switching between Shewhart-like response and EWMA smoothing. ([Tandfonline][3])

3. **MAD-based βₜ (robust threshold scaling):**

   * MADₜ = median\_i |score\_{t-i} − median\_j score\_{t-j}| over window w.
   * βₜ = clamp(MADₜ / MAD\_0, ϵ, 1) (or min(1, MADₜ/MAD\_0)).
   * Lₜ = L\_0 × (1 + c · (βₜ − 1)) or Lₜ = L\_0 × βₜ.
   * Rationale / citation: robust AMEWMA and recent work recommend MAD/IQR to avoid false alarms due to heavy tails or transient variance spikes. ([ScienceDirect][8], [arXiv][7])

4. **CFAR-style βₜ (conditional false alarm control):**

   * Estimate conditional distribution of the statistic given recent variance; pick scaling so P(false alarm | recent dispersion) ≈ target. This is more work but gives principled false-alarm control in non-stationary regimes. ([Wiley Online Library][9])

5. **EWMA-as-estimator inside adaptive CUSUM (Sparks style):**

   * μ̂ₜ = α xₜ + (1 − α) μ̂ₜ₋₁; use μ̂ₜ to set CUSUM reference kₜ or increment. Good when combining benefits of both charts. ([users.iems.northwestern.edu][11])



# Which problems each family addresses (quick mapping)

* **AEWMA (Capizzi & Masarotto)** — solves *inertia vs large shifts* by making λ responsive to residual magnitude (Huber-style). Good when you want one chart to cover both small and large shifts; originally univariate but extensible. ([Tandfonline][3])
* **SNR-based AMEWMA** — makes λ depend on measured signal strength; solves adaptivity to varying signal amplitude and non-stationary noise. Good for processes where energy/variance change signals important regime changes. ([ScienceDirect][8])
* **MAD/CFAR limit-scaling** — solves *variance-induced false alarms* by scaling limits using robust dispersion or conditional-false-alarm design. Good for non-normal or heavy-tailed data. ([arXiv][7], [Wiley Online Library][9])
* **Shrinkage covariance + robust statistics** — solves high-dimensional instability and non-normality in multivariate MEWMA. Good for p \~ n or p > n. ([ScienceDirect][8])
* **ML / data-learned λ schedulers** — solve complex non-linear mapping problems where a simple analytic rule is insufficient; need training data and offline calibration. ([Nature][12])



# Short checklist you can copy into Methods when describing related work

* Capizzi & Masarotto (2003): adaptive λ via bounded score function (Huber); two-step parameter optimization for ARL across shift sizes. ([Tandfonline][3])
* AMEWMA papers (2015–2024): SNR-style λ, MAD/IQR thresholds, CFAR limit-scaling; shrinkage covariance for high-dimension. ([ScienceDirect][8], [arXiv][7])
* Sparks (2000) / ACUSUM family: EWMA used as shift estimator inside CUSUM; adapt reference values for wider shift coverage. ([ResearchGate][10], [users.iems.northwestern.edu][11])


[1]: https://conservancy.umn.edu/bitstreams/d6001e06-c768-4610-9bf2-3e72392bc727/download?utm_source=chatgpt.com "[PDF] A General Multivariate Exponentially Weighted Moving Average ..."
[2]: https://digitalcommons.georgiasouthern.edu/math-sci-facpubs/489/?utm_source=chatgpt.com "A Multivariate Exponentially Weighted Moving Average Control Chart"
[3]: https://www.tandfonline.com/doi/abs/10.1198/004017003000000023?utm_source=chatgpt.com "An Adaptive Exponentially Weighted Moving Average Control Chart"
[4]: https://halweb.uc3m.es/esp/Personal/personas/amalonso/esp/Preprints/Adaptive%20EWMA%20control%20charts%20with%20time-varying%20smoothing%20parameter.pdf?utm_source=chatgpt.com "[PDF] Adaptive EWMA Control Charts with Time-Varying Smoothing ..."
[5]: https://e-tarjome.com/storage/panel/fileuploads/2019-09-21/1569051996_E13528-e-tarjome.pdf?utm_source=chatgpt.com "[PDF] An adaptive exponentially weighted moving average-type control ..."
[6]: https://www.sciencedirect.com/science/article/abs/pii/S0360835218305205?utm_source=chatgpt.com "An adaptive multivariate EWMA chart - ScienceDirect.com"
[7]: https://arxiv.org/abs/2403.03837?utm_source=chatgpt.com "An Adaptive Multivariate Functional EWMA Control Chart"
[8]: https://www.sciencedirect.com/science/article/pii/S0360835218305205?utm_source=chatgpt.com "An adaptive multivariate EWMA chart - ScienceDirect"
[9]: https://onlinelibrary.wiley.com/doi/full/10.1002/qre.3324?utm_source=chatgpt.com "Design of adaptive EWMA control charts using the conditional false ..."
[10]: https://www.researchgate.net/publication/259190508_SPARKS_R_CUSUM_Charts_for_Signalling_Varying_Location_Shifts_Journal_of_Quality_Technology_32_no_2_2000_157-71?utm_source=chatgpt.com "SPARKS, R. \"CUSUM Charts for Signalling Varying Location Shifts ..."
[11]: https://users.iems.northwestern.edu/~apley/papers/2008%2C%20Adaptive%20CUSUM%20procedures%20with%20EWMA-based%20shift%20estimators.pdf?utm_source=chatgpt.com "[PDF] Adaptive CUSUM procedures with EWMA-based shift estimators"
[12]: https://www.nature.com/articles/s41598-024-83780-y?utm_source=chatgpt.com "A novel EWMA-based adaptive control chart for industrial ... - Nature"
[13]: https://www.researchgate.net/publication/379736202_An_Adaptive_Exponentially_Weighted_Moving_Average_Based_Support_Vector_Data_Description_Machine_for_Multivariate_Process?utm_source=chatgpt.com "(PDF) An Adaptive Exponentially Weighted Moving Average Based ..."




