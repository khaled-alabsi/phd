Here’s the extended description and example text structure for:

---

## **1. Introduction**

### **1.1 Motivation**
**Description:**  
Explain why monitoring production processes is crucial. Emphasize the increasing complexity of multivariate data in modern manufacturing and how this complexity creates challenges for traditional quality control methods.

**Example:**  
> In modern production environments, the detection of disruptions and anomalies plays a critical role in ensuring product quality and operational efficiency. As industrial systems become increasingly interconnected and data-rich, traditional univariate monitoring methods often fail to capture subtle or correlated shifts in quality characteristics. This calls for advanced methods that can adapt to multivariate time-dependent data while maintaining interpretability and responsiveness.

---

### **1.2 Problem Statement**
**Description:**  
Identify the limitations in current SPC and multivariate control chart approaches, such as reliance on multivariate normality or static control limits.

**Example:**  
> Classical multivariate control charts, including Hotelling’s $T^2$ and multivariate CUSUM charts, assume a fixed distributional form—typically multivariate normality—and static parameters. These assumptions limit their applicability in real-world environments characterized by noise, non-linear dependencies, and abrupt or gradual shifts in process dynamics.

---

### **1.3 Research Questions**
**Description:**  
List the key questions derived from your exposé (e.g., distribution modeling, smoothing, chart design, user adaptation, performance metrics).

**Example:**  
> This research addresses the following questions:  
> 1. Which alternative multivariate distributions are best suited for modeling correlated quality characteristics?  
> 2. How can smoothing techniques improve anomaly detection in multivariate time series?  
> 3. What are effective approaches for optimizing control chart design in dynamic settings?  
> 4. How can we incorporate user-specific constraints and domain knowledge into control chart design?  
> 5. How can we quantify the contribution of individual variables to a detected disruption?

---

### **1.4 Research Objectives and Contributions**
**Description:**  
Clearly state what your work intends to achieve and how it contributes to the literature.

**Example:**  
> The objective of this dissertation is to develop, implement, and evaluate advanced multivariate control chart techniques that integrate statistical modeling with AI-based anomaly detection. Contributions include:
> - Novel control chart frameworks using non-Gaussian multivariate distributions
> - Integration of neural smoothing mechanisms into time series monitoring
> - A new method for attributing anomalies to specific quality characteristics
> - An open-source Python package for reproducible simulations and evaluations

---

### **1.5 Thesis Structure**
**Description:**  
Short summary of chapters.

**Example:**  
> Chapter 2 surveys the foundational theory of control charts and anomaly detection. Chapter 3 develops the methodological framework. Chapter 4 explores alternative distributions. Chapter 5 details new chart designs and smoothing methods. Chapter 6 presents implementation and experimental results. Chapter 7 discusses real-world validation, followed by a concluding chapter.

---

## **2. Background & Literature Review**

### **2.1 Statistical Process Control and Control Charts**
**Description:**  
Summarize traditional SPC and the role of control charts (Shewhart, CUSUM, EWMA). Emphasize detection goals and statistical assumptions.

**Example:**  
> SPC provides tools for monitoring process variation to ensure stability and quality. The Shewhart chart, introduced in the 1920s, offers simplicity and ease of use but lacks sensitivity to small shifts. To address this, the cumulative sum (CUSUM) and exponentially weighted moving average (EWMA) charts were developed. These charts utilize historical process information, allowing earlier detection of moderate deviations.

---

### **2.2 Multivariate Control Charts**
**Description:**  
Explain the extension to multivariate settings. Discuss Hotelling's $T^2$ , multivariate CUSUM/EWMA, and issues like variable correlation and interpretability.

**Example:**  
> In real-world processes, multiple quality characteristics often interact. Monitoring them individually ignores potential interdependencies. Hotelling’s $T^2$ statistic extends the Shewhart concept to the multivariate domain, using the Mahalanobis distance to measure deviations from the mean vector. However, it lacks granularity in identifying which variable contributes to an alarm.

---

### **2.3 Limitations of Classical Approaches**
**Description:**  
Discuss shortcomings such as fixed distributions, lack of flexibility, and inability to explain root causes.

**Example:**  
> Classical multivariate charts rely heavily on the assumption of multivariate normality, which rarely holds in complex, high-variability manufacturing settings. Furthermore, when a process shift is detected, traditional methods provide limited guidance on which quality characteristic caused the deviation, complicating root cause analysis.

---

### **2.4 Anomaly Detection in Time Series**
**Description:**  
Introduce anomaly detection concepts and relate them to SPC. Include AI-based methods like autoencoders, forecasting residuals, etc.

**Example:**  
> In recent years, machine learning techniques have emerged for detecting anomalies in time series data. Methods such as autoencoders, LSTMs, and hybrid statistical-neural frameworks offer dynamic modeling of data streams. Unlike fixed-threshold control charts, these methods adapt to complex patterns but often sacrifice explainability.

---

### **2.5 Software and Tools**
**Description:**  
Briefly describe available packages and tools (R, Python, pyspc, sklearn, etc.) and gaps that justify your implementation.

**Example:**  
> While libraries such as `pyspc` or `qcc` offer implementations of standard control charts, they do not support modern extensions such as dynamic thresholding, non-Gaussian distributions, or explainable AI components. This motivates the development of a flexible, open-source platform tailored to research needs.

---

Here is the extended description and example structure for:

---

## **3. Methodological Framework**

### **3.1 Conceptual Overview**
**Description:**  
Outline your overall approach to combining classical SPC with modern AI methods. Emphasize your modular view (data → modeling → smoothing → charting → decision).

**Example:**  
> The proposed framework integrates classical statistical process control techniques with modern anomaly detection algorithms in a modular architecture. The pipeline begins with preprocessing and data transformation, followed by probabilistic or neural modeling of normal behavior, smoothing of residuals or scores, and multivariate control chart evaluation. Each stage is independently replaceable, allowing for flexible experimentation.

---

### **3.2 Data Assumptions**
**Description:**  
Define assumptions like stationarity, independence between batches, or known labels for training phases.

**Example:**  
> The framework assumes that historical in-control data is available and representative of normal process behavior. Time series are pre-processed to ensure local stationarity. Labeling of abnormal states is only assumed for evaluation purposes, not during training. Observations within a time window are considered autocorrelated, while separate windows are independent.

---

### **3.3 Model-Based Control Charts**
**Description:**  
Introduce the generalization of control charts based on model residuals, likelihood scores, or latent representations.

**Example:**  
> Instead of raw process variables, the control charts monitor statistical summaries—e.g., reconstruction error from an autoencoder, negative log-likelihood under a fitted copula, or Mahalanobis distance in a learned latent space. This allows capturing non-linear and high-dimensional relationships that traditional multivariate charts cannot exploit.

---

### **3.4 Role of Distributional Modeling**
**Description:**  
Describe why distribution modeling is necessary and what alternatives you explore (e.g., t-distributions, copulas, empirical density).

**Example:**  
> To address non-normality in multivariate settings, the framework explores alternative modeling approaches, including Student-t distributions, vine copulas, and kernel density estimates. These models provide more robust detection thresholds in the presence of outliers and non-linear dependencies.

---

### **3.5 Smoothing and Temporal Dynamics**
**Description:**  
Explain how smoothing techniques (e.g., moving average, EWMA, LSTM memory) improve detection reliability.

**Example:**  
> Detection decisions benefit from temporal smoothing of individual alarms or scores. We apply exponentially weighted moving averages (EWMA) and temporal convolutional smoothing to reduce false alarms and emphasize persistent deviations. In neural settings, recurrent layers learn temporal dependencies implicitly.

---

### **3.6 Explainability and Variable Contribution**
**Description:**  
Discuss your method to identify which variable or group of variables contributed to the detection.

**Example:**  
> Upon detecting an anomaly, the system computes contribution scores by decomposing the residual or distance measure per feature. In the case of autoencoders, reconstruction error is used to rank variable contributions. In statistical models, variable-wise partial likelihood or local Shapley values are applied.

---

### **3.7 Evaluation Strategy**
**Description:**  
Summarize how you validate your method: datasets, metrics, baselines.

**Example:**  
> The evaluation uses both synthetic and real-world datasets. Performance is measured via detection delay, false alarm rate, precision-recall curves, and stability under drift. Baselines include Hotelling’s $T^2$ , multivariate EWMA, and state-of-the-art unsupervised anomaly detectors.

---

## **4. Distributional Assumptions and Extensions**

### **4.1 Importance of Distributional Assumptions**
**Description:**  
Reinforce the role of distributional assumptions in control charts, especially for threshold calculation.

**Example:**  
> Control limits in most multivariate charts are derived from known distributions, such as the chi-squared for Hotelling’s $T^2$. However, real process data often deviate from these assumptions, leading to incorrect thresholds and missed anomalies.

---

### **4.2 Alternative Parametric Distributions**
**Description:**  
Present alternatives like multivariate t-distribution, skew-normal, generalized hyperbolic.

**Example:**  
> The multivariate Student-t distribution provides robustness to heavy-tailed data. The skew-normal and generalized hyperbolic families capture asymmetries and tail behavior observed in industrial time series. These models extend the control chart methodology to more realistic data settings.

---

### **4.3 Copula-Based Models**
**Description:**  
Introduce copulas for modeling dependence separately from marginals. Explain Gaussian vs. vine copulas.

**Example:**  
> Copulas allow flexible modeling of joint distributions by separating marginal behavior from dependence structure. Gaussian copulas are simple but miss tail dependence; vine copulas adapt better to hierarchical or asymmetric dependencies, enabling more accurate joint modeling in control applications.

---

### **4.4 Empirical and Nonparametric Distributions**
**Description:**  
Introduce methods like kernel density estimates or empirical quantile-based thresholds.

**Example:**  
> In cases where parametric models fail, empirical control limits can be derived from historical score distributions. Kernel density estimation enables smooth approximation of the score distribution, facilitating adaptive thresholding without strong parametric assumptions.

---

### **4.5 Impact on Control Limits and False Alarms**
**Description:**  
Show how distribution choice affects control thresholds, sensitivity, and false alarm rate.

**Example:**  
> Incorrect assumptions—e.g., using a normal model on heavy-tailed data—inflate the false alarm rate or miss significant shifts. By tailoring the distribution model to the data, the control limits become more reliable, enabling better balance between sensitivity and specificity.

---


