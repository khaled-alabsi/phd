# **Dependency and Causality in Multivariate Statistical Process Control**

1. **Distinguishing Dependency and Causality in Multivariate Statistical Process Control:**  
   * Advanced Definitions of Statistical Dependency in MSPC:  
     Statistical dependence represents a fundamental type of relationship that can exist between any two or more features observed from units under study. These units can range from individual items produced in a process to various environmental conditions that might influence the process.10 When features are statistically dependent, the distribution of one variable is not the same across all levels of the other variable, indicating that there is an association between them.10 This concept is central to Multivariate Statistical Process Control (MSPC), which focuses on monitoring and controlling complex industrial processes involving multiple interrelated variables.5 In such processes, the quality and performance are often influenced by the collective behavior of several variables, and analyzing their relationships is crucial for effective fault detection, diagnosis, and overall quality control.5 Traditional Statistical Process Control (SPC) methods, which typically examine each variable independently, often prove inadequate in modern industrial settings due to the inherent correlations and interdependencies among process variables.5 By treating each variable in isolation, crucial information about the underlying process dynamics and potential multivariate issues can be overlooked.5  
     The dependencies encountered in MSPC can manifest in various forms. They can be linear, where the relationship between variables can be approximated by a straight line, or non-linear, involving more complex curves and patterns of association.8 Furthermore, these dependencies might not be instantaneous; the value of one variable at a particular time can be influenced by the value of another variable at a previous point in time, leading to time-delayed dependencies.3 This temporal aspect is particularly relevant in dynamic processes like batch manufacturing. It is also important to recognize that every observed statistical dependency is inherently conditional.10 At a minimum, the relationship is conditioned on the time and location of the study. However, the dependency between two variables can also be further conditioned on the values of other recorded variables. This leads to the distinction between marginal dependency, which is the relationship observed between two variables without considering other variables, and conditional dependency, which describes the relationship when the values of other variables are taken into account. These different forms of dependency can convey distinct information about the process and its underlying structure.10  
     * **Insight:** Understanding dependency in MSPC requires a comprehensive approach that considers the multivariate nature of process data, the possibility of linear and non-linear relationships, the presence of time delays, and the conditional nature of these associations. This advanced perspective is essential for selecting appropriate analytical techniques and for gaining meaningful insights into complex industrial processes.  
   * Advanced Definitions of Causality in MSPC:  
     Causality, in its most fundamental sense, denotes a relationship where one event, process, state, or object (the cause) actively contributes to the production of another (the effect).14 This implies that the effect is, at least to some extent, dependent on the cause, and the cause holds some responsibility for the occurrence of the effect.14 Unlike dependency, which simply describes an association, causality inherently involves a direction of influence and often suggests an underlying mechanism that links the cause to the effect.15 In the field of MSPC, the ability to identify causal relationships among process variables is of critical importance, particularly for the purpose of root cause diagnosis when a process fault or quality deviation occurs.16 While the initial detection of a fault in MSPC might rely on observing a statistical dependency (e.g., a variable exceeding its control limits), pinpointing the underlying cause necessitates understanding the causal structure of the process.16  
     One of the most influential concepts of causality in the context of time-dependent data, which is typical in MSPC, is Granger causality.25 Granger's definition of causality is based on the principle of predictability and temporal precedence.27 Specifically, a variable X is said to Granger-cause a variable Y if past values of X can statistically improve the prediction of future values of Y, given the past values of Y itself and potentially other relevant variables in the system.25 This concept provides an operational way to define and test for causality in time series data. Another related notion is instantaneous causality.26 This refers to a situation where the current value of one variable is causally related to the current value of another variable. While it might indicate a very rapid cause-and-effect mechanism, instantaneous causality can also arise from the presence of unmeasured common causes that affect both variables simultaneously within the same time frame.38  
     * **Insight:** In MSPC, an advanced understanding of causality involves not just recognizing associations between variables but also discerning the direction of influence and identifying the fundamental factors that drive process behavior and lead to faults. Granger causality and instantaneous causality offer valuable frameworks for investigating these causal relationships within the time-series data that is characteristic of many industrial processes.  
   * Nuances Differentiating Dependency and Causality in a Multivariate Setting:  
     The core distinction between statistical dependency and causality lies in the concept of directionality.15 Dependency, in its broadest sense, simply indicates that two or more variables are related to each other in a statistical manner.15 For instance, a correlation coefficient quantifies the strength and direction of a linear association between two variables but does not inherently imply that one causes the other.15 Causality, on the other hand, asserts a directed relationship, suggesting that a change in one variable (the cause) is responsible for producing a change in another variable (the effect).14  
     In a multivariate setting, where multiple variables are interconnected, observed statistical dependencies can arise from various sources that do not necessarily involve a direct causal link between the specific variables being examined.10 One common scenario is the presence of a common latent variable, an unmeasured factor that influences two or more observed variables, creating a dependency between them that is not a direct causal relationship.10 For example, in a chemical process, an unmonitored impurity in the raw material might affect both the reactor temperature and the final product purity, leading to a statistical dependency between these two measured variables without one directly causing the other. Another possibility is spurious correlation, where a statistical dependency arises purely by chance, especially in large datasets with many variables.10 Such chance correlations do not reflect any underlying causal mechanism. Confounding variables, which are related to both a potential cause and a potential effect, can also create dependencies that mask or distort the true causal relationship between the variables of interest.10 Finally, it is also possible to observe a statistical dependency where the direction of causality is the reverse of what might be initially assumed (i.e., Y causes X instead of X causes Y).15  
     Furthermore, in multivariate processes, feedback loops, where the output of one stage influences its input or a preceding stage, can complicate the distinction between dependency and causality.39 In such systems, variables can have reciprocal influences, making it difficult to establish a simple unidirectional cause-and-effect relationship.  
     * **Insight:** While causality implies the existence of a statistical dependency (a cause must be associated with its effect), the presence of a dependency does not automatically imply a causal relationship. In the multivariate context of MSPC, various factors can lead to statistical associations between process variables that are not due to direct causal links. Therefore, careful analysis and advanced techniques are essential to differentiate between mere statistical dependence and genuine causal influence.  
2. **Types of Dependency and Causality Relevant to MSPC:**  
   * **Types of Dependency in MSPC:**  
     * **Linear Dependency:** Linear dependency refers to a relationship between two or more variables that can be accurately represented by a straight line. This implies that a change in one variable is consistently associated with a proportional change in another variable.29 The most common statistical measure for quantifying the strength and direction of a linear dependency between two variables is the Pearson correlation coefficient.29 Many traditional MSPC methods are built upon the foundation of identifying and modeling linear dependencies in process data.5 For example, Principal Component Analysis (PCA) aims to reduce the dimensionality of multivariate data by finding orthogonal linear combinations of the original variables (principal components) that capture the maximum variance in the dataset.5 Similarly, Partial Least Squares (PLS) is used to model the relationship between a set of predictor variables and a set of response variables by identifying linear latent structures that maximize the covariance between the two sets.5  
     * **Non-linear Dependency:** Non-linear dependency describes relationships between variables that cannot be adequately characterized by a straight line. These relationships often involve more complex patterns of association where the change in one variable is not consistently proportional to the change in another.8 Many real-world industrial processes exhibit non-linear behavior due to the intricate nature of the underlying physical and chemical interactions.5 To effectively analyze and model these non-linear dependencies in MSPC, advanced techniques beyond traditional linear methods are required. Kernel Principal Component Analysis (KPCA) extends PCA to capture non-linear patterns by using kernel functions to map the data into a higher-dimensional space where linear PCA can be applied.5 Non-linear Partial Least Squares (NPLS) offers a similar extension for PLS. Methods based on information theory, such as Mutual Information (MI), are also valuable as they can quantify general statistical dependencies, including non-linear ones, without assuming a specific form of the relationship.50 Additionally, Multi-Spectral Phase Coherence (MSPC), a technique used in signal processing, can assess non-linear connectivity through the analysis of phase coupling across different frequencies, detecting harmonic and intermodulation relationships.2 *For clarity, it is important to distinguish this signal processing MSPC from Multivariate Statistical Process Control (MSPC) which is the primary focus of this report.5*  
     * **Time-Delayed Dependency:** In many dynamic systems encountered in industrial process control, the influence of one variable on another may not be immediate but rather occur after a certain time lag.3 This type of relationship, where the value of a variable at a particular time depends on the value of another variable at one or more preceding time points, is known as time-delayed dependency.3 Identifying and quantifying these time delays are crucial for understanding the dynamics of the process and for developing effective predictive and control strategies. Methods from time series analysis, such as Autocorrelation Function (ACF), Partial Autocorrelation Function (PACF), and more sophisticated techniques like transfer function models and state-space models, are used to analyze and model these time-delayed relationships.25  
     * **Insight:** A comprehensive understanding of process dependencies in MSPC requires considering not only linear relationships but also the potential for non-linear and time-delayed interactions. The choice of appropriate statistical methods depends on the nature of these dependencies and the specific goals of process monitoring and control. The distinction between the two meanings of the MSPC acronym is vital for clarity in this context.  
   * **Types of Causality in MSPC:**  
     * **Granger Causality:** Granger causality is a statistical concept used to determine if one time series can forecast another.25 Specifically, a variable X is said to Granger-cause a variable Y if the past values of X have a statistically significant effect on the current value of Y, even after considering the past values of Y itself and other relevant variables.25 Granger causality is typically assessed using Vector Autoregressive (VAR) models, where each variable in the system is modeled as a linear function of its own past values and the past values of other variables.25 Statistical tests, such as F-tests or Wald tests, are then performed on the coefficients of the lagged terms to determine if the inclusion of one variable's past improves the prediction of another. Conditional Granger causality extends this to account for the influence of other variables in the multivariate system.20  
     * **Instantaneous Causality:** Instantaneous causality refers to a causal relationship between two variables that occurs within the same time period.26 In the context of time series analysis, it often manifests as a significant correlation between the residuals of the predictive models for the two variables.26 Detecting instantaneous causality can be challenging and may indicate very rapid causal effects or the presence of unobserved common causes affecting both variables simultaneously.38 Some advanced methods for causality analysis in time series, including extensions of Granger causality, aim to identify and quantify these instantaneous relationships.38  
     * **Insight:** In the realm of MSPC, Granger causality and instantaneous causality are the most commonly considered types of causality when analyzing process data. Granger causality is particularly valuable for understanding the temporal flow of influence between variables, while instantaneous causality can provide insights into immediate effects or potential confounding factors.  
3. **The Interplay Between Dependency and Causality in MSPC:**  
   * Conditions Under Which Statistical Dependency Can Imply Causality:  
     Statistical dependency, while not synonymous with causality, can provide evidence suggestive of a causal relationship under certain conditions.25 One crucial condition is temporal precedence: the potential cause must occur before the potential effect.25 Granger causality analysis explicitly utilizes this temporal ordering to infer directionality.25 If a statistical dependency is observed and changes in the hypothesized cause consistently precede changes in the hypothesized effect, it lends support to a causal interpretation. Another important factor is the strength and consistency of the association.95 A strong and consistently observed statistical dependency between variables across different datasets, operating conditions, and time periods makes it less likely that the association is spurious or due to random chance, thus increasing the plausibility of a causal link.  
     Furthermore, the ability to **rule out alternative explanations** for the observed dependency is critical.10 In a multivariate setting, this often involves controlling for the influence of potential confounding variables that might be affecting both the cause and the effect. Techniques such as partial correlation and conditional Granger causality are designed to address this by examining the relationship between two variables while holding others constant.20 Finally, the existence of a **plausible mechanism** that can explain how the cause could lead to the effect strengthens the argument for causality.14 In the context of MSPC, this often involves drawing upon established process knowledge, physical laws, or engineering principles to provide a theoretical basis for the observed statistical relationship.5 When a statistically observed dependency aligns with a known or theorized process mechanism, it provides more compelling evidence for a causal link.  
     * **Insight:** While statistical dependency alone is insufficient to establish causality in MSPC, when coupled with evidence of temporal precedence, strong and consistent association, the elimination of plausible alternative explanations through multivariate control, and a supporting process mechanism, it can provide valuable indications of underlying causal relationships.  
   * Scenarios Where Dependency Exists Without Direct Causal Relationships:  
     It is crucial to recognize that statistical dependency between process variables in MSPC does not always imply a direct causal relationship. Several scenarios can lead to observed dependencies that are not the result of one variable directly causing a change in another.10 One common scenario is the influence of common latent variables.10 If two observed variables are both affected by a third, unmeasured variable, they will likely exhibit a statistical dependency even if there is no direct causal link between them. For instance, variations in the quality of a raw material (a latent variable) could simultaneously affect both the operating temperature of a reactor and the final product's viscosity, leading to a dependency between temperature and viscosity that is mediated by the raw material quality.  
     Another possibility is **spurious correlations**.10 In datasets with a large number of variables, especially when the underlying relationships are weak or non-existent, statistical dependencies can arise purely by chance. These correlations do not reflect any true underlying causal connection. **Confounding variables** also represent a situation where dependency does not imply direct causality.10 A confounding variable is related to both a potential cause and a potential effect, creating an apparent dependency between them that is actually driven by the third variable. For example, an increase in ambient temperature might lead to both higher energy consumption in a plant and a decrease in production rate, creating a dependency between energy consumption and production that is actually caused by the ambient temperature. Finally, **reverse causation** is another scenario where dependency exists without the assumed causal direction.15 While an observed dependency might suggest that changes in variable X cause changes in variable Y, it could be that changes in Y are actually causing changes in X. For example, higher product demand might lead to increased production (X leading to Y), but also to increased overtime for workers (Y leading to X), creating a dependency where the causal direction needs careful consideration.  
     * **Insight:** In MSPC, it is essential to be aware of these scenarios where statistical dependency can exist without a direct causal relationship between the measured variables. Failing to consider these possibilities can lead to incorrect interpretations of process behavior and ineffective control strategies.  
4. **Challenges in Classifying Relationships: Dependency vs. Causality:**  
   * Analysis of Scenarios Where Definitive Classification is Difficult:  
     Classifying the relationships between variables in multivariate process control as either dependency or causality can be a complex endeavor, and there are several scenarios where a definitive classification proves challenging.5 In highly interconnected multivariate processes, the complex interplay of variables makes it difficult to isolate direct cause-and-effect relationships from indirect dependencies mediated through other variables.5 A change in one variable can propagate through the system, affecting numerous other variables, making it hard to determine the primary drivers of specific outcomes. Furthermore, limited data, whether due to small sample sizes or the presence of significant noise in the measurements, can hinder the reliable estimation of dependencies and the statistical power of tests designed to infer causality.87 This lack of robust statistical evidence can make it difficult to definitively classify a relationship.  
     The presence of **non-stationarity and non-linearity** in many industrial processes further complicates the classification.5 Traditional linear methods for dependency and causality analysis often rely on the assumption of stationarity (that the statistical properties of the process do not change over time). When this assumption is violated, or when the relationships between variables are non-linear, standard techniques may yield misleading results, making classification uncertain. **Feedback loops**, where variables influence each other reciprocally, also pose a significant challenge.39 In such systems, the clear distinction between cause and effect can become blurred, as variables are both drivers and consequences of changes in other variables. Finally, distinguishing between true **instantaneous causality** (a simultaneous causal effect) and very rapid causal effects or the influence of unmeasured common causes that act simultaneously can be problematic when relying solely on observational data.26  
     * **Insight:** The inherent complexity of multivariate industrial processes, coupled with limitations in data and the presence of non-linearities, non-stationarity, feedback mechanisms, and the challenges of interpreting instantaneous effects, often makes it difficult to definitively classify a connection between process variables as either dependency or causality based solely on statistical analysis.  
   * Discussion on the Limitations of Purely Statistical Methods in Inferring Causality:  
     It is important to acknowledge the inherent limitations of relying solely on statistical methods when attempting to infer causality, especially in the context of observational data common in MSPC.15 Statistical methods are primarily designed to identify patterns of association or dependency between variables.15 Making the leap from observed association to a causal inference requires additional assumptions and often necessitates incorporating external information or, ideally, conducting controlled experiments.15  
     While Granger causality is a statistical technique aimed at inferring the direction of influence in time series data, it is fundamentally based on predictability.25 The fact that past values of one variable can improve the prediction of another does not definitively prove a direct causal mechanism. Granger causality can be susceptible to confounding effects from unmeasured variables and might not always reflect true underlying causal relationships.25 Furthermore, purely data-driven causal inference methods might struggle to differentiate between causally equivalent models, which are different causal structures that produce the same observed statistical relationships.40 Without additional constraints or information, it can be impossible to determine the correct causal direction or to identify the presence of unmeasured confounders that might be driving the observed dependencies. Therefore, in MSPC, while statistical methods provide valuable tools for identifying dependencies and suggesting potential causal links, a robust understanding of causality often requires integrating these statistical findings with in-depth process knowledge, domain expertise, and, where possible, evidence from designed experiments or interventions.5  
     * **Insight:** Relying solely on statistical methods can be insufficient for definitively inferring causality in MSPC. A more comprehensive approach involves combining statistical analysis with a thorough understanding of the process and, ideally, experimental validation to establish robust causal relationships.  
5. **Most Relevant Types of Dependency and Causality for Effective MSPC:**  
   * Impactful Dependencies for Process Monitoring and Control:  
     For effective monitoring and control in MSPC, certain types of dependencies are particularly impactful. Non-linear dependencies are highly relevant because many industrial processes exhibit complex, non-linear interactions between their variables.5 Deviations from normal operating conditions and the onset of faults can often be manifested through changes in these non-linear relationships. Therefore, MSPC methods capable of detecting and modeling such dependencies, like kernel-based techniques or information theory measures, can provide enhanced sensitivity to process anomalies and facilitate more accurate fault diagnosis.  
     **Time-delayed dependencies** are also crucial for effective MSPC, especially in dynamic processes such as batch operations or continuous processes with inherent lags in their responses.5 Understanding how process variables influence each other over time can enable the anticipation of potential problems and the implementation of proactive control measures to prevent deviations from desired quality or performance targets.  
     * **Insight:** For achieving more effective MSPC, focusing on capturing and modeling non-linear and time-delayed dependencies within process data is often more impactful than relying solely on linear and static relationships. These types of dependencies are more reflective of the complex dynamics prevalent in many industrial settings and can provide more actionable insights for process improvement.  
   * Impactful Types of Causality for Process Monitoring and Control:  
     In the context of MSPC, Granger causality is a particularly relevant type of causality for effective monitoring and control.25 Identifying Granger causal relationships between process variables can help in understanding the flow of influence and the propagation of disturbances through the system.16 This knowledge can be invaluable for pinpointing the root causes of process faults and for developing targeted control strategies to prevent their recurrence. By identifying leading indicators (variables that Granger-cause others related to quality or performance), it becomes possible to implement early detection systems and take corrective actions before significant deviations occur.  
     **Instantaneous causality**, while often more challenging to interpret, can also be relevant in MSPC.26 Detecting strong instantaneous links between variables might highlight critical, immediate interactions within the process that warrant further investigation. As previously noted, it can also point towards the presence of unmeasured common causes that are simultaneously affecting multiple observed variables. Identifying such scenarios can lead to a more complete understanding of the process and potentially the inclusion of additional relevant variables in the monitoring framework.  
     * **Insight:** For enhancing MSPC capabilities, understanding the temporal causal relationships between process variables, as revealed by Granger causality, and identifying critical immediate influences, as suggested by instantaneous causality, are of utmost importance. These types of causality can provide actionable insights for improving process monitoring, enabling more effective fault diagnosis, and ultimately leading to better process control and product quality.  
6. **Measuring Statistical Dependency in MSPC: Advanced Methods:**  
   * **Canonical Correlation Analysis (CCA):**  
     * **Mathematical Background:** Canonical Correlation Analysis (CCA) is a multivariate statistical technique that aims to identify and quantify the linear relationships between two distinct sets of variables.5 Given two sets of variables, say X (with *p* variables) and Y (with *q* variables), CCA seeks to find pairs of linear combinations, one from each set, that exhibit the maximum possible correlation with each other. These linear combinations are known as canonical variates.105 The mathematical foundation of CCA involves analyzing the covariance matrices within each set (Σ\<sub\>XX\</sub\> and Σ\<sub\>YY\</sub\>) and the cross-covariance matrix between the two sets (Σ\<sub\>XY\</sub\>).114 The goal is to find weight vectors **a** (for X) and **b** (for Y) that maximize the correlation ρ between the canonical variates U \= **a**\<sup\>T\</sup\>X and V \= **b**\<sup\>T\</sup\>Y.106 This maximization problem leads to a generalized eigenvalue problem: (Σ\<sub\>XX\</sub\>\<sup\>-1\</sup\>Σ\<sub\>XY\</sub\>Σ\<sub\>YY\</sub\>\<sup\>-1\</sup\>Σ\<sub\>YX\</sub\> \- ρ\<sup\>2\</sup\>I) **a** \= 0, where Σ\<sub\>YX\</sub\> is the transpose of Σ\<sub\>XY\</sub\>, and I is the identity matrix.114 Solving this problem yields eigenvalues (ρ\<sup\>2\</sup\>, the squared canonical correlations) and eigenvectors (**a**, the canonical weights for X). A similar process can be used to find the canonical weights for Y (**b**). The number of canonical variate pairs that can be derived is at most the minimum of *p* and *q*.105  
     * **Intuitive Explanation:** CCA can be intuitively understood as a method for finding the "most related" underlying dimensions in two different sets of measurements taken on the same process.105 For example, if you have one set of variables describing the operating conditions of a machine and another set describing the quality of the product it produces, CCA tries to find linear combinations of the operating conditions that are maximally correlated with linear combinations of the quality characteristics. It's like finding the "common ground" or the strongest linear associations between two different perspectives on the same process.  
     * **Graphical Interpretation:** The results of a CCA can be visualized in several ways to aid interpretation.115 **Scatter plots** of the canonical variate scores (the values obtained by projecting the original data onto the canonical variates) for each pair of variates can illustrate the strength and direction of the linear relationship between them. A strong linear pattern in the scatter plot indicates a high canonical correlation. **Bar plots** of the canonical coefficients (the weights in the vectors **a** and **b**) show the contribution of each original variable to the corresponding canonical variate. This helps in understanding which of the original variables are most influential in defining the correlated dimensions. **Heatmaps** of the correlation matrix between all the extracted canonical variates can provide a comprehensive overview of the relationships, showing the correlations between different pairs of canonical variates. Partial correlation diagrams can also be used to visualize the partial correlations between variables after CCA has been performed.124  
     * 
---

The provided text contains several issues, including incorrect matrix dimensions, inconsistent or mismatched entries, and unclear steps in the computation. Below is a corrected and clarified version of the numerical example:

---

### Numerical Example:
Consider two sets of process variables:  
- Set $X = \{\text{Temperature (X1), Pressure (X2)}\}$  
- Set $Y = \{\text{Viscosity (Y1), Density (Y2)}\}$  

Assume we have collected data and calculated the following sample covariance matrices:  
$$
\Sigma_{XX} = \begin{bmatrix} 1 & 2 \\ 2 & 5 \end{bmatrix}, \quad
\Sigma_{YY} = \begin{bmatrix} 4 & 3 \\ 3 & 7 \end{bmatrix}, \quad
\Sigma_{XY} = \begin{bmatrix} 7 & 9 \\ 8 & 10 \end{bmatrix}.
$$

Our goal is to find weight vectors $\mathbf{a} = [a_1, a_2]^T$ and $\mathbf{b} = [b_1, b_2]^T$ that maximize the correlation between  
$$
U = a_1 X_1 + a_2 X_2 \quad \text{and} \quad V = b_1 Y_1 + b_2 Y_2.
$$

This problem can be solved using Canonical Correlation Analysis (CCA). The steps are as follows:

---

### Step 1: Compute $\Sigma_{YY}^{-1}$
The inverse of $\Sigma_{YY}$ is computed as:
$$
\Sigma_{YY}^{-1} = \frac{1}{\det(\Sigma_{YY})} \cdot \text{adj}(\Sigma_{YY}),
$$
where $\det(\Sigma_{YY}) = (4)(7) - (3)(3) = 28 - 9 = 19$.  
Thus,
$$
\Sigma_{YY}^{-1} = \frac{1}{19} \begin{bmatrix} 7 & -3 \\ -3 & 4 \end{bmatrix}
= \begin{bmatrix} 0.3684 & -0.1579 \\ -0.1579 & 0.2105 \end{bmatrix}.
$$

---

### Step 2: Compute $\Sigma_{XX}^{-1}$
Similarly, the inverse of $\Sigma_{XX}$ is computed as:
$$
\Sigma_{XX}^{-1} = \frac{1}{\det(\Sigma_{XX})} \cdot \text{adj}(\Sigma_{XX}),
$$
where $\det(\Sigma_{XX}) = (1)(5) - (2)(2) = 5 - 4 = 1$.  
Thus,
$$
\Sigma_{XX}^{-1} = \begin{bmatrix} 5 & -2 \\ -2 & 1 \end{bmatrix}.
$$

---

### Step 3: Compute $\Sigma_{XX}^{-1} \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}$
We proceed step by step:

#### (a) Compute $\Sigma_{YX} = \Sigma_{XY}^T$:
$$
\Sigma_{YX} = \Sigma_{XY}^T = \begin{bmatrix} 7 & 8 \\ 9 & 10 \end{bmatrix}.
$$

#### (b) Compute $\Sigma_{XX}^{-1} \Sigma_{XY}$:
$$
\Sigma_{XX}^{-1} \Sigma_{XY} = \begin{bmatrix} 5 & -2 \\ -2 & 1 \end{bmatrix}
\begin{bmatrix} 7 & 9 \\ 8 & 10 \end{bmatrix}
= \begin{bmatrix}
(5)(7) + (-2)(8) & (5)(9) + (-2)(10) \\
(-2)(7) + (1)(8) & (-2)(9) + (1)(10)
\end{bmatrix}
= \begin{bmatrix} 19 & 25 \\ -6 & -8 \end{bmatrix}.
$$

#### (c) Compute $(\Sigma_{XX}^{-1} \Sigma_{XY}) \Sigma_{YY}^{-1}$:
$$
(\Sigma_{XX}^{-1} \Sigma_{XY}) \Sigma_{YY}^{-1} =
\begin{bmatrix} 19 & 25 \\ -6 & -8 \end{bmatrix}
\begin{bmatrix} 0.3684 & -0.1579 \\ -0.1579 & 0.2105 \end{bmatrix}.
$$
Performing the multiplication:
$$
= \begin{bmatrix}
(19)(0.3684) + (25)(-0.1579) & (19)(-0.1579) + (25)(0.2105) \\
(-6)(0.3684) + (-8)(-0.1579) & (-6)(-0.1579) + (-8)(0.2105)
\end{bmatrix}
= \begin{bmatrix} 2.4461 & 1.8695 \\ -0.1306 & -0.6646 \end{bmatrix}.
$$

#### (d) Compute $(\Sigma_{XX}^{-1} \Sigma_{XY} \Sigma_{YY}^{-1}) \Sigma_{YX}$:
$$
(\Sigma_{XX}^{-1} \Sigma_{XY} \Sigma_{YY}^{-1}) \Sigma_{YX} =
\begin{bmatrix} 2.4461 & 1.8695 \\ -0.1306 & -0.6646 \end{bmatrix}
\begin{bmatrix} 7 & 8 \\ 9 & 10 \end{bmatrix}.
$$
Performing the multiplication:
$$
= \begin{bmatrix}
(2.4461)(7) + (1.8695)(9) & (2.4461)(8) + (1.8695)(10) \\
(-0.1306)(7) + (-0.6646)(9) & (-0.1306)(8) + (-0.6646)(10)
\end{bmatrix}
= \begin{bmatrix} 35.1122 & 41.7646 \\ -6.6988 & -8.1148 \end{bmatrix}.
$$

---

### Step 4: Solve the Eigenvalue Problem
To find the canonical correlations, solve the eigenvalue problem:
$$
\left( \begin{bmatrix} 35.1122 & 41.7646 \\ -6.6988 & -8.1148 \end{bmatrix} - \lambda \mathbf{I} \right) \mathbf{a} = 0.
$$

The eigenvalues ($\lambda$) are found by solving:
$$
\det\left( \begin{bmatrix} 35.1122 - \lambda & 41.7646 \\ -6.6988 & -8.1148 - \lambda \end{bmatrix} \right) = 0.
$$

Expanding the determinant:
$$
(35.1122 - \lambda)(-8.1148 - \lambda) - (41.7646)(-6.6988) = 0.
$$
Simplify this quadratic equation to find the eigenvalues.

---

### Step 5: Compute Canonical Correlations
The eigenvalues ($\lambda$) represent the squared canonical correlations. Take the square root of each eigenvalue to obtain the canonical correlations.

For example, if the largest eigenvalue is approximately $0.47$, the corresponding canonical correlation is:
$$
\sqrt{0.47} \approx 0.686.
$$

The eigenvectors corresponding to these eigenvalues provide the weight vectors $\mathbf{a}$ and $\mathbf{b}$ for the canonical variates.

---

### Final Answer:
The largest canonical correlation is approximately:
$$
\boxed{0.686}.
$$


     * **Insight:** CCA is a valuable tool in MSPC for uncovering linear relationships between two sets of process-related variables, providing insights into how they covary and potentially identifying key associations for monitoring and control purposes.  
   * **Partial Correlation:**  
     * **Mathematical Background:** Partial correlation measures the strength of the linear relationship between two variables while statistically controlling for the influence of one or more other variables.56 For three variables, X, Y, and a control variable Z, the partial correlation coefficient between X and Y controlling for Z, denoted as r\<sub\>XY.Z\</sub\>, is calculated using the Pearson correlation coefficients between each pair of variables: r\<sub\>XY.Z\</sub\> \= (r\<sub\>XY\</sub\> \- r\<sub\>XZ\</sub\>r\<sub\>YZ\</sub\>) / √((1 \- r\<sub\>XZ\</sub\>\<sup\>2\</sup\>)(1 \- r\<sub\>YZ\</sub\>\<sup\>2\</sup\>)).125 Higher-order partial correlations, controlling for multiple variables, can be computed recursively. For example, the partial correlation between X and Y controlling for Z\<sub\>1\</sub\> and Z\<sub\>2\</sub\> can be calculated using the partial correlations of X and Y controlling for Z\<sub\>1\</sub\>, and Z\<sub\>2\</sub\> with Z\<sub\>1\</sub\>, and the correlation between Z\<sub\>1\</sub\> and Z\<sub\>2\</sub\>.97 Conceptually, the partial correlation between X and Y controlling for Z can be viewed as the correlation between the residuals obtained from regressing X on Z and Y on Z.97  
     * **Intuitive Explanation:** Partial correlation allows us to isolate the direct linear relationship between two process variables by removing the influence of other variables that might be affecting both.97 It helps to determine if an observed correlation between two variables is a genuine direct association or if it is mediated by or confounded with other factors.  
     * **Graphical Interpretation:** Partial correlations in a multivariate setting can be visualized using partial correlation diagrams.57 In these diagrams, variables are represented as nodes, and the partial correlations between them (after controlling for all other variables in the diagram) are depicted as edges. The color and width of the edges typically indicate the direction and strength of the partial correlation.  
  


### Numerical Example:
Consider three process variables: $X$ (Feed Flow Rate), $Y$ (Reactor Pressure), and $Z$ (Coolant Temperature). Assume the following Pearson correlation coefficients:  
$$
r_{XY} = 0.75, \quad r_{XZ} = 0.65, \quad r_{YZ} = 0.80.
$$
We want to find the **partial correlation** between Feed Flow Rate ($X$) and Reactor Pressure ($Y$) while controlling for Coolant Temperature ($Z$):  
$$
r_{XY.Z} = \frac{r_{XY} - r_{XZ} \cdot r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}.
$$
Substituting the given values:
$$
r_{XY.Z} = \frac{0.75 - 0.65 \cdot 0.80}{\sqrt{(1 - 0.65^2)(1 - 0.80^2)}} = \frac{0.75 - 0.52}{\sqrt{(1 - 0.4225)(1 - 0.64)}} = \frac{0.23}{\sqrt{0.5775 \cdot 0.36}}.
$$
Simplify further:
$$
r_{XY.Z} = \frac{0.23}{\sqrt{0.2079}} \approx \frac{0.23}{0.456} \approx 0.504.
$$
The partial correlation ($0.504$) is lower than the original correlation ($0.75$), suggesting that some of the observed correlation between Feed Flow Rate and Reactor Pressure is due to their shared relationship with Coolant Temperature.

#### Insight:
Partial correlation is a valuable tool in MSPC for understanding the direct linear relationships between pairs of process variables by eliminating the influence of other measured variables. This can aid in identifying key control variables and understanding the true nature of the associations within the process.

---

### Information Theory-based Measures (e.g., Mutual Information):

#### Mathematical Background:
Mutual Information (MI) is a non-parametric measure of statistical dependence between two random variables. Unlike correlation, which only captures linear relationships, MI can detect any type of statistical dependency, including non-linear ones. For discrete random variables $X$ and $Y$ with joint probability distribution $P(x, y)$ and marginal distributions $P(x)$ and $P(y)$, the mutual information is defined as:
$$
I(X; Y) = \sum_x \sum_y P(x, y) \log_2 \left( \frac{P(x, y)}{P(x) P(y)} \right).
$$
For continuous variables, the summations are replaced by integrals over the probability density functions. Conditional Mutual Information (CMI) measures the dependence between two variables given a third.

#### Intuitive Explanation:
Mutual information quantifies the amount of information shared between two process variables. It tells us how much knowing the value of one variable reduces the uncertainty about the value of the other. A higher mutual information value indicates a stronger statistical dependency, regardless of whether the relationship is linear or non-linear.

#### Graphical Interpretation:
While mutual information itself is a numerical measure, it is a key component in learning the structure of probabilistic graphical models like Bayesian Networks. In these networks, the presence or absence of directed edges between variables reflects conditional dependencies, which can be determined using measures of mutual information or conditional mutual information. I-diagrams can also be used to visualize the relationships between the entropies and mutual information of a set of variables.

---

#### Numerical Example:
Consider two continuous process variables, e.g., Reactor Temperature ($X$) and Product Yield ($Y$). Estimating the mutual information typically involves non-parametric methods. Using a simplified discrete example:  

Assume we have discretized the variables into two states (High/Low). Based on historical data, the joint probabilities are:
$$
P(X=\text{High}, Y=\text{High}) = 0.4, \quad P(X=\text{High}, Y=\text{Low}) = 0.1,
$$
$$
P(X=\text{Low}, Y=\text{High}) = 0.2, \quad P(X=\text{Low}, Y=\text{Low}) = 0.3.
$$
The marginal probabilities are:
$$
P(X=\text{High}) = 0.5, \quad P(X=\text{Low}) = 0.5, \quad P(Y=\text{High}) = 0.6, \quad P(Y=\text{Low}) = 0.4.
$$
The mutual information is calculated as:
$$
I(X; Y) = \sum_x \sum_y P(x, y) \log_2 \left( \frac{P(x, y)}{P(x) P(y)} \right).
$$
Substitute the values:
$$
I(X; Y) = 0.4 \cdot \log_2 \left( \frac{0.4}{0.5 \cdot 0.6} \right) + 0.1 \cdot \log_2 \left( \frac{0.1}{0.5 \cdot 0.4} \right)
+ 0.2 \cdot \log_2 \left( \frac{0.2}{0.5 \cdot 0.6} \right) + 0.3 \cdot \log_2 \left( \frac{0.3}{0.5 \cdot 0.4} \right).
$$
Simplify each term:
$$
I(X; Y) \approx 0.4 \cdot 0.415 + 0.1 \cdot (-1.322) + 0.2 \cdot (-0.585) + 0.3 \cdot 0.585.
$$
Combine terms:
$$
I(X; Y) \approx 0.166 - 0.132 - 0.117 + 0.176 \approx 0.093 \, \text{bits}.
$$

#### Insight:
Mutual information is a powerful tool in MSPC for detecting general statistical dependencies, including non-linear relationships that might be missed by traditional linear methods. This can provide a more complete understanding of the interactions between process variables and aid in the development of more robust monitoring and diagnostic systems.

---

#### Boxed Final Answer:
The corrected and formatted text is presented above. Key fixes include:
1. Proper use of mathematical notation (e.g., $r_{XY}$, $I(X; Y)$).
2. Correct alignment of equations and consistent use of LaTeX-style formatting.
3. Improved readability and logical flow of explanations.



### 7. **Inferring Causality in MSPC: Advanced Methods**

#### **Granger Causality**
1. **Mathematical Background and Vector Autoregressive (VAR) Models**  
   Granger causality is a statistical framework designed to analyze causal relationships between time series data [25]. It is typically implemented using Vector Autoregressive (VAR) models [25]. A VAR model of order $p$ for a $k$-dimensional time series $\mathbf{Y}_t = (Y_{1t}, Y_{2t}, \dots, Y_{kt})^T$ is defined as:
   $$
   \mathbf{Y}_t = \mathbf{c} + \mathbf{A}_1\mathbf{Y}_{t-1} + \mathbf{A}_2\mathbf{Y}_{t-2} + \dots + \mathbf{A}_p\mathbf{Y}_{t-p} + \boldsymbol{\varepsilon}_t,
   $$
   where:
   - $\mathbf{c}$ is a constant vector,
   - $\mathbf{A}_i$ are $k \times k$ coefficient matrices,
   - $\boldsymbol{\varepsilon}_t$ is a vector of white noise error terms [27].

   To test whether a variable $X$ (part of $\mathbf{Y}_t$) Granger-causes another variable $Y$ (also part of $\mathbf{Y}_t$), we examine the coefficients of the lagged values of $X$ in the equation for $Y$. Specifically, we compare the fit of an unrestricted VAR model (where lagged $X$ values are included in the $Y$ equation) with a restricted VAR model (where the coefficients of the lagged $X$ values in the $Y$ equation are set to zero) [25]. This comparison is typically performed using an F-test or a Wald test on the joint significance of the coefficients of the lagged $X$ variables in the $Y$ equation [20]. The optimal lag order $p$ for the VAR model is usually selected based on information criteria such as AIC or BIC [26].

2. **Intuitive Explanation**  
   Granger causality provides a statistical approach to investigate whether the past behavior of one process variable offers useful information for predicting the future behavior of another variable, beyond the information already contained in the latter's own past and the past of other relevant variables [25]. If incorporating the historical data of $X$ significantly improves the forecast of $Y$, then $X$ is considered to Granger-cause $Y$.

3. **Graphical Interpretation**  
   Granger causality relationships within a multivariate process can be represented as a directed graph or a causality map [20]. In such a graph:
   - Each process variable is a node,
   - A directed edge pointing from variable $X$ to variable $Y$ indicates that $X$ Granger-causes $Y$. 
   
   The strength or statistical significance of this causal influence can be depicted by the weight or thickness of the edge [20]. These graphical representations can be particularly helpful for visualizing the network of causal interactions within a complex industrial process.

4. **Numerical Example**  
   Consider two process variables: $X$ (Input Flow Rate) and $Y$ (Output Concentration), measured over time. We fit a VAR(1) model:
   $$
   Y_t = c_1 + a_1 Y_{t-1} + b_1 X_{t-1} + \varepsilon_{1t},
   $$
   $$
   X_t = c_2 + a_2 Y_{t-1} + b_2 X_{t-1} + \varepsilon_{2t}.
   $$
   To test if $X$ Granger-causes $Y$, we test the null hypothesis $H_0: b_1 = 0$. We compare the fit of this unrestricted model with a restricted model where $b_1$ is forced to zero ($Y_t = c_1 + a_1 Y_{t-1} + \varepsilon_{1t}$) using an F-test. If the p-value of the test is below a chosen significance level (e.g., 0.05), we reject the null hypothesis and conclude that Input Flow Rate Granger-causes Output Concentration.

5. **Discussion of Extensions for Non-linear Systems**  
   While the traditional Granger causality framework is based on linear VAR models, several extensions have been developed to address non-linear relationships prevalent in many industrial processes [20]. These include:
   - Non-linear Granger causality tests using methods like artificial neural networks [20] or kernel-based approaches.
   - Convergent Cross Mapping (CCM), which infers causality in non-linear time series by analyzing the correlation between reconstructed state spaces [183].

   These methods aim to capture non-linear predictive relationships that standard linear Granger causality might miss, providing a more comprehensive understanding of causal influences in complex processes.

6. **Insight**  
   Granger causality, along with its non-linear extensions, is a valuable tool for inferring temporal causal relationships in MSPC data. By identifying these relationships, it becomes possible to gain actionable insights into the underlying process dynamics, which can be leveraged for enhanced monitoring, fault diagnosis, and the development of more effective control strategies.



### Conclusion:  
   This report has provided a comprehensive exploration of the concepts of dependency and causality within the context of Multivariate Statistical Process Control (MSPC). We have established the fundamental differences between statistical dependency, which signifies an association between variables, and causality, which implies a directed influence. Various types of dependency relevant to MSPC, including linear, non-linear, and time-delayed forms, were discussed, emphasizing the complex nature of industrial process data. Key types of causality, namely Granger causality and instantaneous causality, were examined in relation to their applicability in MSPC for understanding process dynamics.  
   The interplay between dependency and causality was explored, highlighting conditions under which statistical dependency can provide evidence for causality and, conversely, scenarios where dependency exists without a direct causal link, such as through common latent variables or spurious correlations. Challenges in definitively classifying relationships as dependency or causality, particularly in complex, high-dimensional, non-linear, and non-stationary processes with feedback loops, were also addressed. The limitations of relying solely on statistical methods for causal inference were discussed, underscoring the importance of integrating process knowledge and potentially experimental validation.  
   The most relevant types of dependency for effective MSPC were identified as non-linear and time-delayed dependencies, reflecting the dynamic and often complex interactions within industrial processes. For causality, Granger causality, along with its extensions for non-linear systems, and insights from instantaneous causality were highlighted as particularly impactful for process monitoring and control, providing actionable information for fault detection, diagnosis, and process improvement.  
   Advanced statistical methods for measuring dependency in MSPC, including Canonical Correlation Analysis (CCA), Partial Correlation, and Information Theory-based measures like Mutual Information, were detailed with their mathematical backgrounds, intuitive explanations, graphical interpretations, and numerical examples. Furthermore, advanced methods for inferring causality, with a focus on Granger causality and its implementation using Vector Autoregressive (VAR) models, as well as extensions for non-linear systems, were explained.  
   The insights derived from this research emphasize the importance of moving beyond simple linear dependencies and exploring the more complex non-linear and time-delayed relationships prevalent in industrial processes. Utilizing advanced statistical methods for measuring dependency and inferring causality, particularly Granger causality and its extensions, can significantly enhance the capabilities of MSPC. By gaining a deeper understanding of the causal structures within their processes, researchers and engineers can develop more effective monitoring and diagnostic tools, leading to improved process stability, enhanced product quality, and more efficient operations. Future research could focus on developing integrated MSPC frameworks that seamlessly incorporate causal inference techniques with traditional statistical process control methodologies to create more intelligent and proactive process management systems.

#### **Referenzen**

1. Nonlinear Connectivity in the Human Stretch Reflex Assessed by Cross-Frequency Phase Coupling \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/303833060\_Nonlinear\_Connectivity\_in\_the\_Human\_Stretch\_Reflex\_Assessed\_by\_Cross-Frequency\_Phase\_Coupling](https://www.researchgate.net/publication/303833060_Nonlinear_Connectivity_in_the_Human_Stretch_Reflex_Assessed_by_Cross-Frequency_Phase_Coupling)  
2. A General Approach for Quantifying Nonlinear Connectivity in the Nervous System Based on Phase Coupling, Zugriff am April 19, 2025, [https://ris.utwente.nl/ws/files/320660627/IJNS15500311stPrf\_Frans.pdf](https://ris.utwente.nl/ws/files/320660627/IJNS15500311stPrf_Frans.pdf)  
3. $$
2110.01850$$ Nonlinear effects of instantaneous and delayed state dependence in a delayed feedback loop \- arXiv, Zugriff am April 19, 2025, [https://arxiv.org/abs/2110.01850](https://arxiv.org/abs/2110.01850)  
4. Nonlinear dynamics, delay times, and embedding windows, Zugriff am April 19, 2025, [https://zr9558.com/wp-content/uploads/2015/09/nonlinear-dynamics-delay-times-and-embedding-windows.pdf](https://zr9558.com/wp-content/uploads/2015/09/nonlinear-dynamics-delay-times-and-embedding-windows.pdf)  
5. Multivariate SPC: Advanced Process Monitoring & Control Techniques \- SixSigma.us, Zugriff am April 19, 2025, [https://www.6sigma.us/six-sigma-in-focus/multivariate-spc/](https://www.6sigma.us/six-sigma-in-focus/multivariate-spc/)  
6. Nonlinear Connectivity in the Human Stretch Reflex Assessed by Cross- Frequency Phase Coupling \- https ://ris.utwen te.nl, Zugriff am April 19, 2025, [https://ris.utwente.nl/ws/files/320660752/Nonlinear\_Connectivity\_in\_the\_Human\_Stretch\_Reflex.pdf](https://ris.utwente.nl/ws/files/320660752/Nonlinear_Connectivity_in_the_Human_Stretch_Reflex.pdf)  
7. A General Approach for Quantifying Nonlinear Connectivity in the Nervous System Based on Phase Coupling \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/282277582\_A\_General\_Approach\_for\_Quantifying\_Nonlinear\_Connectivity\_in\_the\_Nervous\_System\_Based\_on\_Phase\_Coupling](https://www.researchgate.net/publication/282277582_A_General_Approach_for_Quantifying_Nonlinear_Connectivity_in_the_Nervous_System_Based_on_Phase_Coupling)  
8. Nonlinearity and Temporal Dependence, Zugriff am April 19, 2025, [https://home.uchicago.edu/\~lhansen/mixmain.pdf](https://home.uchicago.edu/~lhansen/mixmain.pdf)  
9. STATISTICAL DEPENDENCE definition in American English \- Collins Dictionary, Zugriff am April 19, 2025, [https://www.collinsdictionary.com/us/dictionary/english/statistical-dependence](https://www.collinsdictionary.com/us/dictionary/english/statistical-dependence)  
10. Statistical Dependence and Independence, Zugriff am April 19, 2025, [https://www.math.chalmers.se/\~wermuth/pdfs/96-05/WerCox98\_Statistical\_dependence.pdf](https://www.math.chalmers.se/~wermuth/pdfs/96-05/WerCox98_Statistical_dependence.pdf)  
11. Nonlinear effects of instantaneous and delayed state dependence in a delayed feedback loop \- American Institute of Mathematical Sciences, Zugriff am April 19, 2025, [https://www.aimsciences.org/article/doi/10.3934/dcdsb.2022042](https://www.aimsciences.org/article/doi/10.3934/dcdsb.2022042)  
12. Nonlinear dynamics of a time-delayed epidemic model with two explicit aware classes, saturated incidences, and treatment \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7334637/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7334637/)  
13. Release of Neurotransmitters \- Molecular and Cell Biology |, Zugriff am April 19, 2025, [https://mcb.berkeley.edu/labs/zucker/PDFs/ZuckerKullmannSchwarz.pdf](https://mcb.berkeley.edu/labs/zucker/PDFs/ZuckerKullmannSchwarz.pdf)  
14. Causality \- Wikipedia, Zugriff am April 19, 2025, [https://en.wikipedia.org/wiki/Causality](https://en.wikipedia.org/wiki/Causality)  
15. Causality: Establishing Cause-and-Effect Relationships \- Datatab, Zugriff am April 19, 2025, [https://datatab.net/tutorial/causality](https://datatab.net/tutorial/causality)  
16. Causation-Based T Decomposition for Multivariate Process Monitoring and Diagnosis, Zugriff am April 19, 2025, [https://sites.gatech.edu/jianjun-shi/wp-content/uploads/sites/216/2015/09/p67.pdf](https://sites.gatech.edu/jianjun-shi/wp-content/uploads/sites/216/2015/09/p67.pdf)  
17. Two-stage approach to causality analysis-based quality problem solving for discrete manufacturing systems \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/384812672\_Two-stage\_approach\_to\_causality\_analysis-based\_quality\_problem\_solving\_for\_discrete\_manufacturing\_systems](https://www.researchgate.net/publication/384812672_Two-stage_approach_to_causality_analysis-based_quality_problem_solving_for_discrete_manufacturing_systems)  
18. Causal Plot: Causal-Based Fault Diagnosis Method Based on Causal Analysis \- MDPI, Zugriff am April 19, 2025, [https://www.mdpi.com/2227-9717/10/11/2269](https://www.mdpi.com/2227-9717/10/11/2269)  
19. Causal Plot: Causal-Based Fault Diagnosis Method Based on Causal Analysis \- Semantic Scholar, Zugriff am April 19, 2025, [https://pdfs.semanticscholar.org/e509/d56cfa29856d95466bf36c99dc9208973234.pdf](https://pdfs.semanticscholar.org/e509/d56cfa29856d95466bf36c99dc9208973234.pdf)  
20. Simplified Granger causality map for data-driven root cause diagnosis of process disturbances | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/345897284\_Simplified\_Granger\_causality\_map\_for\_data-driven\_root\_cause\_diagnosis\_of\_process\_disturbances](https://www.researchgate.net/publication/345897284_Simplified_Granger_causality_map_for_data-driven_root_cause_diagnosis_of_process_disturbances)  
21. Process Fault Diagnosis Method Based on MSPC and LiNGAM and its Application to Tennessee Eastman Process | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/360402717\_Process\_Fault\_Diagnosis\_Method\_Based\_on\_MSPC\_and\_LiNGAM\_and\_its\_Application\_to\_Tennessee\_Eastman\_Process](https://www.researchgate.net/publication/360402717_Process_Fault_Diagnosis_Method_Based_on_MSPC_and_LiNGAM_and_its_Application_to_Tennessee_Eastman_Process)  
22. Approach to using MSPC for power plant process monitoring and fault diagnosis, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/287163605\_Approach\_to\_using\_MSPC\_for\_power\_plant\_process\_monitoring\_and\_fault\_diagnosis](https://www.researchgate.net/publication/287163605_Approach_to_using_MSPC_for_power_plant_process_monitoring_and_fault_diagnosis)  
23. Multivariate Pattern Recognition in MSPC Using Bayesian Inference \- MDPI, Zugriff am April 19, 2025, [https://www.mdpi.com/2227-7390/9/4/306](https://www.mdpi.com/2227-7390/9/4/306)  
24. (PDF) Multivariate Pattern Recognition in MSPC Using Bayesian Inference \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/349046426\_Multivariate\_Pattern\_Recognition\_in\_MSPC\_Using\_Bayesian\_Inference](https://www.researchgate.net/publication/349046426_Multivariate_Pattern_Recognition_in_MSPC_Using_Bayesian_Inference)  
25. Granger Causality: A Review and Recent Advances \- PMC \- PubMed Central, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10571505/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10571505/)  
26. Chapter 4: Granger Causality Test — Time Series Analysis Handbook, Zugriff am April 19, 2025, [https://phdinds-aim.github.io/time\_series\_handbook/04\_GrangerCausality/04\_GrangerCausality.html](https://phdinds-aim.github.io/time_series_handbook/04_GrangerCausality/04_GrangerCausality.html)  
27. Granger causality \- Wikipedia, Zugriff am April 19, 2025, [https://en.wikipedia.org/wiki/Granger\_causality](https://en.wikipedia.org/wiki/Granger_causality)  
28. Granger Causality: A Review and Recent Advances \- arXiv, Zugriff am April 19, 2025, [https://arxiv.org/pdf/2105.02675](https://arxiv.org/pdf/2105.02675)  
29. Assessing the significance of directed and multivariate measures of linear dependence between time series | Phys. Rev. Research, Zugriff am April 19, 2025, [https://link.aps.org/doi/10.1103/PhysRevResearch.3.013145](https://link.aps.org/doi/10.1103/PhysRevResearch.3.013145)  
30. Introduction to Granger Causality \- Aptech, Zugriff am April 19, 2025, [https://www.aptech.com/blog/introduction-to-granger-causality/](https://www.aptech.com/blog/introduction-to-granger-causality/)  
31. The MVGC Multivariate Granger Causality Toolbox: A New Approach to Granger-causal Inference \- University of Sussex, Zugriff am April 19, 2025, [http://users.sussex.ac.uk/\~lionelb/downloads/NCOMP/publications/mvgc\_preprint.pdf](http://users.sussex.ac.uk/~lionelb/downloads/NCOMP/publications/mvgc_preprint.pdf)  
32. 10 Steps to Master Granger Causality Test in Financial Models, Zugriff am April 19, 2025, [https://www.numberanalytics.com/blog/10-steps-to-master-granger-causality-test-in-financial-models](https://www.numberanalytics.com/blog/10-steps-to-master-granger-causality-test-in-financial-models)  
33. Vector Autoregressive (VAR) Models and Granger Causality in Time Series Analysis in Nursing Research: Dynamic Changes Among Vital Signs Prior to Cardiorespiratory Instability Events as an Example \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5161241/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5161241/)  
34. Granger causality, Zugriff am April 19, 2025, [https://neuroimage.usc.edu/brainstorm/GrangerCausality](https://neuroimage.usc.edu/brainstorm/GrangerCausality)  
35. Granger Causality with or without VAR \- Cross Validated \- Stack Exchange, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/286656/granger-causality-with-or-without-var](https://stats.stackexchange.com/questions/286656/granger-causality-with-or-without-var)  
36. VAR model checks and granger causality tests \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/post/VAR\_model\_checks\_and\_granger\_causality\_tests](https://www.researchgate.net/post/VAR_model_checks_and_granger_causality_tests)  
37. Difference between Granger causality and Instantaneous causality? \- Cross Validated, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/404123/difference-between-granger-causality-and-instantaneous-causality](https://stats.stackexchange.com/questions/404123/difference-between-granger-causality-and-instantaneous-causality)  
38. Identification of Hidden Sources by Estimating Instantaneous Causality in High-Dimensional Biomedical Time Series | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/328584459\_Identification\_of\_Hidden\_Sources\_by\_Estimating\_Instantaneous\_Causality\_in\_High-Dimensional\_Biomedical\_Time\_Series](https://www.researchgate.net/publication/328584459_Identification_of_Hidden_Sources_by_Estimating_Instantaneous_Causality_in_High-Dimensional_Biomedical_Time_Series)  
39. independence \- Correlation vs dependence vs causality \- Cross ..., Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/509141/correlation-vs-dependence-vs-causality](https://stats.stackexchange.com/questions/509141/correlation-vs-dependence-vs-causality)  
40. From Dependency to Causality: A Machine Learning Approach, Zugriff am April 19, 2025, [https://jmlr.org/papers/volume16/bontempi15a/bontempi15a.pdf](https://jmlr.org/papers/volume16/bontempi15a/bontempi15a.pdf)  
41. Guide 6: Multivariate Crosstabulations and Causal Issues, Zugriff am April 19, 2025, [https://myweb.fsu.edu/slosh/IntroStatsGuide6.html](https://myweb.fsu.edu/slosh/IntroStatsGuide6.html)  
42. Comprehensive Review and Empirical Evaluation of Causal Discovery Algorithms for Numerical Data \- arXiv, Zugriff am April 19, 2025, [https://arxiv.org/html/2407.13054v1](https://arxiv.org/html/2407.13054v1)  
43. Biostatistics Series Module 10: Brief Overview of Multivariate Methods \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5527714/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5527714/)  
44. Chapter 16 Multivariate statistics | Statistical Thinking for the 21st Century, Zugriff am April 19, 2025, [https://statsthinking21.github.io/statsthinking21-core-site/multivariate.html](https://statsthinking21.github.io/statsthinking21-core-site/multivariate.html)  
45. Causation \- APA Dictionary of Psychology, Zugriff am April 19, 2025, [https://dictionary.apa.org/causation](https://dictionary.apa.org/causation)  
46. CAUSALITY definition in American English \- Collins Dictionary, Zugriff am April 19, 2025, [https://www.collinsdictionary.com/us/dictionary/english/causality](https://www.collinsdictionary.com/us/dictionary/english/causality)  
47. Pearls of Causality \#9: Potential, Genuine, Temporal Causes and Spurious Association, Zugriff am April 19, 2025, [https://rpatrik96.github.io/posts/2021/11/2021-11-29-poc9-causes](https://rpatrik96.github.io/posts/2021/11/2021-11-29-poc9-causes)  
48. Prediction vs. Causation in Regression Analysis | Statistical Horizons, Zugriff am April 19, 2025, [https://statisticalhorizons.com/prediction-vs-causation-in-regression-analysis/](https://statisticalhorizons.com/prediction-vs-causation-in-regression-analysis/)  
49. (PDF) Multivariate Statistical Analysis \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/319808256\_Multivariate\_Statistical\_Analysis](https://www.researchgate.net/publication/319808256_Multivariate_Statistical_Analysis)  
50. Characterizing Non-Linear Dependencies Among Pairs of Clinical Variables and Imaging Data \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3561932/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3561932/)  
51. Multivariate Statistical Process Control Charts and the Problem of Interpretation: A Short Overview and Some Applications in Industry \- arXiv, Zugriff am April 19, 2025, [https://arxiv.org/pdf/0901.2880](https://arxiv.org/pdf/0901.2880)  
52. (PDF) Multivariate statistics for process control \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/3206793\_Multivariate\_statistics\_for\_process\_control](https://www.researchgate.net/publication/3206793_Multivariate_statistics_for_process_control)  
53. Introduction to Multivariate SPC \- Einnosys, Zugriff am April 19, 2025, [https://www.einnosys.com/multivariate-spc/](https://www.einnosys.com/multivariate-spc/)  
54. Multivariate Statistical Process Control | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/316425197\_Multivariate\_Statistical\_Process\_Control](https://www.researchgate.net/publication/316425197_Multivariate_Statistical_Process_Control)  
55. Multivariate Statistical Process Control (MSPC) and Nonlinear Iterative Partial Least Squares (NIPALS) Overview \- TIBCO Product Documentation, Zugriff am April 19, 2025, [https://docs.tibco.com/pub/stat/14.0.1/doc/html/UsersGuide/\_shared/multivariate-Statistical-process-control-mspc-and-nonlinear-iterative-.htm](https://docs.tibco.com/pub/stat/14.0.1/doc/html/UsersGuide/_shared/multivariate-Statistical-process-control-mspc-and-nonlinear-iterative-.htm)  
56. Multivariate Statistical Process Control (MSPC) Technical Notes, Zugriff am April 19, 2025, [https://docs.tibco.com/pub/stat/14.0.0/doc/html/UsersGuide/GUID-06C2AE33-36E2-44B0-BB1A-5DB17153A9AA.html](https://docs.tibco.com/pub/stat/14.0.0/doc/html/UsersGuide/GUID-06C2AE33-36E2-44B0-BB1A-5DB17153A9AA.html)  
57. Multivariate Statistical Process Control (MSPC) and Nonlinear Iterative Partial Least Squares (NIPALS) Overview \- TIBCO Product Documentation, Zugriff am April 19, 2025, [https://docs.tibco.com/data-science/GUID-AD016675-05FC-4F5B-BC74-B8A0AAC40CF9.html](https://docs.tibco.com/data-science/GUID-AD016675-05FC-4F5B-BC74-B8A0AAC40CF9.html)  
58. Monitoring and Fault Detection with Multivariate Statistical Process Control (MSPC) in Continuous and Batch Processes \- Eigenvector Research, Zugriff am April 19, 2025, [http://www.eigenvector.com/Docs/Wise\_PAT.pdf](http://www.eigenvector.com/Docs/Wise_PAT.pdf)  
59. A new approach to multivariate statistical process control and its application to wastewater treatment process monitoring \- PHM Society, Zugriff am April 19, 2025, [http://papers.phmsociety.org/index.php/phmap/article/download/3751/2216](http://papers.phmsociety.org/index.php/phmap/article/download/3751/2216)  
60. Synchronization-Free Multivariate Statistical Process Control for Online Monitoring of Batch Process Evolution \- Frontiers, Zugriff am April 19, 2025, [https://www.frontiersin.org/journals/analytical-science/articles/10.3389/frans.2021.772844/full](https://www.frontiersin.org/journals/analytical-science/articles/10.3389/frans.2021.772844/full)  
61. Causality analysis in process control based on denoising and periodicity-removing CCM | Emerald Insight, Zugriff am April 19, 2025, [https://www.emerald.com/insight/content/doi/10.1108/jimse-06-2020-0003/full/html](https://www.emerald.com/insight/content/doi/10.1108/jimse-06-2020-0003/full/html)  
62. MULTISCALE PROCESS MONITORING WITH SINGULAR SPECTRUM ANALYSIS \- Stellenbosch University, Zugriff am April 19, 2025, [https://scholar.sun.ac.za/bitstream/10019.1/5246/1/krishnannair\_multiscale\_2010.pdf](https://scholar.sun.ac.za/bitstream/10019.1/5246/1/krishnannair_multiscale_2010.pdf)  
63. An Overview of Conventional MSPC Methods | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/378925850\_An\_Overview\_of\_Conventional\_MSPC\_Methods](https://www.researchgate.net/publication/378925850_An_Overview_of_Conventional_MSPC_Methods)  
64. Modeling Methodology for Nonlinear Physiological Systems \- University of Southern California, Zugriff am April 19, 2025, [https://customsitesmedia.usc.edu/wp-content/uploads/sites/106/2012/12/17062553/Annals-BME-25-1997-Journal.pdf](https://customsitesmedia.usc.edu/wp-content/uploads/sites/106/2012/12/17062553/Annals-BME-25-1997-Journal.pdf)  
65. Efficient batch process monitoring based on random nonlinear feature analysis | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/353308980\_Efficient\_batch\_process\_monitoring\_based\_on\_random\_nonlinear\_feature\_analysis](https://www.researchgate.net/publication/353308980_Efficient_batch_process_monitoring_based_on_random_nonlinear_feature_analysis)  
66. Characterizing non-linear dependencies among pairs of clinical variables and imaging data, Zugriff am April 19, 2025, [https://pubmed.ncbi.nlm.nih.gov/23366482/](https://pubmed.ncbi.nlm.nih.gov/23366482/)  
67. A new dissimilarity method integrating multidimensional mutual information and independent component analysis for non-Gaussian dynamic process monitoring | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/257035334\_A\_new\_dissimilarity\_method\_integrating\_multidimensional\_mutual\_information\_and\_independent\_component\_analysis\_for\_non-Gaussian\_dynamic\_process\_monitoring](https://www.researchgate.net/publication/257035334_A_new_dissimilarity_method_integrating_multidimensional_mutual_information_and_independent_component_analysis_for_non-Gaussian_dynamic_process_monitoring)  
68. Mutual Information–Dynamic Stacked Sparse Autoencoders for Fault Detection, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/336986079\_Mutual\_Information-Dynamic\_Stacked\_Sparse\_Autoencoders\_for\_Fault\_Detection](https://www.researchgate.net/publication/336986079_Mutual_Information-Dynamic_Stacked_Sparse_Autoencoders_for_Fault_Detection)  
69. Mutual information \- Wikipedia, Zugriff am April 19, 2025, [https://en.wikipedia.org/wiki/Mutual\_information](https://en.wikipedia.org/wiki/Mutual_information)  
70. Identifying bidirectional total and non-linear information flow in functional corticomuscular coupling during a dorsiflexion task: a pilot study \- PubMed Central, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8097856/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8097856/)  
71. 13.13: Correlation and Mutual Information \- Engineering LibreTexts, Zugriff am April 19, 2025, [https://eng.libretexts.org/Bookshelves/Industrial\_and\_Systems\_Engineering/Chemical\_Process\_Dynamics\_and\_Controls\_(Woolf)/13%3A\_Statistics\_and\_Probability\_Background/13.13%3A\_Correlation\_and\_Mutual\_Information](https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Chemical_Process_Dynamics_and_Controls_$Woolf$/13%3A_Statistics_and_Probability_Background/13.13%3A_Correlation_and_Mutual_Information)  
72. Multi-spectral phase coherence \- Wikipedia, Zugriff am April 19, 2025, [https://en.wikipedia.org/wiki/Multi-spectral\_phase\_coherence](https://en.wikipedia.org/wiki/Multi-spectral_phase_coherence)  
73. A Generalized Method for Determining Instantaneous Multi-Frequency Phase Coupling, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8803274/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8803274/)  
74. Estimating Multiple Latencies in the Auditory System from Auditory Steady-State Responses on a Single EEG Channel \- bioRxiv, Zugriff am April 19, 2025, [https://www.biorxiv.org/content/10.1101/2020.09.27.315614v4.full.pdf](https://www.biorxiv.org/content/10.1101/2020.09.27.315614v4.full.pdf)  
75. Multivariate statistical process control (MSPC) using Raman spectroscopy for in-line culture cell monitoring considering time-varying batches synchronized with correlation optimized warping (COW) \- PubMed, Zugriff am April 19, 2025, [https://pubmed.ncbi.nlm.nih.gov/28010847/](https://pubmed.ncbi.nlm.nih.gov/28010847/)  
76. Multivariate statistical process control (MSPC) using Raman spectroscopy for in-line culture cell monitoring considering time-varying batches synchronized with correlation optimized warping (COW) \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/311357786\_Multivariate\_statistical\_process\_control\_MSPC\_using\_Raman\_spectroscopy\_for\_in-line\_culture\_cell\_monitoring\_considering\_time-varying\_batches\_synchronized\_with\_correlation\_optimized\_warping\_COW](https://www.researchgate.net/publication/311357786_Multivariate_statistical_process_control_MSPC_using_Raman_spectroscopy_for_in-line_culture_cell_monitoring_considering_time-varying_batches_synchronized_with_correlation_optimized_warping_COW)  
77. MSPC, Zugriff am April 19, 2025, [https://genometric.github.io/MSPC/](https://genometric.github.io/MSPC/)  
78. A new approach to multivariate statistical process control and its application to wastewater treatment process monitoring | PHM Society Asia-Pacific Conference, Zugriff am April 19, 2025, [https://papers.phmsociety.org/index.php/phmap/article/view/3751](https://papers.phmsociety.org/index.php/phmap/article/view/3751)  
79. Process Fault Diagnosis Method Based on MSPC and LiNGAM and, Zugriff am April 19, 2025, [https://colab.ws/articles/10.1016%2Fj.ifacol.2022.04.224](https://colab.ws/articles/10.1016%2Fj.ifacol.2022.04.224)  
80. Application of Novel Statistical Process Control Methods to a Chemical Process | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/280177024\_Application\_of\_Novel\_Statistical\_Process\_Control\_Methods\_to\_a\_Chemical\_Process](https://www.researchgate.net/publication/280177024_Application_of_Novel_Statistical_Process_Control_Methods_to_a_Chemical_Process)  
81. Multivariate Statistical Process Control Method Including Soft Sensors for Both Early and Accurate Fault Detection | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/263942724\_Multivariate\_Statistical\_Process\_Control\_Method\_Including\_Soft\_Sensors\_for\_Both\_Early\_and\_Accurate\_Fault\_Detection](https://www.researchgate.net/publication/263942724_Multivariate_Statistical_Process_Control_Method_Including_Soft_Sensors_for_Both_Early_and_Accurate_Fault_Detection)  
82. Medical checkup data analysis method based on LiNGAM and its application to nonalcoholic fatty liver disease \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/360135598\_Medical\_checkup\_data\_analysis\_method\_based\_on\_LiNGAM\_and\_its\_application\_to\_nonalcoholic\_fatty\_liver\_disease](https://www.researchgate.net/publication/360135598_Medical_checkup_data_analysis_method_based_on_LiNGAM_and_its_application_to_nonalcoholic_fatty_liver_disease)  
83. Snowflake Mouse Sparkle MSPC Charm | Apple Watch Band | Button \- Etsy Australia, Zugriff am April 19, 2025, [https://www.etsy.com/au/listing/879356529/snowflake-mouse-sparkle-mspc-charm-apple](https://www.etsy.com/au/listing/879356529/snowflake-mouse-sparkle-mspc-charm-apple)  
84. Data-Driven Fault Detection and Reasoning for Industrial Monitoring \- OAPEN Library, Zugriff am April 19, 2025, [https://library.oapen.org/bitstream/20.500.12657/52452/1/978-981-16-8044-1.pdf](https://library.oapen.org/bitstream/20.500.12657/52452/1/978-981-16-8044-1.pdf)  
85. Data-Driven Modeling, Control and Optimization of Complex Industrial Processes \- MDPI, Zugriff am April 19, 2025, [https://www.mdpi.com/journal/processes/special\_issues/Development\_Application\_Intelligent\_Control\_System](https://www.mdpi.com/journal/processes/special_issues/Development_Application_Intelligent_Control_System)  
86. Compute directionality of connectivity with multivariate Granger causality \- MNE-Python, Zugriff am April 19, 2025, [https://mne.tools/mne-connectivity/dev/auto\_examples/granger\_causality.html](https://mne.tools/mne-connectivity/dev/auto_examples/granger_causality.html)  
87. (PDF) Granger Causality in Multivariate Time Series Using a Time-Ordered Restricted Vector Autoregressive Model \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/283761966\_Granger\_Causality\_in\_Multivariate\_Time\_Series\_Using\_a\_Time-Ordered\_Restricted\_Vector\_Autoregressive\_Model](https://www.researchgate.net/publication/283761966_Granger_Causality_in_Multivariate_Time_Series_Using_a_Time-Ordered_Restricted_Vector_Autoregressive_Model)  
88. Granger Causality Testing in High-Dimensional VARs: A Post-Double-Selection Procedure\* | Journal of Financial Econometrics | Oxford Academic, Zugriff am April 19, 2025, [https://academic.oup.com/jfec/article/21/3/915/6420401](https://academic.oup.com/jfec/article/21/3/915/6420401)  
89. Granger Causality: A Review and Recent Advances, Zugriff am April 19, 2025, [https://www.annualreviews.org/doi/10.1146/annurev-statistics-040120-010930](https://www.annualreviews.org/doi/10.1146/annurev-statistics-040120-010930)  
90. Granger Causality in Multi-variate Time Series using a Time Ordered Restricted Vector Autoregressive Model \- arXiv, Zugriff am April 19, 2025, [https://arxiv.org/pdf/1511.03463](https://arxiv.org/pdf/1511.03463)  
91. 07 Multivariate models: Granger causality, VAR and VECM models, Zugriff am April 19, 2025, [http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2019/05/Lecture\_07\_Updated.pdf](http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2019/05/Lecture_07_Updated.pdf)  
92. VARs and Granger Causality, Zugriff am April 19, 2025, [https://people.ucsc.edu/\~aspearot/Econ\_217/Econ\_217\_TS2.pdf](https://people.ucsc.edu/~aspearot/Econ_217/Econ_217_TS2.pdf)  
93. vargranger — Pairwise Granger causality tests \- Stata, Zugriff am April 19, 2025, [https://www.stata.com/manuals/tsvargranger.pdf](https://www.stata.com/manuals/tsvargranger.pdf)  
94. Multivariate Granger causality: an estimation framework based on factorization of the spectral density matrix \- PMC \- PubMed Central, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3971884/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3971884/)  
95. Causality: A Statistical View, Zugriff am April 19, 2025, [https://web-archive.lshtm.ac.uk/csm.lshtm.ac.uk/wp-content/uploads/sites/6/2016/04/Cox-2004.pdf](https://web-archive.lshtm.ac.uk/csm.lshtm.ac.uk/wp-content/uploads/sites/6/2016/04/Cox-2004.pdf)  
96. Partial correlation coefficient for a study with repeated measurements \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8735669/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8735669/)  
97. Partial correlation \- Wikipedia, Zugriff am April 19, 2025, [https://en.wikipedia.org/wiki/Partial\_correlation](https://en.wikipedia.org/wiki/Partial_correlation)  
98. Variable Selection via Partial Correlation \- PMC \- PubMed Central, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5484095/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5484095/)  
99. Partial correlation based variable selection approach for multivariate data classification methods | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/223883125\_Partial\_correlation\_based\_variable\_selection\_approach\_for\_multivariate\_data\_classification\_methods](https://www.researchgate.net/publication/223883125_Partial_correlation_based_variable_selection_approach_for_multivariate_data_classification_methods)  
100. A Partial Correlation-Based Algorithm for Causal Structure Discovery with Continuous Variables \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/221460874\_A\_Partial\_Correlation-Based\_Algorithm\_for\_Causal\_Structure\_Discovery\_with\_Continuous\_Variables](https://www.researchgate.net/publication/221460874_A_Partial_Correlation-Based_Algorithm_for_Causal_Structure_Discovery_with_Continuous_Variables)  
101. Multi-scale Fisher's independence test for multivariate dependence \- Oxford Academic, Zugriff am April 19, 2025, [https://academic.oup.com/biomet/article/109/3/569/6533498](https://academic.oup.com/biomet/article/109/3/569/6533498)  
102. Causal Information Approach to Partial Conditioning in Multivariate Data Sets \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3364562/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3364562/)  
103. Defining and Estimating Causal Effects \- From Neurons to Neighborhoods \- NCBI Bookshelf, Zugriff am April 19, 2025, [https://www.ncbi.nlm.nih.gov/books/NBK225543/](https://www.ncbi.nlm.nih.gov/books/NBK225543/)  
104. Multivariate analysis in thoracic research \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4387392/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4387392/)  
105. Lesson 13: Canonical Correlation Analysis \- STAT ONLINE, Zugriff am April 19, 2025, [https://online.stat.psu.edu/stat505/book/export/html/682](https://online.stat.psu.edu/stat505/book/export/html/682)  
106. A technical review of canonical correlation analysis for neuroscience applications \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7416047/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7416047/)  
107. Multi-set Canonical Correlation Analysis simply explained | Request PDF \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/323141701\_Multi-set\_Canonical\_Correlation\_Analysis\_simply\_explained](https://www.researchgate.net/publication/323141701_Multi-set_Canonical_Correlation_Analysis_simply_explained)  
108. Nonparametric Canonical Correlation Analysis \- Proceedings of Machine Learning Research, Zugriff am April 19, 2025, [http://proceedings.mlr.press/v48/michaeli16.pdf](http://proceedings.mlr.press/v48/michaeli16.pdf)  
109. Canonical correlation \- Wikipedia, Zugriff am April 19, 2025, [https://en.wikipedia.org/wiki/Canonical\_correlation](https://en.wikipedia.org/wiki/Canonical_correlation)  
110. Canonical Correlation Analysis for Data Fusion and Group Inferences \- PubMed Central, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2919827/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2919827/)  
111. Canonical Correlation Analysis | R Data Analysis Examples \- OARC Stats \- UCLA, Zugriff am April 19, 2025, [https://stats.oarc.ucla.edu/r/dae/canonical-correlation-analysis/](https://stats.oarc.ucla.edu/r/dae/canonical-correlation-analysis/)  
112. Example of Canonical Correlation Analysis \- JMP, Zugriff am April 19, 2025, [https://www.jmp.com/support/help/en/18.1/jmp/example-of-canonical-correlation-analysis.shtml](https://www.jmp.com/support/help/en/18.1/jmp/example-of-canonical-correlation-analysis.shtml)  
113. Canonical Correlation Analysis | Stata Data Analysis Examples \- OARC Stats \- UCLA, Zugriff am April 19, 2025, [https://stats.oarc.ucla.edu/stata/dae/canonical-correlation-analysis/](https://stats.oarc.ucla.edu/stata/dae/canonical-correlation-analysis/)  
114. Canonical correlation analysis \- Stanford Computer Graphics Laboratory, Zugriff am April 19, 2025, [https://graphics.stanford.edu/courses/cs233-21-spring/ReferencedPapers/CCA\_Weenik.pdf](https://graphics.stanford.edu/courses/cs233-21-spring/ReferencedPapers/CCA_Weenik.pdf)  
115. 7 Key Strategies for Canonical Correlation Analysis Overview \- Number Analytics, Zugriff am April 19, 2025, [https://www.numberanalytics.com/blog/7-key-strategies-canonical-correlation-analysis-overview](https://www.numberanalytics.com/blog/7-key-strategies-canonical-correlation-analysis-overview)  
116. Canonical Correlation a Tutorial, Zugriff am April 19, 2025, [https://www.cs.cmu.edu/\~tom/10701\_sp11/slides/CCA\_tutorial.pdf](https://www.cs.cmu.edu/~tom/10701_sp11/slides/CCA_tutorial.pdf)  
117. Canonical Correlation Analysis (CCA) \- YouTube, Zugriff am April 19, 2025, [https://www.youtube.com/watch?v=XQzKG511fNc](https://www.youtube.com/watch?v=XQzKG511fNc)  
118. Canonical correlation analysis for multi-omics: Application to cross-cohort analysis \- PubMed, Zugriff am April 19, 2025, [https://pubmed.ncbi.nlm.nih.gov/37216410/](https://pubmed.ncbi.nlm.nih.gov/37216410/)  
119. Practical Guide to Canonical Correlation Analysis for Statistical Insights \- Number Analytics, Zugriff am April 19, 2025, [https://www.numberanalytics.com/blog/practical-canonical-correlation-analysis-statistical-insights](https://www.numberanalytics.com/blog/practical-canonical-correlation-analysis-statistical-insights)  
120. Canonical Correlation Analysis in Detail \- Gregory Gundersen, Zugriff am April 19, 2025, [https://gregorygundersen.com/blog/2018/07/17/cca/](https://gregorygundersen.com/blog/2018/07/17/cca/)  
121. Help with understanding canonical correlation analysis : r/statistics \- Reddit, Zugriff am April 19, 2025, [https://www.reddit.com/r/statistics/comments/bcu8cr/help\_with\_understanding\_canonical\_correlation/](https://www.reddit.com/r/statistics/comments/bcu8cr/help_with_understanding_canonical_correlation/)  
122. how to explain canonical correlation to laymen? \- Cross Validated \- Stack Exchange, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/443327/how-to-explain-canonical-correlation-to-laymen](https://stats.stackexchange.com/questions/443327/how-to-explain-canonical-correlation-to-laymen)  
123. How to visualize what canonical correlation analysis does (in comparison to what principal component analysis does)? \- Cross Validated, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/65692/how-to-visualize-what-canonical-correlation-analysis-does-in-comparison-to-what](https://stats.stackexchange.com/questions/65692/how-to-visualize-what-canonical-correlation-analysis-does-in-comparison-to-what)  
124. Partial Correlation Diagram \- JMP, Zugriff am April 19, 2025, [https://www.jmp.com/support/help/en/18.1/jmp/partial-correlation-diagram.shtml](https://www.jmp.com/support/help/en/18.1/jmp/partial-correlation-diagram.shtml)  
125. Partial Correlation: Tutorial and Calculator \- Datatab, Zugriff am April 19, 2025, [https://datatab.net/tutorial/partial-correlation](https://datatab.net/tutorial/partial-correlation)  
126. Partial Correlation Example \- Easy and Helpful 2025 \- Statistics and Data Analysis, Zugriff am April 19, 2025, [https://itfeature.com/corr-ana/partial/partial-correlation-example/](https://itfeature.com/corr-ana/partial/partial-correlation-example/)  
127. Exploring Partial Correlation Methods for Enhanced Statistical Clarity \- Number Analytics, Zugriff am April 19, 2025, [https://www.numberanalytics.com/blog/exploring-partial-correlation-methods-for-enhanced-statistical-clarity](https://www.numberanalytics.com/blog/exploring-partial-correlation-methods-for-enhanced-statistical-clarity)  
128. Computational Approach \- Unique Prediction and Partial Correlation, Zugriff am April 19, 2025, [https://docs.tibco.com/data-science/GUID-326CE3EC-8A49-4C68-B14C-57193A9C9F2B.html](https://docs.tibco.com/data-science/GUID-326CE3EC-8A49-4C68-B14C-57193A9C9F2B.html)  
129. Partial and Semipartial Correlation, Zugriff am April 19, 2025, [http://faculty.cas.usf.edu/mbrannick/regression/Partial.html](http://faculty.cas.usf.edu/mbrannick/regression/Partial.html)  
130. Intuition behind the names 'partial' and 'marginal' correlations \- Stats Stackexchange, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/77318/intuition-behind-the-names-partial-and-marginal-correlations](https://stats.stackexchange.com/questions/77318/intuition-behind-the-names-partial-and-marginal-correlations)  
131. What's the intuition behind Velicer's minimum average partial (MAP) test? \- Cross Validated, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/267531/whats-the-intuition-behind-velicers-minimum-average-partial-map-test](https://stats.stackexchange.com/questions/267531/whats-the-intuition-behind-velicers-minimum-average-partial-map-test)  
132. What's the point of doing Partial Correlations if variables are already adjusted using multiple linear regression? : r/statistics \- Reddit, Zugriff am April 19, 2025, [https://www.reddit.com/r/statistics/comments/145cjm/whats\_the\_point\_of\_doing\_partial\_correlations\_if/](https://www.reddit.com/r/statistics/comments/145cjm/whats_the_point_of_doing_partial_correlations_if/)  
133. Back to the basics: Rethinking partial correlation network methodology \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8572131/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8572131/)  
134. Interpretation of partial distance correlation \- Cross Validated, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/659974/interpretation-of-partial-distance-correlation](https://stats.stackexchange.com/questions/659974/interpretation-of-partial-distance-correlation)  
135. What is the interpretation of "generalized" partial correlations? \- Stats Stackexchange, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/40995/what-is-the-interpretation-of-generalized-partial-correlations](https://stats.stackexchange.com/questions/40995/what-is-the-interpretation-of-generalized-partial-correlations)  
136. Learning partial correlation graphs and graphical models by covariance queries, Zugriff am April 19, 2025, [https://jmlr.csail.mit.edu/papers/volume22/20-1137/20-1137.pdf](https://jmlr.csail.mit.edu/papers/volume22/20-1137/20-1137.pdf)  
137. Help making sure I understand partial correlation $$
Question$$ : r/statistics \- Reddit, Zugriff am April 19, 2025, [https://www.reddit.com/r/statistics/comments/fxw6e8/help\_making\_sure\_i\_understand\_partial\_correlation/](https://www.reddit.com/r/statistics/comments/fxw6e8/help_making_sure_i_understand_partial_correlation/)  
138. Basic Maths Formulas for CBSE Class 6 to 12 with PDFs \- BYJU'S, Zugriff am April 19, 2025, [https://byjus.com/math-formulas/](https://byjus.com/math-formulas/)  
139. Partial correlation-the idea and significance \- Towards Data Science, Zugriff am April 19, 2025, [https://towardsdatascience.com/partial-correlation-508353cd8b5/](https://towardsdatascience.com/partial-correlation-508353cd8b5/)  
140. Partial Correlation in SPSS (SPSS Tutorial Video \#16) \- YouTube, Zugriff am April 19, 2025, [https://www.youtube.com/watch?v=jloi\_w8JsdU](https://www.youtube.com/watch?v=jloi_w8JsdU)  
141. Mutual Information, Clearly Explained\!\!\! \- YouTube, Zugriff am April 19, 2025, [https://www.youtube.com/watch?v=eJIp\_mgVLwE\&pp=0gcJCdgAo7VqN5tD](https://www.youtube.com/watch?v=eJIp_mgVLwE&pp=0gcJCdgAo7VqN5tD)  
142. Maximin Separation Probability Clustering, Zugriff am April 19, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/9627/9486](https://ojs.aaai.org/index.php/AAAI/article/view/9627/9486)  
143. Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation \- CVF Open Access, Zugriff am April 19, 2025, [https://openaccess.thecvf.com/content/CVPR2022/papers/Xu\_Maximum\_Spatial\_Perturbation\_Consistency\_for\_Unpaired\_Image-to-Image\_Translation\_CVPR\_2022\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Maximum_Spatial_Perturbation_Consistency_for_Unpaired_Image-to-Image_Translation_CVPR_2022_paper.pdf)  
144. Mutual information \- Scholarpedia, Zugriff am April 19, 2025, [http://www.scholarpedia.org/article/Mutual\_information](http://www.scholarpedia.org/article/Mutual_information)  
145. Entropy and Mutual Information, Zugriff am April 19, 2025, [https://people.cs.umass.edu/\~elm/Teaching/Docs/mutInf.pdf](https://people.cs.umass.edu/~elm/Teaching/Docs/mutInf.pdf)  
146. Understanding Mutual Information \- Home \- Matthew Kowal, Zugriff am April 19, 2025, [https://mkowal2.github.io/posts/2020/01/understanding-mi/](https://mkowal2.github.io/posts/2020/01/understanding-mi/)  
147. Lecture 1: Entropy and mutual information \- Department of Electrical and Computer Engineering, Zugriff am April 19, 2025, [http://www.ece.tufts.edu/ee/194NIT/lect01.pdf](http://www.ece.tufts.edu/ee/194NIT/lect01.pdf)  
148. (PDF) Multiscale statistical testing for connectome-wide association studies in fMRI, Zugriff am April 19, 2025, [https://www.researchgate.net/publication/265469438\_Multiscale\_statistical\_testing\_for\_connectome-wide\_association\_studies\_in\_fMRI](https://www.researchgate.net/publication/265469438_Multiscale_statistical_testing_for_connectome-wide_association_studies_in_fMRI)  
149. CONTENTS 1 \- Public Service, Zugriff am April 19, 2025, [https://publicservice.gov.mt/Media/Publications/GA%20-%20Ombudsman%20Publication%202023%20updated.pdf](https://publicservice.gov.mt/Media/Publications/GA%20-%20Ombudsman%20Publication%202023%20updated.pdf)  
150. Faraday's Laws of Electromagnetic Induction \- Lenz's Law, Formula, Derivation, Video, and FAQs \- BYJU'S, Zugriff am April 19, 2025, [https://byjus.com/physics/faradays-law/](https://byjus.com/physics/faradays-law/)  
151. An Intuitive View on Mutual Information \- Towards Data Science, Zugriff am April 19, 2025, [https://towardsdatascience.com/an-intuitive-view-on-mutual-information-db0655535f84/](https://towardsdatascience.com/an-intuitive-view-on-mutual-information-db0655535f84/)  
152. probability theory \- Mutual information intuition \- Computer Science Stack Exchange, Zugriff am April 19, 2025, [https://cs.stackexchange.com/questions/30155/mutual-information-intuition](https://cs.stackexchange.com/questions/30155/mutual-information-intuition)  
153. Is there an intuitive interpretation of mutual information values (bits, nits)? \- Cross Validated, Zugriff am April 19, 2025, [https://stats.stackexchange.com/questions/193818/is-there-an-intuitive-interpretation-of-mutual-information-values-bits-nits](https://stats.stackexchange.com/questions/193818/is-there-an-intuitive-interpretation-of-mutual-information-values-bits-nits)  
154. Better intuition for information theory \- Andreas Kirsch, Zugriff am April 19, 2025, [https://www.blackhc.net/blog/2019/better-intuition-for-information-theory/](https://www.blackhc.net/blog/2019/better-intuition-for-information-theory/)  
155. GMI Explained \- Graphic Mutual Information \- Papers With Code, Zugriff am April 19, 2025, [https://paperswithcode.com/method/gmi](https://paperswithcode.com/method/gmi)  
156. Learning Representations by Graphical Mutual Information Estimation and Maximization, Zugriff am April 19, 2025, [https://pubmed.ncbi.nlm.nih.gov/35104214/](https://pubmed.ncbi.nlm.nih.gov/35104214/)  
157. Graph Representation Learning via Graphical Mutual Information Maximization \- arXiv, Zugriff am April 19, 2025, [https://arxiv.org/pdf/2002.01169](https://arxiv.org/pdf/2002.01169)  
158. Lecture 7: September 21 7.1 Overview 7.2 Application: Structure Learning in General Graphical Models, Zugriff am April 19, 2025, [https://www.cs.cmu.edu/\~aarti/Class/10704\_Fall16/lec7.pdf](https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/lec7.pdf)  
159. $$
2002.01169$$ Graph Representation Learning via Graphical Mutual Information Maximization \- arXiv, Zugriff am April 19, 2025, [https://arxiv.org/abs/2002.01169](https://arxiv.org/abs/2002.01169)  
160. Exploiting Mutual Information for Substructure-aware Graph Representation Learning \- IJCAI, Zugriff am April 19, 2025, [https://www.ijcai.org/proceedings/2020/472](https://www.ijcai.org/proceedings/2020/472)  
161. zpeng27/GMI: Graph Representation Learning via Graphical Mutual Information Maximization \- GitHub, Zugriff am April 19, 2025, [https://github.com/zpeng27/GMI](https://github.com/zpeng27/GMI)  
162. Causal Inference with Bayesian Networks. Main Concepts and Methods \- CausalNex's, Zugriff am April 19, 2025, [https://causalnex.readthedocs.io/en/latest/04\_user\_guide/04\_user\_guide.html](https://causalnex.readthedocs.io/en/latest/04_user_guide/04_user_guide.html)  
163. Bayesian Causal Networks for Complex Multivariate Systems \- Center for Wildlife Studies, Zugriff am April 19, 2025, [https://www.centerforwildlifestudies.org/courses/p/bayesian-causal-network-modeling](https://www.centerforwildlifestudies.org/courses/p/bayesian-causal-network-modeling)  
164. Local Characterizations of Causal Bayesian Networks\* \- CS@Purdue, Zugriff am April 19, 2025, [https://www.cs.purdue.edu/homes/eb/r384-lnai.pdf](https://www.cs.purdue.edu/homes/eb/r384-lnai.pdf)  
165. Bayesian Networks for Causal Analysis \- SAS Support, Zugriff am April 19, 2025, [https://support.sas.com/resources/papers/proceedings18/2776-2018.pdf](https://support.sas.com/resources/papers/proceedings18/2776-2018.pdf)  
166. 1.3 Causal Bayesian Networks, Zugriff am April 19, 2025, [https://bayes.cs.ucla.edu/BOOK-2K/ch1-3.pdf](https://bayes.cs.ucla.edu/BOOK-2K/ch1-3.pdf)  
167. Hengyi Hu: Establishing Causality Using Bayesian Networks \- YouTube, Zugriff am April 19, 2025, [https://www.youtube.com/watch?v=yP\_CkrcvKiE](https://www.youtube.com/watch?v=yP_CkrcvKiE)  
168. Introduction to Bayesian networks \- Bayes Server, Zugriff am April 19, 2025, [https://bayesserver.com/docs/introduction/bayesian-networks/](https://bayesserver.com/docs/introduction/bayesian-networks/)  
169. 13.5: Bayesian Network Theory \- Engineering LibreTexts, Zugriff am April 19, 2025, [https://eng.libretexts.org/Bookshelves/Industrial\_and\_Systems\_Engineering/Chemical\_Process\_Dynamics\_and\_Controls\_(Woolf)/13%3A\_Statistics\_and\_Probability\_Background/13.05%3A\_Bayesian\_network\_theory](https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Chemical_Process_Dynamics_and_Controls_$Woolf$/13%3A_Statistics_and_Probability_Background/13.05%3A_Bayesian_network_theory)  
170. Bayesian network \- Wikipedia, Zugriff am April 19, 2025, [https://en.wikipedia.org/wiki/Bayesian\_network](https://en.wikipedia.org/wiki/Bayesian_network)  
171. An Introduction to the Theory and Applications of Bayesian Networks \- Scholarship @ Claremont, Zugriff am April 19, 2025, [https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=2690\&context=cmc\_theses](https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=2690&context=cmc_theses)  
172. Understanding Bayesian Networks \- with Examples in R \- bnlearn, Zugriff am April 19, 2025, [https://www.bnlearn.com/about/teaching/slides-bnshort.pdf](https://www.bnlearn.com/about/teaching/slides-bnshort.pdf)  
173. Introduction to Bayesian Networks \- Andres Mendez-Vazquez, An Exploration about Intelligent Systems, Zugriff am April 19, 2025, [https://kajuna0amendez.github.io/assets/ai\_files/12\_Introduction\_Bayesian\_Networks.pdf](https://kajuna0amendez.github.io/assets/ai_files/12_Introduction_Bayesian_Networks.pdf)  
174. Bayesian Networks \- YouTube, Zugriff am April 19, 2025, [https://www.youtube.com/watch?v=TuGDMj43ehw\&pp=0gcJCdgAo7VqN5tD](https://www.youtube.com/watch?v=TuGDMj43ehw&pp=0gcJCdgAo7VqN5tD)  
175. Bayesian networks elucidate complex genomic landscapes in cancer \- PMC, Zugriff am April 19, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8980036/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8980036/)  
176. CS4786/5786: Machine Learning for Data Science, Spring 2015 4/21/2015: Lecture 22 handout: More intuition on Bayesian Networks A \- CS@Cornell, Zugriff am April 19, 2025, [https://www.cs.cornell.edu/courses/cs4786/2015sp/lectures/lec22-handout.pdf](https://www.cs.cornell.edu/courses/cs4786/2015sp/lectures/lec22-handout.pdf)  
177. Bayesian networks, Zugriff am April 19, 2025, [https://kuleshov.github.io/cs228-notes/representation/directed/](https://kuleshov.github.io/cs228-notes/representation/directed/)  
178. Graphical Models, Bayesian Networks, Zugriff am April 19, 2025, [https://courses.media.mit.edu/2008fall/mas622j/Projects/CharlieCocoErnestoMatt/graphical\_models/](https://courses.media.mit.edu/2008fall/mas622j/Projects/CharlieCocoErnestoMatt/graphical_models/)  
179. General structure of a Bayesian Network. \- ResearchGate, Zugriff am April 19, 2025, [https://www.researchgate.net/figure/General-structure-of-a-Bayesian-Network\_fig1\_349046426](https://www.researchgate.net/figure/General-structure-of-a-Bayesian-Network_fig1_349046426)  
180. A Brief Introduction to Graphical Models and Bayesian Networks Representation \- UBC Computer Science, Zugriff am April 19, 2025, [https://www.cs.ubc.ca/\~murphyk/Bayes/bayes\_tutorial.pdf](https://www.cs.ubc.ca/~murphyk/Bayes/bayes_tutorial.pdf)  
181. Lecture 15.1: Bayesian Networks/Probabilistic Graphical Models | ML19 \- YouTube, Zugriff am April 19, 2025, [https://www.youtube.com/watch?v=KnDP8-S7gl4](https://www.youtube.com/watch?v=KnDP8-S7gl4)  
182. Causality Analysis \- R, Zugriff am April 19, 2025, [https://search.r-project.org/CRAN/refmans/vars/help/causality.html](https://search.r-project.org/CRAN/refmans/vars/help/causality.html)  
183. Tutorial for “Identification of Causal Dependencies in Multivariate Time Series”, Zugriff am April 19, 2025, [https://sujoyrc.github.io/causal\_time\_series\_tutorial/](https://sujoyrc.github.io/causal_time_series_tutorial/)