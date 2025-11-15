# Boundary-Based Multivariate Relationship Discovery: Research Analysis

## 1. Core Research Idea

### 1.1 Methodology Overview

**Approach:** Discover temporal relationships between process variables by tracking co-occurrence of distributional boundary violations with hierarchical refinement.

**Key Concepts:**
- **Distributional Boundaries:** Define normal operating ranges using statistical thresholds (mean ± kσ)
- **Excursion Detection:** Binary classification of values as inside/outside boundaries
- **Co-occurrence Tracking:** Count simultaneous boundary violations across variable pairs
- **Hierarchical Refinement:** Progressive boundary tightening (2σ → 1.5σ → 1σ) to capture multi-scale relationships
- **Time-Lagged Analysis:** Discover temporal dependencies where one variable's excursion precedes another's
- **Relationship Network:** Graph representation of discovered dependencies with strength scores

### 1.2 Distinguishing Features

**What Makes This Approach Different:**

1. **Interpretability-First Design**
   - Uses intuitive boundary violations rather than complex statistical models
   - Visual timeline diagrams show when variables spike together
   - Easily explainable to domain experts and process engineers

2. **Scale-Aware Detection**
   - Multi-level boundaries capture both major disturbances (wide boundaries) and subtle coordinated changes (narrow boundaries)
   - Different relationships emerge at different scales

3. **Lag-Explicit Reporting**
   - Organizes discovered relationships by time delay (lag 0, 1, 2, ..., n)
   - Makes temporal structure of process dynamics transparent

4. **Simplicity**
   - No assumptions about linearity, stationarity, or distribution types
   - Minimal hyperparameters compared to deep learning approaches
   - Computationally efficient for real-time online monitoring

---

## 2. Related Approaches in Literature

### 2.1 Traditional Statistical Methods

**Granger Causality**
- Tests if past values of variable X improve prediction of variable Y
- **Limitation:** Assumes linear relationships and stationarity
- **Your Advantage:** No linearity assumption, handles non-stationary data

**Mutual Information / Time-Delayed Mutual Information (TDMI)**
- Measures how knowing one variable reduces uncertainty about another at different time lags
- **Limitation:** Computationally intensive, requires large sample sizes, difficult to interpret
- **Your Advantage:** Simpler counting approach, more interpretable results

**Principal Component Analysis (PCA) Based Control Charts**
- Standard approach for multivariate process monitoring using T² and Q statistics
- **Limitation:** Difficult to interpret which specific variables are involved in faults
- **Your Advantage:** Explicitly identifies variable relationships and provides visual timelines

### 2.2 Modern Deep Learning Approaches

**Graph Neural Networks (GNN) for Time Series**
- Recent surge in GNN-based methods that model inter-variable relationships as graphs
- Multi-scale dynamic GNNs capture spatial-temporal dependencies in industrial sensor data
- **Limitation:** Black-box models, difficult to interpret, require substantial training data
- **Your Advantage:** White-box approach, no training required, works with limited data

**Transformer-Based Anomaly Detection**
- Attention mechanisms capture long-term dependencies in multivariate time series
- **Limitation:** Computationally expensive, require careful tuning, lack interpretability
- **Your Advantage:** Much simpler, transparent decision-making process

**Deep Autoencoders & Variational Autoencoders**
- Learn compressed representations and detect anomalies through reconstruction error
- Difficulty interpreting reconstruction errors and boundaries between normal/abnormal
- **Your Advantage:** Clear, interpretable boundaries; explicit relationship identification

### 2.3 Interpretable & Causal Discovery Methods

**Explainable AI (XAI) for Anomaly Detection**
- Methods like SHAP and LIME provide post-hoc explanations for deep learning models
- Correlation-based explainable anomaly detection using Pearson correlation and latent correlation matrices
- **Similarity:** Both emphasize interpretability
- **Your Advantage:** Built-in interpretability rather than post-hoc explanations

**Causal Discovery from Time Series**
- Recent work on discovering causal relationships in industrial processes for degradation monitoring
- Survey of temporal causal discovery methods including constraint-based and score-based approaches
- Data-driven causality using interpretable machine learning and process mining
- **Similarity:** Both aim to discover relationships between variables
- **Difference:** You focus on co-occurrence patterns rather than formal causal inference

**Feature Importance in Ensemble Models**
- Using Random Forest feature importance to establish causal relationships in time series
- **Similarity:** Interpretable relationship discovery
- **Difference:** You use boundary violations rather than prediction-based importance

### 2.4 Boundary-Based Detection Methods

**Support Vector Data Description (SVDD)**
- Defines boundary around normal data points for one-class classification
- **Similarity:** Both use boundary concepts
- **Difference:** You apply boundaries to individual variables then track co-occurrences

**Change Point Detection**
- Identifies when statistical properties of data change over time
- **Limitation:** Typically focuses on single change moments, not coordinated changes
- **Your Advantage:** Explicitly tracks which variables change together

---

## 3. Research Gaps Your Approach Addresses

### 3.1 Interpretability Gap
Most existing anomaly detection methods (especially deep learning) operate as "black boxes" that are difficult to interpret

**Your Solution:** Timeline visualizations showing when variables exceed boundaries make relationships immediately visible to process engineers.

### 3.2 Multi-Scale Relationship Gap
Boundary between normal and abnormal is often imprecisely defined and constantly evolving

**Your Solution:** Hierarchical refinement captures relationships at multiple scales, from major faults to subtle coordinated variations.

### 3.3 Temporal Structure Gap
Many methods detect relationships but don't explicitly characterize time lags.

**Your Solution:** Lag-organized reporting clearly shows which relationships are simultaneous vs. delayed, revealing process dynamics.

### 3.4 Practical Applicability Gap
Deep learning methods require large labeled datasets, substantial computational resources, and careful hyperparameter tuning

**Your Solution:** Simple, parameter-light approach suitable for resource-constrained industrial environments.

---

## 4. Improvement Potential & Future Research Directions

### 4.1 Short-Term Enhancements

**1. Advanced Boundary Methods**
- Beyond fixed σ-based boundaries:
  - Quantile-based boundaries (more robust to outliers)
  - Adaptive boundaries that evolve with gradual process drift
  - Context-dependent boundaries (different for startup vs. steady-state)

**2. Weighted Co-occurrence**
- Not all excursions are equal:
  - Weight by excursion magnitude (how far outside boundaries)
  - Weight by duration of excursion
  - Consider frequency of co-occurrence patterns

**3. Directionality Detection**
- Current approach is symmetric (A ↔ B)
- Add asymmetric analysis to determine direction (A → B vs. B → A)
- Use lag information + excursion ordering to infer causality

**4. Multivariate Co-occurrence**
- Track when 3+ variables simultaneously violate boundaries
- Identify cascading failure patterns
- Build hypergraphs instead of pairwise graphs

### 4.2 Integration with Modern ML

**5. Hybrid Approach: Combine with GNNs**
Graph neural networks excel at learning complex spatial-temporal dependencies

**Potential:** Use your boundary-based relationships to **initialize** GNN graph structure
- GNNs often struggle with graph structure learning
- Your method provides interpretable, physics-informed initial graph
- GNN can then refine relationships through training
- Best of both worlds: interpretable foundation + learning capacity

**6. Physics-Informed Enhancement**
- Incorporate physical process knowledge into graph construction (e.g., controller-process relationships in ICS)
- Validate discovered relationships against known process physics
- Flag unexpected relationships for expert investigation

**7. Attention Mechanisms for Relationship Strength**
- Learn attention weights on different lag values
- Automatically identify most important temporal dependencies
- Maintain interpretability through attention visualization

### 4.3 Advanced Analytics

**8. Dynamic Relationship Tracking**
- Monitor changes in causal relationships over time to detect degradation
- Track how relationship strengths evolve
- Detect when relationships break down (early fault warning)
- Identify emergence of new unexpected relationships

**9. Root Cause Analysis**
- When fault detected, trace back through relationship network
- Identify "upstream" variables that triggered cascade
- Prioritize which sensor/actuator requires attention

**10. Probabilistic Relationships**
- Instead of binary co-occurrence counts, estimate probability distributions
- Model uncertainty in relationship strengths
- Enable risk-based decision making

### 4.4 Scalability & Online Learning

**11. Incremental Online Updates**
- Current approach requires batch recomputation
- Develop incremental algorithms that update as new data arrives
- Critical for real-time industrial applications

**12. Dimensionality Reduction**
- For processes with 100+ variables:
  - First stage: Use your method to identify strongly connected variable groups
  - Second stage: Apply deep learning only to relevant subsets
  - Reduces computational burden while maintaining interpretability

**13. Distributed Computing**
- Partition variable space across computing nodes
- Parallel co-occurrence computation
- Scale to very large industrial processes

### 4.5 Domain-Specific Extensions

**14. Fault Type Classification**
- Tennessee Eastman has 20+ distinct fault types
- Use relationship patterns as "fault signatures"
- Build classifier: relationship pattern → fault type
- Enable automated diagnosis

**15. Transfer Learning Across Processes**
- Learn common relationship patterns from multiple facilities
- Transfer knowledge when monitoring new but similar processes
- Reduce cold-start problem

**16. Multi-Modal Integration**
- Combine time series data with:
  - Maintenance logs (text)
  - Operator actions (event data)
  - Visual inspection data (images)
- Richer context for relationship discovery

### 4.6 Theoretical Foundations

**17. Statistical Significance Testing**
- Develop formal hypothesis tests for relationship significance
- Account for multiple comparisons (many variable pairs)
- Provide confidence intervals on relationship strengths

**18. Identifiability Conditions**
- Causal discovery methods have well-studied identifiability conditions (faithfulness, causal sufficiency)
- Establish conditions under which your boundary-based relationships correctly identify true dependencies
- Characterize when method might produce spurious relationships

**19. Information-Theoretic Interpretation**
- Connect boundary violations to mutual information concepts
- Prove relationships between co-occurrence counts and information-theoretic measures
- Establish theoretical guarantees on detection performance

### 4.7 Emerging Technology Integration

**20. Large Language Models (LLMs) for Interpretation**
- Recent work on using LLMs to interpret causal graphs and provide domain-aware explanations
- Feed discovered relationships + variable metadata to LLM
- Generate natural language explanations of fault scenarios
- Interface for operators: "Explain why pressure increased"

**21. Digital Twin Integration**
- Use discovered relationships to build/validate digital twins
- Simulate "what-if" scenarios
- Test control strategies in virtual environment

**22. Federated Learning for Privacy**
- Multiple facilities contribute to relationship discovery
- Without sharing raw sensitive process data
- Learn common patterns while preserving proprietary information

---

## 5. Research Publication Strategy

### 5.1 Positioning

**Unique Selling Points for Publication:**
1. **Interpretability** - Bridge between "detect anomaly" and "understand cause"
2. **Simplicity** - Practical alternative to complex deep learning
3. **Multi-scale analysis** - Hierarchical refinement as novel contribution
4. **Validation** - Demonstrate on Tennessee Eastman benchmark

### 5.2 Target Venues

**Tier 1 Journals:**
- *IEEE Transactions on Industrial Informatics*
- *Control Engineering Practice*
- *Journal of Process Control*
- *Computers & Chemical Engineering*

**Conferences:**
- *American Control Conference (ACC)*
- *IEEE Conference on Decision and Control (CDC)*
- *IFAC Symposium on Fault Detection, Supervision and Safety*

### 5.3 Potential Paper Titles

1. "Interpretable Multivariate Relationship Discovery through Hierarchical Boundary Analysis for Process Monitoring"
2. "Scale-Aware Co-occurrence Mining for Explainable Fault Diagnosis in Industrial Processes"
3. "From Detection to Understanding: A Boundary-Based Framework for Temporal Relationship Discovery in Process Control"

---

## 6. Validation & Benchmarking Plan

### 6.1 Datasets

**Primary:**
- Tennessee Eastman Process (your current focus)
- All 20+ fault types
- Compare offline analysis vs. online detection performance

**Additional Benchmarks:**
- Secure Water Treatment (SWaT) - ICS security benchmark
- WADI (Water Distribution) - Another ICS benchmark
- Real industrial data from collaborating plant (if available)

### 6.2 Comparison Baselines

**Statistical Methods:**
- Granger causality
- Transfer entropy / Mutual information
- PCA + T² control charts

**Deep Learning:**
- LSTM-VAE (reconstruction-based)
- GNN-based methods (GDN, MTAD-GAT, recent multi-scale approaches)
- Transformer-based detectors

**Evaluation Metrics:**
1. **Detection Performance:** Precision, Recall, F1-score, AUC-ROC
2. **Relationship Quality:** How many known relationships discovered? How many spurious?
3. **Interpretability:** User study with domain experts
4. **Computational Efficiency:** Time complexity, memory usage
5. **Online Performance:** Detection delay, throughput

---

## 7. Expected Contributions

### 7.1 Methodological Contributions

1. **Novel framework** combining distributional boundaries with hierarchical refinement for relationship discovery
2. **Lag-based reporting** methodology that organizes findings by temporal structure
3. **Scalable algorithm** for online anomaly detection using learned relationship networks

### 7.2 Practical Contributions

1. **Interpretable tool** for process engineers to understand variable interactions
2. **Visual diagnostics** (timeline diagrams, network graphs) for fault investigation
3. **Benchmark results** on Tennessee Eastman demonstrating effectiveness

### 7.3 Theoretical Contributions

1. **Characterization** of when boundary-based co-occurrence captures true dependencies
2. **Analysis** of relationship detection under different noise conditions
3. **Connection** between your approach and information-theoretic measures

---

## 8. Risks & Mitigation

### 8.1 Potential Weaknesses

**Risk 1: Spurious Relationships**
- Two variables might exceed boundaries simultaneously due to common cause (confounding)
- **Mitigation:** Compare against partial correlation, use conditional independence tests

**Risk 2: Sensitivity to Boundary Choice**
- Results may depend heavily on σ values chosen
- **Mitigation:** Hierarchical refinement addresses this; also perform sensitivity analysis

**Risk 3: Nonlinear Relationships**
- Boundary violations might miss complex nonlinear dependencies
- **Mitigation:** Combine with kernel-based or deep learning methods for validation

### 8.2 Comparison Fairness

**Challenge:** Deep learning methods train on labeled data; yours doesn't require labels
- Make comparison fair by testing in unsupervised/semi-supervised settings
- Highlight this as advantage rather than limitation

---

## 9. Summary

Your research idea addresses a critical gap in process monitoring: the need for **interpretable, actionable relationship discovery** that bridges detection and diagnosis. By combining simple boundary-based concepts with hierarchical refinement and temporal analysis, you offer a practical alternative to complex black-box methods.

**Key Strengths:**
- ✅ Interpretable by design
- ✅ Works with limited data
- ✅ Computationally efficient
- ✅ Provides lag information explicitly
- ✅ Suitable for online monitoring

**Areas for Enhancement:**
- Integration with modern ML (GNNs, LLMs)
- Formal statistical foundations
- Dynamic relationship tracking
- Multi-scale extensions

This approach has strong publication potential in control engineering and industrial informatics venues, with practical relevance for real-world process monitoring applications.

---

## 10. Comprehensive List of Similar Work with Links

### 10.1 Interpretable Anomaly Detection

**1. Explainable Correlation-Based Anomaly Detection for Industrial Control Systems**
- **Focus:** Correlation-based explainable anomaly detection using Latent Correlation Matrix
- **Key Methods:** LSTM-AE, Pearson correlation, Multivariate Gaussian Distribution
- **Link:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1508821/full
- **Relevance:** Similar focus on interpretability in ICS, uses correlation rather than boundaries

**2. An Interpretable Method for Anomaly Detection in Multivariate Time Series Predictions**
- **Focus:** Gradient optimization-based interpretation for deep learning anomaly detectors
- **Key Methods:** DeepAID, optimization-based interpretation
- **Link:** https://www.mdpi.com/2076-3417/15/13/7479
- **Relevance:** Addresses interpretability but as post-hoc explanation vs. your built-in interpretability

**3. An Interpretable Multivariate Time-Series Anomaly Detection Method Based on Adaptive Mask**
- **Focus:** Adaptive masking for interpretability in cyber-physical systems
- **Key Methods:** Series saliency module, adaptive mask learning
- **Link:** https://ieeexplore.ieee.org/document/10177932/
- **Relevance:** Different approach to interpretability through masking

### 10.2 Boundary-Based & Threshold Methods

**4. Deep Learning Advancements in Anomaly Detection (Survey)**
- **Focus:** Comprehensive survey including boundary-based methods like SVDD
- **Key Methods:** Support Vector Data Description, hybrid clustering approaches
- **Link:** https://arxiv.org/html/2503.13195v1
- **Relevance:** Discusses boundary definition for anomaly detection

**5. Leveraging Distributional Statistics in Anomaly Detection**
- **Focus:** Using distributional properties (mean, std, IQR, quantiles) for anomaly detection
- **Key Methods:** Statistical thresholds, z-scores, IQR-based boundaries
- **Link:** https://medium.com/@nlztrk/leveraging-distributional-statistics-in-anomaly-detection-bf00eebf8370
- **Relevance:** Similar use of statistical boundaries but for univariate analysis

**6. EPASAD: Ellipsoid Decision Boundary Based Process-Aware Detector**
- **Focus:** Boundary-based detection for micro-stealthy attacks in ICS
- **Key Methods:** Ellipsoid boundaries, process-aware monitoring
- **Link:** https://cybersecurity.springeropen.com/articles/10.1186/s42400-023-00162-z
- **Relevance:** Uses boundary concepts for ICS security

### 10.3 Graph Neural Networks for Time Series

**7. A Survey on Graph Neural Networks for Time Series (Comprehensive)**
- **Focus:** GNN4TS covering forecasting, classification, anomaly detection, imputation
- **Key Methods:** Spatial-temporal GNNs, graph construction methods
- **Link:** https://arxiv.org/abs/2307.03759
- **Link (IEEE):** https://dl.acm.org/doi/10.1109/TPAMI.2024.3443141
- **Relevance:** Your method could integrate with GNN approaches - comprehensive survey

**8. MSDG: Multi-Scale Dynamic Graph Neural Network for Industrial Time Series**
- **Focus:** Multi-scale analysis for industrial sensor data anomaly detection
- **Key Methods:** Dynamic graph construction, multi-scale sliding windows
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11598168/
- **Link (MDPI):** https://www.mdpi.com/1424-8220/24/22/7218
- **Relevance:** Similar multi-scale concept, GNN-based vs. your boundary-based

**9. Graph Attention Network and Informer for Multivariate Time Series Anomaly Detection**
- **Focus:** Combining GAT with Informer for long-term dependencies
- **Key Methods:** Graph Attention, Transformer architecture
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10935277/
- **Relevance:** State-of-art GNN method for comparison

**10. TopoGDN: Topological Analysis Enhanced Graph Attention Network**
- **Focus:** Fine-grained spatial-temporal analysis with topological features
- **Key Methods:** Multi-scale temporal convolution, topological analysis
- **Link:** https://dl.acm.org/doi/10.1145/3627673.3679614
- **Relevance:** Recent (2024) state-of-art method for benchmarking

**11. Masked Graph Neural Networks for Unsupervised Anomaly Detection**
- **Focus:** Masking strategy in GNNs for learning relationships
- **Key Methods:** Graph masking, adversarial training
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10490803/
- **Relevance:** Different approach to relationship learning

**12. Multi-Level Graph Attention Network for ICS Anomaly Detection**
- **Focus:** Using physical process and controller information for multi-level graphs
- **Key Methods:** PCGAT, physics-informed graph construction
- **Link:** https://www.mdpi.com/2076-0825/14/5/210
- **Relevance:** Shows value of multi-level analysis like your hierarchical refinement

**13. A Novel Method Based on Spatial-Temporal Graph Learning**
- **Focus:** Capturing spatial-temporal dependencies for MTS anomaly detection
- **Key Methods:** Graph convolutional networks, temporal modeling
- **Link:** https://link.springer.com/article/10.1007/s44443-025-00024-3
- **Relevance:** State-of-art spatial-temporal relationship modeling

### 10.4 Deep Learning Anomaly Detection (General Surveys)

**14. A Survey of Deep Anomaly Detection in Multivariate Time Series (2025)**
- **Focus:** Comprehensive taxonomy of deep learning methods for MTSAD
- **Key Methods:** AE, VAE, GAN, Diffusion models, Transformers
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11723367/
- **Link (MDPI):** https://www.mdpi.com/1424-8220/25/1/190
- **Relevance:** Comprehensive baseline methods for comparison

**15. Deep Learning for Anomaly Detection in Multivariate Time Series**
- **Focus:** Challenges and approaches in MTS anomaly detection
- **Key Methods:** RNNs, LSTMs, CNNs, Transformers
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S1566253522001774
- **Relevance:** Understanding limitations of existing deep learning approaches

### 10.5 Causal Discovery & Relationship Learning

**16. Time Series Causal Relationships Discovery Through Feature Importance**
- **Focus:** Using Random Forest feature importance for causal discovery
- **Key Methods:** Ensemble models, feature importance ranking
- **Link:** https://www.nature.com/articles/s41598-023-37929-w
- **Relevance:** Alternative interpretable approach to relationship discovery

**17. Interpretability of Causal Discovery in Tracking Deterioration**
- **Focus:** Causal discovery for industrial process degradation monitoring
- **Key Methods:** Causal graphs, domain expert integration, Jaccard distance
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11207435/
- **Relevance:** Shows value of causal interpretability in industrial settings

**18. Data-Driven Dynamic Causality Analysis of Industrial Systems**
- **Focus:** Combining interpretable ML with process mining for causality
- **Key Methods:** Decision trees, process discovery, event logs
- **Link:** https://link.springer.com/article/10.1007/s10845-021-01903-y
- **Relevance:** Interpretable causality analysis using discrete events

**19. CausalFormer: Interpretable Transformer for Temporal Causal Discovery**
- **Focus:** Deep learning with interpretability for causal discovery
- **Key Methods:** Attention masks, Regression Relevance Propagation
- **Link:** https://www.themoonlight.io/en/review/causalformer-an-interpretable-transformer-for-temporal-causal-discovery
- **Relevance:** State-of-art interpretable deep causal discovery

**20. Causal Discovery from Temporal Data: An Overview (ACM Survey)**
- **Focus:** Comprehensive survey of temporal causal discovery methods
- **Key Methods:** PCMCI, score-based, constraint-based approaches
- **Link:** https://dl.acm.org/doi/10.1145/3705297
- **Relevance:** Theoretical foundations for temporal causal discovery

**21. Detecting and Quantifying Causal Associations in Large Time Series**
- **Focus:** PCMCI method for causal discovery with high-dimensional data
- **Key Methods:** Conditional independence testing, PC algorithm variants
- **Link:** https://www.science.org/doi/10.1126/sciadv.aau4996
- **Relevance:** Well-established causal discovery method for comparison

**22. Causal Inference Meets Deep Learning: A Comprehensive Survey**
- **Focus:** Integration of causal inference with deep learning
- **Key Methods:** Structural causal models, counterfactuals
- **Link:** https://spj.science.org/doi/10.34133/research.0467
- **Relevance:** Future direction for your research

**23. A Survey on Causal Discovery Methods (I.I.D. and Time Series)**
- **Focus:** Comprehensive overview of causal discovery algorithms
- **Key Methods:** Constraint-based, score-based, functional causal models
- **Link:** https://arxiv.org/html/2303.15027v4
- **Relevance:** Theoretical background and comparison frameworks

**24. Exploring Causal Learning Through Graph Neural Networks**
- **Focus:** GNNs for causal learning with explainability
- **Key Methods:** Causal GNNs, explainability techniques
- **Link:** https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70024
- **Relevance:** Integration of causality with GNNs

**25. Causal and Interpretable Rules for Time Series Analysis**
- **Focus:** Rule-based interpretable causal discovery
- **Key Methods:** Association rules, temporal patterns
- **Link:** https://dl.acm.org/doi/10.1145/3447548.3467161
- **Relevance:** Alternative interpretable approach

### 10.6 Tennessee Eastman Process Specific

**26. Fault Detection and Diagnosis of Tennessee Eastman Using Multivariate Control Charts**
- **Focus:** PCA/PLS-based monitoring of TEP
- **Key Methods:** MDMVCC platform, T² and Q statistics
- **Link:** https://www.researchgate.net/publication/357910235_Fault_Detection_and_Diagnosis_of_the_Tennessee_Eastman_Process_using_Multivariate_Control_Charts
- **Relevance:** Direct comparison on same dataset

**27. The Tennessee Eastman Process: An Open Source Benchmark**
- **Focus:** Overview of TEP dataset and its use in anomaly detection research
- **Key Methods:** Data exploration, benchmark description
- **Link:** https://keepfloyding.github.io/posts/Ten-East-Proc-Intro/
- **Relevance:** Understanding the TEP dataset structure

**28. An Extended Tennessee Eastman Simulation Dataset (Reinartz et al.)**
- **Focus:** Extended TEP dataset with more variables and fault types
- **Key Methods:** Updated simulator, comprehensive fault scenarios
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0098135421000594
- **Relevance:** Potential extended dataset for validation

**29. Tennessee Eastman Process Simulation Data (Kaggle)**
- **Focus:** Publicly available TEP data for anomaly detection
- **Link:** https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset
- **Relevance:** Data source

**30. Loading and Exploring the TEP Dataset**
- **Focus:** Practical guide to working with TEP data
- **Key Methods:** Data preprocessing, exploration techniques
- **Link:** https://keepfloyding.github.io/posts/data-explor-TEP-1/
- **Relevance:** Data handling best practices

### 10.7 Change Point & Anomaly Detection

**31. Anomaly and Change Point Detection with Concept Drift**
- **Focus:** Distinguishing between anomalies and change points
- **Key Methods:** Distribution sampling, backward searching
- **Link:** https://link.springer.com/article/10.1007/s11280-023-01181-z
- **Relevance:** Addresses similar challenges in detecting distributional changes

**32. A Survey of Methods for Time Series Change Point Detection**
- **Focus:** Comprehensive overview of change point detection methods
- **Key Methods:** Bayesian, likelihood-based, kernel methods
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC5464762/
- **Relevance:** Related to boundary violation detection

**33. Detecting Structural Changes in Distributions (Distributional Time Series)**
- **Focus:** Change point detection in distributional data
- **Key Methods:** Multiple change-point detection, binary segmentation
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0888327023002510
- **Relevance:** Theoretical foundation for distributional changes

### 10.8 Practical Implementation Resources

**34. Deep Learning-Based Anomaly Detection (GitHub Survey)**
- **Focus:** Curated list of anomaly detection papers with code
- **Key Methods:** Collection of state-of-art methods
- **Link:** https://github.com/bitzhangcy/Deep-Learning-Based-Anomaly-Detection
- **Relevance:** Code implementations for comparison

### 10.9 Related Control Chart Methods

**35. A Multivariate Control Chart for Simultaneously Monitoring Mean and Variability**
- **Focus:** Single chart for monitoring both location and dispersion
- **Key Methods:** EWMA with GLR test
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0167947310001295
- **Relevance:** Traditional control chart approach for comparison

**36. A Multivariate Monitoring Method Based on Dual Control Chart**
- **Focus:** Dual statistics for multivariate monitoring
- **Key Methods:** Deviance variables, dual control chart
- **Link:** https://www.researchgate.net/publication/321282023_A_Multivariate_Monitoring_Method_Based_on_Dual_Control_Chart
- **Relevance:** Alternative multivariate monitoring approach

---

## 10.10 How to Use These Resources

### Priority Reading Order:

**Phase 1: Understanding the Landscape (Surveys)**
1. Survey on GNN for Time Series (#7)
2. Survey of Deep Anomaly Detection in MTS (#14, #15)
3. Causal Discovery from Temporal Data (#20)

**Phase 2: Interpretable Methods (Most Relevant)**
1. Explainable Correlation-Based Anomaly Detection (#1)
2. Time Series Causal Relationships via Feature Importance (#16)
3. Interpretability of Causal Discovery in Industrial Processes (#17)

**Phase 3: Boundary & Threshold Approaches**
1. Deep Learning Survey with SVDD discussion (#4)
2. Leveraging Distributional Statistics (#5)
3. EPASAD boundary-based method (#6)

**Phase 4: State-of-Art Comparison Baselines**
1. MSDG Multi-Scale GNN (#8)
2. TopoGDN (#10)
3. Multi-Level Graph Attention (#12)

**Phase 5: Tennessee Eastman Specific**
1. Open Source Benchmark overview (#27)
2. Data exploration guide (#30)
3. Fault detection with control charts (#26)

### Research Strategy:

**For Literature Review:**
- Start with surveys (#7, #14, #20, #23) to understand taxonomy
- Read 5-10 most cited interpretable methods
- Focus on recent (2024-2025) papers for state-of-art

**For Methodology:**
- Deep dive into boundary-based methods (#4, #5, #6)
- Study multi-scale approaches (#8, #10)
- Understand causal discovery fundamentals (#20, #21)

**For Validation:**
- Study TEP-specific papers (#26-30)
- Identify key baseline methods from surveys
- Note evaluation metrics used in recent papers

**For Future Work:**
- GNN integration possibilities (#7, #8, #9, #10, #12)
- Causal inference extensions (#22, #24)
- LLM integration directions (#22)