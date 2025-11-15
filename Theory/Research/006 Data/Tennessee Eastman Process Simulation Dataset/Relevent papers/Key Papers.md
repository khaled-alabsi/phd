Here are selected research works that used the Tennessee Eastman (TE) process explicitly for **data classification** or **process monitoring**, excluding those focused solely on anomaly detection:

---

### 📌 Key Papers on Classification/Monitoring (not just anomaly detection)

1. **Bayesian Classifiers Applied to the Tennessee Eastman Process** (Santos et al., 2013)
   – Introduces dynamic Markov blanket classifiers, comparing against Naïve Bayes & TAN. Demonstrates high classification accuracy and interpretable variable relationships on TE data ([Wiley Online Library][1]).

2. **A Fault Diagnosis Model for TE Processes Based on Feature Selection and Probabilistic Neural Network** (Xu et al., 2022)
   – Uses heuristic feature selection (e.g., SVM-RFE) plus a PNN optimized with a bio-heuristic algorithm for multi-class fault classification on TE ([MDPI][2]).

3. **Feature Selection for Fault Detection Systems: Application to the Tennessee Eastman Process** (Chebel‑Morello & Malinowski, 2015)
   – Applies selection of most relevant variables and evaluates classification performance over selected fault types ([SpringerLink][3]).

4. **A Practical Application of Detection‑Based Multiclass Classification of Faults: TE Process** (Basha et al., AIChE 2020)
   – Combines PCA‑based process monitoring with detection methods and maps them into multiclass fault classification, comparing to deep learning .

5. **Hierarchical Deep Recurrent Neural Network Method for Fault Detection and Diagnosis** (Agarwal et al., 2020)
   – Proposes Supervised Deep Recurrent Autoencoder (hierarchical) for classification of incipient and non‑incipient faults on TE ([arXiv][4]).

6. **Online Fault Detection and Classification Using SPC + Riemannian Geometry** (Miraliakbar et al., 2025)
   – Presents the FARM framework, achieving \~82–84.5% online fault classification accuracy on TE ([arXiv][5]).

7. **Supervised Local Neural Network Classifiers via EM Clustering for TE Diagnosis** (Ayubi Rad & Yazdanpanah, 2015)
   – Builds supervised NN classifiers using clustering to improve multiclass fault diagnosis on TE ([SpringerLink][6]).

8. **RBF Networks + Wavelet-based Fault Diagnosis for TEP** (Liu et al., 2012)
   – Uses radial basis function networks with wavelet feature extraction for classification of TE process faults ([Scientific.Net][7]).

9. **Logical Analysis of Data (LAD) for Fault Diagnosis on TE Process** (Mortada et al., 2017)
   – Applies rule-based Boolean pattern mining for multiclass fault classification, providing interpretable patterns on TE ([ScienceDirect][8]).

---

### 🧭 Summary by Approach

| Methodology                         | Example Papers                                                                                                                                                        |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Probabilistic / Bayesian**        | Santos et al. (Bayesian networks) ([SAGE Journals][9])                                                                                                                |
| **Neural Networks / Deep Learning** | Xu et al. (PNN), Agarwal et al. (Deep RNN), Ayubi Rad & Yazdanpanah (EM+NN), Liu et al. (RBF+Wavelet) ([MDPI][2], [arXiv][4], [SpringerLink][6], [Scientific.Net][7]) |
| **Statistical / PCA-based**         | Basha et al., Chebel‑Morello & Malinowski ([SpringerLink][3])                                                                                                         |
| **Hybrid SPC + Geometry**           | Miraliakbar et al. (SPC + Riemannian) ([arXiv][5])                                                                                                                    |
| **Rule‑Based Boolean Miners**       | Mortada et al. (LAD) ([ScienceDirect][8])                                                                                                                             |

---

### ✅ Recommendation

* **For interpretable/classical methods**, start with **Bayesian classifiers** (Santos et al.) or **LAD rule-based** classifiers (Mortada et al.).
* **For advanced ML**, consider **hierarchical deep RNN** (Agarwal et al.) or **FARM’s SPC + Riemannian** hybrid (Miraliakbar et al.) for strong online classification results.
* **Feature-focused approaches** (Chebel‑Morello; Xu et al.) effectively reduce dimensionality and boost classifier performance.

Let me know if you'd like deeper summaries, datasets, code availability, or help implementing one of these!

[1]: https://onlinelibrary.wiley.com/doi/10.1111/risa.12112?utm_source=chatgpt.com "Bayesian Classifiers Applied to the Tennessee Eastman Process - Santos - 2014 - Risk Analysis - Wiley Online Library"
[2]: https://www.mdpi.com/2076-3417/12/17/8868?utm_source=chatgpt.com "A Fault Diagnosis Model for Tennessee Eastman Processes Based on Feature Selection and Probabilistic Neural Network"
[3]: https://link.springer.com/article/10.1007/s10489-015-0694-6?utm_source=chatgpt.com "Feature selection for fault detection systems: application to the Tennessee Eastman process | Applied Intelligence"
[4]: https://arxiv.org/abs/2012.03861?utm_source=chatgpt.com "Hierarchical Deep Recurrent Neural Network based Method for Fault Detection and Diagnosis"
[5]: https://arxiv.org/abs/2504.01276?utm_source=chatgpt.com "Online Fault Detection and Classification of Chemical Process Systems Leveraging Statistical Process Control and Riemannian Geometric Analysis"
[6]: https://link.springer.com/article/10.1007/s10845-021-01742-x?utm_source=chatgpt.com "Fault classification in the process industry using polygon generation and deep learning | Journal of Intelligent Manufacturing"
[7]: https://www.scientific.net/AMR.546-547.828?utm_source=chatgpt.com "Fault Diagnosis System of Tennessee-Eastman Process Based on RBF Networks and Wavelet | Scientific.Net"
[8]: https://www.sciencedirect.com/science/article/abs/pii/S0957417417307984?utm_source=chatgpt.com "Fault diagnosis in industrial chemical processes using interpretable patterns based on Logical Analysis of Data - ScienceDirect"
[9]: https://journals.sagepub.com/doi/10.1177/0959651818764510?utm_source=chatgpt.com "Model-based fault detection and diagnosis of complex chemical processes: A case study of the Tennessee Eastman process - Khaoula Tidriri, Nizar Chatti, Sylvain Verron, Teodor Tiplica, 2018"
