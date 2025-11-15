# **approach steps** presented in the paper "*Fault detection using CUSUM based techniques with application to the Tennessee Eastman Process (TEP)*" (MoMT4‑02) ([nt.ntnu.no][1]):


## 1. **Problem setup and motivation**

* The focus is on detecting **three specific faults** in the Tennessee Eastman Process (TEP) that prior methods (e.g. PCA, DPCA, CVA) failed to detect because the deviations were **statistically very small** ([nt.ntnu.no][1]).
* These are:

  * IDV(3): step change in D feed temperature (affects reactor cooling water flow)
  * IDV(9): random variation in D feed temperature (reactor cooling outlet temperature)
  * IDV(15): condenser cooling water valve stiction (manipulated flow) ([nt.ntnu.no][1])

---

## 2. **CUSUM-based detection methods**

* Utilize **location CUSUM (LCS)** for detecting small shifts in the **mean** (used for IDV(3)), and **scale CUSUM (SCS)** for small increases in **variance** (used for IDV(9) and IDV(15)) ([nt.ntnu.no][1]).
* For each of the three faults, a **relevant variable** is selected based on process knowledge:

  * LCS on XMV$$
10] for IDV(3)
  * SCS on XMEAS$$
21] for IDV(9)
  * SCS on XMV$$
11] for IDV(15) ([nt.ntnu.no][1])

---

## 3. **CUSUM control chart statistic definition**

* The CUSUM accumulates deviations over time using formulas like:

  ```text
  Ci+ = max(0, Ci−1 + (xi − μ0)/k − slack)
  Ci− = max(0, Ci−1 − (xi − μ0)/k − slack)
  ```

  * LCS monitors mean shifts, SCS monitors variance changes via transformed standardized data ([nt.ntnu.no][1]).

* The statistic signals a fault when it **exceeds a threshold H**, calibrated to target Type I error (false alarm rate) and acceptable ARLₒ.c (average run length under fault) ([nt.ntnu.no][1]).

---

## 4. **Chart implementation & Average Run Length (ARLₒ.c)**

* Sampling is done every **3 minutes** (sampling frequency 1/180 Hz).
* Faults are introduced at sample 160 (\~8 hours into the operation). The expected time to detection (ARLₒ.c) is then estimated from that point via simulations ([nt.ntnu.no][1]).
* Reported ARLₒ.c results:

  | Fault ID | Statistic | ARLₒ.c (hours) |                   |
  | -------- | --------- | -------------- | ----------------- |
  | IDV(3)   | LCS       | \~127 h        |                   |
  | IDV(9)   | SCS       | \~8.2 h        |                   |
  | IDV(15)  | SCS       | \~41 h         | ([nt.ntnu.no][1]) |

---

## 5. **Combined monitoring via Hotelling’s T²**

* To consolidate monitoring, the three individual CUSUMs (LCS for IDV(3), SCS for IDV(9 & 15)) are combined into a **single multivariate T² chart**.

* The vector of cumulative sums is fed into Hotelling’s T² statistic:



  where *x* is the vector of three CUSUM values and *S* is the covariance matrix ([nt.ntnu.no][1]).

* This combined chart successfully detects the individual and simultaneous occurrence of the faults. ARLₒ.c under T² for different scenarios:

  | Fault(s)         | T²‑based ARLₒ.c (hours) |                   |
  | ---------------- | ----------------------- | ----------------- |
  | IDV(3)           | \~102.4 h               |                   |
  | IDV(9)           | \~276 h                 |                   |
  | IDV(15)          | \~89.7 h                |                   |
  | IDV(3) & IDV(15) | \~41.3 h                | ([nt.ntnu.no][1]) |

---

## 6. **Control–detection interaction and tuning trade-offs**

* Since detection is based on manipulated or feedback variables within closed‑loop control loops, **retuning controllers** can speed up detection (decrease ARLₒ.c).
* However, faster detection generally leads to **increased process variability**, which may degrade performance or cause actuator wear.
* The paper demonstrates this trade-off for IDV(15): increasing proportional gain reduces ARLₒ.c but increases variance in manipulated variable XMV$$
11] ([nt.ntnu.no][1]).

---

### 🔍 Recap of the Core Steps

1. **Select fault-relevant variables** based on process knowledge.
2. **Apply univariate CUSUM (LCS or SCS)** to each variable.
3. **Estimate ARLₒ.c** via simulation to gauge detection speed.
4. **Combine the three CUSUMs** using a multivariate T² chart for unified monitoring.
5. **Examine controller tuning effects** to balance detection speed vs variability.


[1]: https://www.nt.ntnu.no/users/skoge/prost/proceedings/dycops-2010/Papers_DYCOPS2010/MoMT4-02.pdf?utm_source=chatgpt.com "[PDF] Fault Detection Using CUSUM Based Techniques with Application ..."
