# **Tennessee Eastman Process (TEP) Variables: Full Overview**

The Tennessee Eastman Process is a benchmark chemical production process widely used in process control and fault detection research. It includes 52 variables:

---

## **Measured Variables (XMEAS$$
1–41])**

These are process outputs, sensors, or analyzer values (continuous or sampled). Units are typically in engineering scale (e.g., % valve open, ppm, kg/hr, °C).

| ID         | Description                                     |
| ---------- | ----------------------------------------------- |
| XMEAS$$
1]  | A Feed (stream 1) flow (kg/hr)                  |
| XMEAS$$
2]  | D Feed (stream 2) flow (kg/hr)                  |
| XMEAS$$
3]  | E Feed (stream 3) flow (kg/hr)                  |
| XMEAS$$
4]  | A + C Feed (stream 4) flow (kg/hr)              |
| XMEAS$$
5]  | Recycle flow (stream 8) (kg/hr)                 |
| XMEAS$$
6]  | Reactor feed rate (stream 6) (kg/hr)            |
| XMEAS$$
7]  | Reactor pressure (kPa)                          |
| XMEAS$$
8]  | Reactor level (%)                               |
| XMEAS$$
9]  | Reactor temperature (°C)                        |
| XMEAS$$
10] | Purge rate (stream 9) (kg/hr)                   |
| XMEAS$$
11] | Product separator temperature (°C)              |
| XMEAS$$
12] | Product separator level (%)                     |
| XMEAS$$
13] | Product separator pressure (kPa)                |
| XMEAS$$
14] | Product separator underflow (stream 10) (kg/hr) |
| XMEAS$$
15] | Stripper level (%)                              |
| XMEAS$$
16] | Stripper pressure (kPa)                         |
| XMEAS$$
17] | Stripper underflow (stream 11) (kg/hr)          |
| XMEAS$$
18] | Stripper temperature (°C)                       |
| XMEAS$$
19] | Stripper steam flow (kg/hr)                     |
| XMEAS$$
20] | Compressor work (kW)                            |
| XMEAS$$
21] | Reactor cooling water outlet temperature (°C)   |
| XMEAS$$
22] | Separator cooling water outlet temperature (°C) |
| XMEAS$$
23] | Component A in stream 6 (%)                     |
| XMEAS$$
24] | Component B in stream 6 (%)                     |
| XMEAS$$
25] | Component C in stream 6 (%)                     |
| XMEAS$$
26] | Component D in stream 6 (%)                     |
| XMEAS$$
27] | Component E in stream 6 (%)                     |
| XMEAS$$
28] | Component F in stream 6 (%)                     |
| XMEAS$$
29] | Component A in stream 9 (%)                     |
| XMEAS$$
30] | Component B in stream 9 (%)                     |
| XMEAS$$
31] | Component C in stream 9 (%)                     |
| XMEAS$$
32] | Component D in stream 9 (%)                     |
| XMEAS$$
33] | Component E in stream 9 (%)                     |
| XMEAS$$
34] | Component F in stream 9 (%)                     |
| XMEAS$$
35] | Component D in stream 11 (%)                    |
| XMEAS$$
36] | Component E in stream 11 (%)                    |
| XMEAS$$
37] | Component F in stream 11 (%)                    |
| XMEAS$$
38] | Component A in stream 11 (%)                    |
| XMEAS$$
39] | Component B in stream 11 (%)                    |
| XMEAS$$
40] | Component C in stream 11 (%)                    |
| XMEAS$$
41] | Sampled component F in stream 11 (ppm)          |

---

## **Manipulated Variables (XMV$$
1–11])**

These are control inputs or actuator settings, generally adjusted by PID or MPC controllers.

| ID       | Description                                  |
| -------- | -------------------------------------------- |
| XMV$$
1]  | D Feed flow (stream 2) (% valve opening)     |
| XMV$$
2]  | E Feed flow (stream 3) (% valve opening)     |
| XMV$$
3]  | A + C Feed flow (stream 4) (% valve opening) |
| XMV$$
4]  | Recycle flow (stream 8) (% valve opening)    |
| XMV$$
5]  | Reactor pressure setpoint (kPa)              |
| XMV$$
6]  | Reactor cooling water flow (% valve)         |
| XMV$$
7]  | Condenser cooling water flow (% valve)       |
| XMV$$
8]  | Compressor recycle valve (% valve)           |
| XMV$$
9]  | Purge valve (% valve)                        |
| XMV$$
10] | Separator product rate (stream 10) (% valve) |
| XMV$$
11] | Stripper steam valve (% valve)               |

---

## Notes for Fault Detection

* TEP includes 21 fault scenarios (IDV = 1–21).
* Variables are strongly correlated and non-linear.
* Key for multivariate control chart analysis, PCA, ICA, neural network monitoring, and anomaly detection.
