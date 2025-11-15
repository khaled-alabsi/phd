## ⚙️ What Is an Analyzer?

An **analyzer** is a special sensor that **measures chemical composition** of product streams—things like:

* **%G**, **%H**, **%I** (these are product or by-product chemicals)
* **Purity**, **conversion**, or **yield** values

### Unlike normal sensors (pressure, flow, temperature):

* Analyzers don't measure physical properties.
* They **sample chemical content**, often **offline** or **with delay**.
* Think of it like a **lab instrument** that's connected to the process.

---

## 🧪 Where Is It in the Diagram?

In the flowsheet:

* You’ll find the **“Analyzer”** box just **after the separator/stripper**, tapping into the **product stream**.
* It represents a **real-world chemical analyzer** unit.

---

## ⏱ How Does It Behave?

1. **Delayed output:** There's a time lag between sampling and when the result is available.
2. **Low frequency:** It may only update every few minutes.
3. **Noisy values:** Chemical analyzers aren't perfect; they introduce variability.
4. **Sparse signals:** Not all analyzers measure all the time or for all species.

---

## 💡 Why Use It?

Because:

* You **cannot measure composition directly** with flowmeters or thermocouples.
* To **control product quality**, you need **composition feedback** (e.g., %G should be > 90%).

However:

* Because it's **slow**, you can’t use it for fast feedback loops.
* You use it for:

  * **Product quality monitoring**
  * **Performance evaluation**
  * **Fault diagnosis** (e.g., sudden purity drop = possible leak or bad feed)

---

## 🧩 In the Dataset

The composition variables often linked to analyzers are:

* G, H, I concentrations
* Performance indicators like production rate or yield

These variables are either:

* **Synthetic measurements** with delay, or
* **Ground truth** labels for evaluation only (not usable in control)

---

## 🔁 Summary by Analogy

| Device   | Measures             | Update Speed   | Usage                      |
| -------- | -------------------- | -------------- | -------------------------- |
| Sensor   | Pressure/Temp/Flow   | Fast (1s–3s)   | Feedback Control (PID)     |
| Analyzer | Chemical composition | Slow (30–60s+) | Product Quality Monitoring |

---

### ✅ Example: Analyzer Input and Output in Tennessee–Eastman Process

The analyzer does **not directly control** the process; instead, it **receives a sample from the product stream** and **returns chemical composition results**.

---

## 🛠️ Input to the Analyzer

This is the **product stream** leaving the **separator or stripper**—a **mixture of chemicals**.

Example:

* A liquid product stream (e.g., Stream 11) with unknown exact composition.

### It contains:

* Unreacted feed (A, B, C, D, E)
* Desired products (G, H)
* By-products or inerts (I, etc.)

So input is:

```plaintext
Flow stream → mixture of chemical species: A, B, C, D, E, G, H, I
```

---

## 📤 Output from the Analyzer

The analyzer outputs **estimated concentrations or mole fractions** of key species.

Example output (after 60 seconds delay):

```plaintext
%G = 83.2
%H = 15.4
%I = 1.4
```

This tells operators or monitoring systems:

* Product G is the main component (desired).
* H is a secondary product/by-product.
* I is a waste product or impurity.

---

## Dataset Context

In many implementations (e.g., Python simulators or synthetic TE datasets), you’ll find variables like:

```python
analyzer_output = {
    'G_purity': 83.2,
    'H_concentration': 15.4,
    'I_concentration': 1.4
}
```

These are typically:

* Updated less frequently than other sensors (e.g., every 10–60 steps),
* Lagged relative to process changes,
* Used for **supervisory monitoring**, not PID control.

---

To separate a **mixture of chemicals** into its **original components** in a process like Tennessee–Eastman, you use **unit operations** based on physical and thermodynamic properties—**not analyzers**. The analyzer only tells you *what* is in the stream, not *how to split it*. Here's how separation is achieved:

---

## 🔁 Separation Stages in the Tennessee–Eastman Process

### 1. **Separator (Vapor–Liquid Separator)**

* **Function:** Splits the reactor output into **vapor** and **liquid**.
* **How:** Based on **boiling points** and **partial pressures** (flash separation).
* **Result:**

  * Light gases (unreacted feed, by-products) → **vapor phase**
  * Heavier, condensable products (like G, H, I) → **liquid phase**

> Controlled via pressure and condenser cooling (e.g., `XMV(11)`).

---

### 2. **Stripper Column**

* **Function:** Further purifies the **liquid** from the separator.
* **How:** Injects **steam** (`XMV(9)`) from the bottom to strip volatile components upward.
* **Result:**

  * Desired products like G, H stay in the **bottom product**
  * Light volatile impurities (maybe B or E) go to **overhead vapor**

> Controlled by steam flow and bottom flow (`XMV(8)`).

---

## 🧪 Can We Get Back the Exact Original Components?

Not fully. Here's why:

* **Chemical reactions already happened** in the reactor:

  * A + C + D → G + H (simplified)
* The outputs are **not just a mixture**, but a result of **irreversible chemical reactions**.
* Some feeds (like A, D, E) are partially consumed.
* Some components (like G and H) didn’t exist in the feed—they are **products**.

So you cannot *recover all original inputs* (like A, D, E) from the product stream.

---

## ✅ What You *Can* Separate

| Type                          | Can Be Separated? | How                                        |
| ----------------------------- | ----------------- | ------------------------------------------ |
| **Phases** (vapor/liquid)     | Yes               | Flash separator                            |
| **Volatility-based**          | Yes               | Stripper (steam + temperature)             |
| **Components after reaction** | Partially         | By separating unreacted feed from products |
| **Original feed chemicals**   | No (fully)        | Because they’ve reacted                    |

---

## Analyzer Role Again

The analyzer just reports:

```plaintext
%G = 83%, %H = 16%, %I = 1%
```

You can't feed that into a separation unit—it’s only for **monitoring**.

---

