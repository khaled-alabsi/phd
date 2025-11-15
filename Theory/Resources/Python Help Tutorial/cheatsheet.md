# Cheatsheet

## Numpy

### `np.arange()`

**Purpose**: Create an array with regularly spaced values.

**Syntax**:

```python
np.arange([start,] stop[, step])
```

* `start`: First value (default: `0`)
* `stop`: End (exclusive)
* `step`: Spacing (default: `1`)

**Examples**:

```python
np.arange(5)             # → [0, 1, 2, 3, 4]
np.arange(1, 6)          # → [1, 2, 3, 4, 5]
np.arange(0, 10, 3)      # → [0, 3, 6, 9]
np.arange(0, 1, 0.3)     # → [0. , 0.3, 0.6, 0.9]
```

**Behavior**:

* `stop` is **excluded**
* If `step` doesn't divide the interval exactly, the last value is the largest `start + n*step < stop`
* Float steps may cause **precision issues** (e.g., `0.1 + 0.1 + 0.1 ≠ 0.3` exactly)

---

### Use `np.linspace()` when:

* You want a fixed **number of points**
* You want to **include both endpoints**

**Syntax**:

```python
np.linspace(start, stop, num)
```

**Example**:

```python
np.linspace(0, 1, 5)  # → [0. , 0.25, 0.5 , 0.75, 1. ]
```




## Ploting

## 🧠 **Matplotlib Bar Plot – Complete Cheatsheet**

---

### **1. Core Function**

```python
plt.bar(x, height, width=0.8, label=None)
```

* **x**: x-positions (1 per bar)
* **height**: bar heights (same length as x)
* **width**: controls bar thickness (default `0.8`)
* **label**: legend label (used in `plt.legend()`)

> 📌 `plt.bar` plots **one bar per `x` value**.

---

### **2. How Many Bars Will Be Plotted?**

* If `x` has length 5 and `height` has length 5 → 5 bars will be drawn.

```python
x = [0, 1, 2]
height = [10, 20, 30]
plt.bar(x, height)  # 3 bars at x=0,1,2
```

---

### **3. Plotting Multiple Bars per Group (Grouped Bars)**

```python
group_spacing = 3.0
x = np.arange(0, len(groups) * group_spacing, group_spacing)
bar_width = group_spacing * 0.8 / num_models

for i, model in enumerate(models):
    plt.bar(x + i * bar_width, values[model], width=bar_width, label=model)
```

* `x + i * bar_width`: offsets bars for each model inside group
* `group_spacing`: space between groups
* `bar_width`: bar thickness per model

---

### **4. Ticks (`plt.xticks`, `plt.yticks`)**

#### `plt.xticks(tick_positions, tick_labels)`

* Controls **location** and **text** shown on the x-axis

Example:

```python
x = [0, 1, 2]
labels = ["A", "B", "C"]
plt.xticks(x, labels)
```

> 📌 Ticks = where matplotlib draws axis values and labels


To **control the number of grid lines** in a `matplotlib` plot, you must explicitly set the **ticks** on each axis, because grid lines are drawn at tick positions.

---

### ✅ **Control Number of Grid Lines**

#### **1. Set Ticks Explicitly**

```python
plt.yticks(np.linspace(ymin, ymax, n_ticks))
plt.xticks(np.linspace(xmin, xmax, n_ticks))
```

* `np.linspace(start, stop, num_ticks)` gives evenly spaced tick positions.
* Grid lines will match these positions.

#### Example: 5 grid lines on Y-axis between 0 and 1

```python
plt.yticks(np.linspace(0, 1, 5))  # → [0. , 0.25, 0.5 , 0.75, 1. ]
plt.grid(axis='y')                # Grid will follow these ticks
```

---

### ✅ **Alternative: Use `MultipleLocator` (fine control)**

```python
from matplotlib.ticker import MultipleLocator

plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))  # grid every 0.2
plt.grid(axis='y')
```

* Use when you want **fixed spacing** (e.g. every 0.2)
* Works for **integer axes** as well (e.g. `MultipleLocator(1)`)

---

### ✅ Optional: Minor Grid Lines

```python
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', alpha=0.3)
plt.grid(which='major', linestyle='--', alpha=0.7)
```

* Adds lighter, denser grid lines between main ticks
* Use `which='minor'` for more visual aid

---

### 🧠 Summary

| Method                         | Use When                          | Example                                |
| ------------------------------ | --------------------------------- | -------------------------------------- |
| `plt.yticks(np.linspace(...))` | Fixed number of grid lines        | `linspace(0, 1, 6)` for 6 y-grid lines |
| `MultipleLocator(x)`           | Fixed interval between grid lines | `MultipleLocator(0.25)`                |
| `minorticks_on()`              | Add extra light gridlines         | `plt.grid(which='minor')`              |

Grid count = tick count, so control **ticks → gridlines**.

---

### **5. Grid**

```python
plt.grid(axis="y", linestyle="--", alpha=0.5)
```

* `axis`: "x", "y", or "both"
* `linestyle`: dashed `--`, dotted `:`, solid `-`
* `alpha`: transparency (0 → invisible, 1 → solid)

---

### **6. Labels and Titles**

```python
plt.title("My Plot")
plt.xlabel("Fault Type")
plt.ylabel("Accuracy Score")
```

---

### **7. Legend**

```python
plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1.05, 1))
```

* `loc`: position (e.g., "upper right", "lower left")
* `bbox_to_anchor`: place legend *outside* the plot

---

### **8. Layout Control – `plt.tight_layout()`**

* Fixes **overlapping labels, ticks, and legends**
* Automatically adjusts margins and spacing

Use **after all plotting** commands:

```python
plt.tight_layout()
```

---

### **9. Control Figure Size**

```python
plt.figure(figsize=(width_inch, height_inch))
```

Example:

```python
plt.figure(figsize=(12, 4))
```

---

### **10. Show Plot**

```python
plt.show()
```

---

### ✅ Summary Table

| Feature         | Function                                    | Notes                                       |
| --------------- | ------------------------------------------- | ------------------------------------------- |
| Plot bars       | `plt.bar(x, height)`                        | One bar per `x`, height must match          |
| Multiple models | `x + i * bar_width`                         | Use group spacing + offset for grouped bars |
| X ticks         | `plt.xticks(positions, labels)`             | Labels shown at given `x` positions         |
| Y ticks         | `plt.yticks(np.arange(0, 1.1, 0.1))`        | Control tick density/interval               |
| Title/labels    | `plt.title()`, `plt.xlabel()`, ...          | Descriptive axis and plot titles            |
| Grid lines      | `plt.grid(axis="y", linestyle="--")`        | Visual support for comparing bar heights    |
| Legend          | `plt.legend(title=..., bbox_to_anchor=...)` | Place legend outside with `bbox_to_anchor`  |
| Layout fixing   | `plt.tight_layout()`                        | Prevent overlap (ticks, legend, title)      |
| Size control    | `plt.figure(figsize=(w, h))`                | Set plot dimensions in inches               |
| Show            | `plt.show()`                                | Must call to display the figure             |

---

