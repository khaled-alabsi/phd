### `fig` and `ax` in Matplotlib — Complete Guide

In Matplotlib, plotting with `fig` and `ax` gives you **precise control** over plots and is preferred for multiple subplots or customization.

---

### 1. What are `fig` and `ax`?

```python
fig, ax = plt.subplots()
```

* `fig`: the **figure** object — the entire window or image that holds all your plots.
* `ax`: the **axes** object — the actual plot area (can be one or many).

---

### 2. Basic Example (Single Plot)

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])  # Draws a line
ax.set_title("Simple Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()
```

#### `ax.plot([1, 2, 3], [4, 5, 6])` explained

This line draws a **line plot** on the given subplot (`ax`) using the `plot()` method.

Parameters:

* **`[1, 2, 3]`** → x-values (horizontal axis)
* **`[4, 5, 6]`** → y-values (vertical axis)

This plots the following points:

* (1, 4)
* (2, 5)
* (3, 6)

Matplotlib automatically connects these points with straight lines, so you get a line going through those three points.



What happens under the hood:

1. `ax.plot(...)` tells Matplotlib:

   * "On this axes (`ax`), plot these points."
2. It scales the x-axis and y-axis automatically based on the input.
3. It draws a line connecting the points in order.

Visual result:

You’ll see a straight line going from:

* (1,4) → (2,5) → (3,6)

This is a simple **line plot**. You can customize it by adding style:

```python
ax.plot([1, 2, 3], [4, 5, 6], color='red', linestyle='--', marker='o')
```

This would draw:

* a red dashed line
* with circular markers at each point

---

### 3. Multiple Subplots

```python
fig, ax = plt.subplots(2, 3)  # 2 rows, 3 columns of subplots
```

Now `ax` is a 2D NumPy array:

```python
ax[0, 0]  # First row, first column
ax[1, 2]  # Second row, third column
```

You can do:

```python
ax[0, 0].plot(...)
ax[1, 2].set_title(...)
```

---

### 4. Looping Over Subplots

```python
fig, ax = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        ax[i, j].plot([0, 1], [i, j])
        ax[i, j].set_title(f"Plot {i},{j}")
```

---

### 5. Flattening `ax`

If you're unsure of how many plots you’ll have, or want to loop easily:

```python
fig, ax = plt.subplots(2, 2)
ax = ax.flatten()
for i in range(4):
    ax[i].plot([0, 1], [i, i+1])
```

Flattening `ax` in Matplotlib — What, Why, and How

When you use `plt.subplots(nrows, ncols)` and ask for **multiple subplots**, `ax` becomes a **2D NumPy array** of Axes objects. Flattening it turns this 2D array into a 1D array for **easier iteration or access**.

Example: Before Flattening

```python
fig, ax = plt.subplots(2, 2)  # ax.shape == (2, 2)
```

This gives you a layout like:

```
ax[0, 0]   ax[0, 1]
ax[1, 0]   ax[1, 1]
```

Accessing each subplot requires two indices.

Problem

If you want to loop over all axes easily:

```python
for i in range(4):
    ax[i]  # ❌ Fails: ax is 2D
```

Solution: Flatten it

```python
ax = ax.flatten()  # Now ax is a flat array: shape (4,)
```

You can now write:

```python
for i in range(4):
    ax[i].plot([0, 1], [i, i+1])
```

This works regardless of the subplot layout, as long as you flatten.

Summary

* **Before**: `ax[i, j]` — 2D indexing
* **After**:  `ax[i]` — 1D indexing (simpler for loops)

Use `ax.flatten()` when:

* You create a grid of subplots
* You want to iterate over them in a single loop


Let's break down the **difference between `.ravel()` and `.flatten()`** clearly, using simple structure and visuals.



🔍 WHAT are `.ravel()` and `.flatten()`?

Both convert a multi-dimensional NumPy array (like your `ax`) into a **1D array**:

```python
ax.ravel()     # Returns a flattened view (not a copy)
ax.flatten()   # Returns a flattened copy (a new object)
```



 HOW do they differ?

| Feature           | `.ravel()`                    | `.flatten()`                       |
| ----------------- | ----------------------------- | ---------------------------------- |
| Returns           | **View** (linked to original) | **Copy** (independent of original) |
| Memory allocation | No new array (when possible)  | Always allocates new memory        |
| Performance       | Faster, no memory duplication | Slightly slower, uses more memory  |
| Changes to result | Affect original array         | Do **not** affect original array   |



Example to show difference:

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
r = a.ravel()
f = a.flatten()

r[0] = 100
f[1] = 200

print(a)
# Output shows:
# [[100   2]
#  [  3   4]]   ← ravel() affected `a`, flatten() did not
```


WHY it matters for `plt.subplots()`:

When you write:

```python
fig, ax = plt.subplots(2, 2)  # ax is a 2D array of Axes
ax = ax.ravel()               # or ax.flatten()
```

Both **work fine** for looping and plotting. But:

* Use `.ravel()` when you don’t need a new independent object
* Use `.flatten()` only if you want to modify it separately from the original

Summary: Which to use?

| If you want to...              | Use            |
| ------------------------------ | -------------- |
| Loop through axes efficiently  | `ax.ravel()`   |
| Create a separate copy of axes | `ax.flatten()` |

In 99% of Matplotlib cases: **use `.ravel()`** — it's faster and sufficient.

---

### 6. Summary of `ax` methods

Use `ax.` to:

* `.plot(x, y)`
* `.scatter(x, y)`
* `.bar(x, height)`
* `.set_title("...")`
* `.set_xlabel("...")`, `.set_ylabel("...")`
* `.axhline(y, ls='--')`, `.axvline(x)`

---

### Why use `fig`, `ax`?

* More powerful than `plt.plot(...)`
* Easier subplot layout
* Clean, object-oriented code

