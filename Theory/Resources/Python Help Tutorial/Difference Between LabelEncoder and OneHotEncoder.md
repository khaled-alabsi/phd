# Difference Between LabelEncoder and OneHotEncoder

## Purpose

* **LabelEncoder**: Converts categorical labels to integer labels.
* **OneHotEncoder**: Converts categorical labels to one-hot encoded vectors.

---

## Details

| Encoder       | Output Format             | Typical Use Case                                                                                   | Output Example (3 classes: A, B, C)      |
| ------------- | ------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| LabelEncoder  | 1D integer array          | When labels are integers for classification                                                        | A → 0, B → 1, C → 2                      |
| OneHotEncoder | 2D one-hot encoded matrix | When explicit binary vector representation is required (e.g., for `categorical_crossentropy` loss) | A → $$
1,0,0], B → $$
0,1,0], C → $$
0,0,1] |

---

## When to Use Which Encoder

* Use **LabelEncoder** if your model uses `sparse_categorical_crossentropy` loss and expects integer labels.
* Use **OneHotEncoder** if your model uses `categorical_crossentropy` loss and expects one-hot encoded labels.

---

## Code Examples

### LabelEncoder Example

```python
from sklearn.preprocessing import LabelEncoder

labels = ['A', 'B', 'C', 'A', 'B']
le = LabelEncoder()
labels_int = le.fit_transform(labels)  # Output: array([0, 1, 2, 0, 1])

# Use with Keras:
# model.compile(loss='sparse_categorical_crossentropy', ...)
# model.fit(X_train, labels_int, ...)
```

### OneHotEncoder Example

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

labels = np.array(['A', 'B', 'C', 'A', 'B']).reshape(-1, 1)
enc = OneHotEncoder(sparse_output=False)
labels_onehot = enc.fit_transform(labels)
# Output:
# array([
#  [1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.],
#  [1., 0., 0.],
#  [0., 1., 0.]
# ])

# Use with Keras:
# model.compile(loss='categorical_crossentropy', ...)
# model.fit(X_train, labels_onehot, ...)
```

---

## Summary

| Scenario                                                 | Encoder + Loss                                   |
| -------------------------------------------------------- | ------------------------------------------------ |
| Labels are integer encoded, simpler memory use           | LabelEncoder + sparse\_categorical\_crossentropy |
| Labels are one-hot encoded, explicit probability vectors | OneHotEncoder + categorical\_crossentropy        |

