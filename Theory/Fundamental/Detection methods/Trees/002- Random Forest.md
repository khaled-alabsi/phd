# Random Forests 

##  Decision Trees & Random Forests: A Consolidated Guide

This guide breaks down the components of Random Forests, starting from the basic building block (a single Decision Tree) and moving to the full ensemble model.

***

### 1. The Decision Tree: Core Concepts

A **Decision Tree** is a model that recursively splits a dataset based on feature values to make predictions. Think of it as a flowchart of questions, where each internal node asks a question about a feature, and each leaf node provides a final answer. The primary goal during this process is to create splits that result in the most homogeneous (or "pure") subsets of data.

#### Key Components
* **Node**: An internal point in the tree that represents a test on a feature (e.g., "Is `feature_X` less than `threshold_Y`?"). It directs data to a left or right child node.
* **Leaf**: A terminal node that holds the final prediction. For classification, this is the most common class label. For regression, it's the average of the target values.
* **Splitting Criterion**: A function used to measure the quality of a split. The algorithm seeks t^o maximize the "purity gain" at each split.

---

### 2. Measuring Impurity: The Splitting Criterion

To decide the best feature and threshold for a split, a decision tree needs to quantify the "impurity" of a node. A lower impurity score means the data in the node is more uniform.

Think of "impurity" like a box of colored balls. If the box has only red balls, it's "pure"—very easy to describe, nothing special stands out. If the box has a mix of red, blue, and green balls, it's "impure"—more mixed, less clear, and harder to describe with one word.

In a decision tree, impurity tells us how mixed the data is at a split. The goal is to find features (questions) that split the box into groups where each group is as "pure" as possible (mostly one color). The more a feature helps separate the colors, the more important or "special" that feature is for making decisions. So, features that reduce impurity the most are considered the most important for the tree.

Let’s use a simple example with two features and three classes as the target.

Suppose you have a dataset of fruits. Each fruit has:
- Feature 1: Color (Red, Green, Yellow)
- Feature 2: Size (Small, Medium, Large)
- Target (Class): Apple, Banana, or Grape

Example data:
| Color  | Size   | Fruit   |
|--------|--------|---------|
| Red    | Small  | Grape   |
| Red    | Medium | Apple   |
| Green  | Small  | Grape   |
| Yellow | Large  | Banana  |
| Green  | Medium | Apple   |
| Yellow | Medium | Banana  |
| Red    | Large  | Apple   |
| Green  | Large  | Apple   |
| Yellow | Small  | Banana  |

If you split the data by Color:
- Red group: Grape, Apple, Apple (2 Apples, 1 Grape)
- Green group: Grape, Apple, Apple (2 Apples, 1 Grape)
- Yellow group: Banana, Banana, Banana (all Bananas)

The Yellow group is pure (only Bananas), but Red and Green are mixed (higher impurity).

If you split by Size:
- Small group: Grape, Grape, Banana (2 Grapes, 1 Banana)
- Medium group: Apple, Apple, Banana (2 Apples, 1 Banana)
- Large group: Banana, Apple, Apple (2 Apples, 1 Banana)

All groups are mixed (higher impurity).

So, splitting by Color gives you at least one pure group (Yellow), while splitting by Size gives you only mixed groups. This means "Color" is a more important feature for reducing impurity in this example.

In summary: impurity measures how mixed the classes are after a split. The feature that creates the purest groups (least mixed classes) is the most important for the tree at that step.

#### For Classification
* **Gini Impurity**: Measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the subset. A Gini score of 0 represents perfect purity.
    $$
Gini = 1 - \sum_{i=1}^{C} p_i^2$$
    where $p_i$ is the proportion of samples belonging to class $i$.

* **Entropy**: Measures the level of disorder or uncertainty in a node. It is based on information theory.
    $$
Entropy = - \sum_{i=1}^{C} p_i \log_2 p_i$$

#### For Regression
* **Variance Reduction**: Measures the decrease in the variance of the target variable after a split. The goal is to find splits that create child nodes with lower variance than the parent node.
    $$
Var(S) = \frac{1}{|S|} \sum_{x \in S} (y - \bar{y})^2$$

---

### 3. Building a Single Decision Tree

The process of building a tree is **greedy**, meaning it makes the best possible decision at each step without considering future splits.

#### Algorithm Steps
1.  **Start with the full dataset** $(X, y)$ at the root node.
2.  **Find the best split**:
    * Iterate through all features.
    * For each feature, iterate through potential threshold values.
    * Calculate the impurity gain (e.g., reduction in Gini impurity or variance) for each potential split.
    * Select the feature and threshold that provide the **highest gain**.
3.  **Check stopping conditions**: Do not split a node if:
    * The maximum allowed depth (`max_depth`) has been reached.
    * The number of samples in the node is below a minimum threshold (`min_samples_split`).
    * The node is **pure** (contains samples from only one class).
    * Splitting the node does not lead to a sufficient improvement in impurity.
4.  **Recurse**: If no stopping condition is met, split the data into left and right subsets based on the best split found. Recursively repeat the process for each new child node.
5.  **Create leaves**: When a stopping condition is met, the node becomes a leaf. The prediction at this leaf is the **majority class** (for classification) or the **mean value** (for regression) of the samples in that node.

A single, deep decision tree is prone to **overfitting** because it can create complex rules that memorize the training data but fail to generalize to new, unseen data.

---

### 4. The Random Forest: An Ensemble of Trees

A **Random Forest** is an ensemble learning method that builds multiple decision trees and merges their predictions to produce a more accurate and stable result. It introduces two key sources of randomness to ensure the trees are different from one another, which is crucial for the model's performance.

#### Core Randomization Mechanisms
1.  **Bootstrap Sampling (Instance Randomization)**: For each of the `n_estimators` trees, a new training set is created by sampling from the original dataset **with replacement**. This is called a **bootstrap sample**.
    * **Effect**: Each tree is trained on a slightly different subset of the data. On average, about two-thirds of the original data appears in any given bootstrap sample. The remaining one-third forms the **Out-of-Bag (OOB)** sample.

2.  **Random Feature Selection (Feature Randomization)**: When building each tree, at **every split**, the algorithm considers only a random subset of the available features.
    * **Typical default values for `max_features`**:
        * **Classification**: `sqrt(total_features)`
        * **Regression**: `total_features / 3`
    * **Effect**: This prevents any single feature from dominating all the trees. It **decorrelates** the trees, forcing them to learn a wider variety of patterns in the data. This reduces the overall model variance significantly at the cost of a slight increase in bias.

> **Warning**: An ensemble model built with bootstrapping but **without** random feature selection is a **Bagging** ensemble, not a Random Forest. The feature randomization at each split is the key ingredient.

#### Building and Prediction
1.  **Training**: For `n_estimators` iterations:
    * Create a bootstrap sample of the data.
    * Build a decision tree on this sample, using the random feature selection method at each split.
    * Store the trained tree.
2.  **Prediction**: To make a prediction for a new sample:
    * **Classification**: Each tree in the forest "votes" for a class. The final prediction is the class with the **majority vote**.
    * **Regression**: Each tree predicts a numerical value. The final prediction is the **average** of all individual tree predictions.

---

### 5. Practical Considerations & Hyperparameters

#### Key Hyperparameters
* `n_estimators`: The number of trees in the forest. More trees generally improve performance, but with diminishing returns.
* `max_features`: The size of the random subset of features to consider at each split.
* `max_depth`: The maximum depth of each tree. Deeper trees can overfit.
* `min_samples_split`: The minimum number of samples required to split an internal node.
* `bootstrap`: A boolean indicating whether to use bootstrap sampling. For a Random Forest, this should be `True`.

#### Important Properties
* **Generalization**: Individual trees are high-variance, low-bias models that overfit. By averaging their predictions, the Random Forest becomes a low-variance model that **generalizes well**.
* **Parallelism**: Since each tree is built independently, the training process can be easily parallelized across multiple CPU cores.
* **Out-of-Bag (OOB) Error**: The data not included in a tree's bootstrap sample (the OOB data) can be used as a built-in validation set to estimate the model's generalization error without needing a separate test set.

---

### 6. Data Structures (Pseudocode)

```python
# Represents a single node or leaf in a tree
class TreeNode:
    is_leaf: bool
    prediction: Any  # Class label or mean value
    split_feature_index: int
    threshold: float
    left_child: 'TreeNode'
    right_child: 'TreeNode'

# Represents a single decision tree
class DecisionTree:
    root: TreeNode

    def fit(data, labels):
        # Implements the recursive splitting algorithm
        ...
    def predict(sample):
        # Traverses the tree from the root to a leaf
        ...

# Represents the entire forest
class RandomForest:
    trees: List[DecisionTree]

    def fit(data, labels):
        # Builds n_estimators trees using randomization
        ...
    def predict(sample):
        # Aggregates predictions from all trees
        ...
```