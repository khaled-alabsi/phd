### **Minimum Spanning Tree (MST) Analysis**

Minimum Spanning Tree (MST) analysis is a graph-based technique used to simplify and visualize complex networks by reducing them to their most essential connections while preserving overall structure. It has applications in various fields, such as clustering, finance, biology, and network optimization.

---

### **What is a Minimum Spanning Tree (MST)?**

- A **spanning tree** of a graph is a subset of its edges that connects all the nodes (vertices) without forming cycles.
- A **minimum spanning tree (MST)** is a spanning tree with the smallest possible total edge weight.

For a graph $G = (V, E)$ :
- $V$ : Set of vertices (nodes).
- $E$ : Set of edges, each with an associated weight $w(e)$.

The MST minimizes the total weight:
$$
\text{Total Weight} = \sum_{e \in E_{\text{MST}}} w(e)
$$
where $E_{\text{MST}} \subseteq E$ is the set of edges in the minimum spanning tree.

---

### **Steps for MST Analysis**

#### 1. Represent Data as a Weighted Graph
- Nodes ($V$ ) represent the entities or variables.
- Edges ($E$ ) represent relationships between nodes, with weights reflecting the strength or cost of the connection (e.g., distances, correlations, or mutual dependencies).

#### 2. Compute Pairwise Weights
- For numerical datasets, edge weights are typically calculated as:
  - **Euclidean distance** between data points.
  - **Dissimilarity** (e.g., $1 - \text{correlation}$ ) for correlation-based datasets.
- For similarity-based networks, weights might directly represent closeness or strength of similarity.

#### 3. Apply an MST Algorithm
Common algorithms to compute MST include:
- **Prim's Algorithm**: Greedily grows the MST by adding the smallest edge that connects a new node to the tree.
- **Kruskal's Algorithm**: Sorts all edges by weight and adds the smallest edge that doesn’t form a cycle.

#### 4. Interpret the Resulting Tree
- The MST captures the most important connections in the dataset.
- Weak, redundant, or indirect edges are removed, reducing noise while preserving the primary structure.

---

### Detailed Applications of Minimum Spanning Tree (MST) Analysis

MST analysis is widely applied across multiple disciplines due to its ability to simplify and structure data by preserving the most important connections. Below are the detailed applications of MST analysis in various domains.

---

### **1. Clustering and Pattern Recognition**

#### Application:
- MSTs are used to identify clusters of related data points or variables.
- By removing edges with high weights, clusters can be formed from the remaining tree structure.

#### Example:
- **Hierarchical Clustering**: MST-based clustering cuts the tree into subtrees by removing the most significant edges, identifying distinct groups in the data.
- **Image Segmentation**: In computer vision, MSTs help group similar pixels or regions into coherent segments, such as in medical image analysis or object detection.

#### Benefits:
- Handles non-convex shapes of clusters.
- Preserves important relationships within the clusters.

---

### **2. Network Simplification and Visualization**

#### Application:
- In large networks, MSTs help reduce complexity by representing only the most essential connections.

#### Examples:
- **Biological Networks**:
  - MSTs are used to simplify **protein interaction networks** or **genetic regulatory networks**, identifying critical connections between entities.
- **Social Networks**:
  - MSTs help visualize the most influential relationships or communities within a social network.

#### Benefits:
- Reduces noise and redundant information.
- Provides an interpretable and visually appealing representation of the network.

---

### **3. Feature Selection and Dimensionality Reduction**

#### Application:
- MSTs help identify key features or variables by emphasizing connections that represent strong dependencies.

#### Examples:
- **In Machine Learning**:
  - MSTs highlight the most connected features in high-dimensional data, which can then be selected for training models.
- **Genomics**:
  - MSTs help select subsets of genes or biomarkers that are most informative for understanding diseases or predicting outcomes.

#### Benefits:
- Improves computational efficiency by reducing data dimensions.
- Enhances model performance by focusing on the most relevant features.

---

### **4. Finance and Economics**

#### Application:
- MSTs analyze relationships in financial markets, helping to understand dependencies between assets or economic indicators.

#### Examples:
- **Stock Market Analysis**:
  - MSTs are used to study correlations between stock prices. The tree reveals which stocks are most strongly related, helping portfolio managers understand diversification opportunities.
- **Currency Networks**:
  - MSTs are applied to exchange rates between currencies, identifying central currencies in the market and understanding exchange dynamics.

#### Benefits:
- Detects clusters of assets with high correlations.
- Identifies market leaders or influential assets.

---

### **5. Anomaly Detection**

#### Application:
- MSTs help identify anomalies or outliers by focusing on long or weak edges in the tree.

#### Examples:
- **In Cybersecurity**:
  - MSTs detect anomalous behavior in network traffic, where unusual connections between nodes indicate potential threats.
- **In Industrial Monitoring**:
  - MSTs are used in manufacturing to detect sensor faults or unusual machine behavior by examining weak connections or deviations in the tree structure.

#### Benefits:
- Reduces false positives by focusing on structural deviations.
- Provides interpretable results for anomaly identification.

---

### **6. Transportation and Logistics**

#### Application:
- MSTs optimize routes and connections, reducing the overall cost of transportation or delivery.

#### Examples:
- **Network Design**:
  - MSTs help design efficient **road networks**, **railway systems**, or **utility grids** by minimizing construction costs while ensuring connectivity.
- **Supply Chain Optimization**:
  - MSTs optimize delivery routes by minimizing travel distances or transportation costs.

#### Benefits:
- Reduces costs while maintaining full connectivity.
- Balances efficiency and simplicity in network design.

---

### **7. Evolutionary Biology and Phylogenetics**

#### Application:
- MSTs help reconstruct evolutionary trees and analyze the relationships between species.

#### Examples:
- **Phylogenetic Tree Construction**:
  - MSTs represent evolutionary relationships between species, based on genetic or morphological similarities.
- **Gene Flow Analysis**:
  - MSTs model how genes are exchanged between populations, identifying the most likely ancestral paths.

#### Benefits:
- Provides a simplified, interpretable structure for complex biological data.
- Highlights evolutionary or genetic similarities effectively.

---

### **8. Energy and Power Systems**

#### Application:
- MSTs are used to design and analyze efficient power grids.

#### Examples:
- **Power Distribution Networks**:
  - MSTs help identify the shortest, most efficient paths for electricity distribution while minimizing energy loss.
- **Smart Grids**:
  - MSTs simplify and analyze connections in smart grids to optimize resource allocation.

#### Benefits:
- Reduces power loss and operational costs.
- Ensures robust and reliable connectivity.

---

### **9. Telecommunications**

#### Application:
- MSTs are used to optimize communication networks and minimize costs.

#### Examples:
- **Network Design**:
  - MSTs ensure all nodes (e.g., routers, base stations) are connected with the minimum infrastructure cost.
- **Routing Protocols**:
  - MSTs improve efficiency in routing algorithms for data packet delivery.

#### Benefits:
- Minimizes construction and maintenance costs.
- Improves network efficiency and reliability.

---

### **10. Data Compression**

#### Application:
- MSTs help identify the most important features or relationships, leading to efficient data compression techniques.

#### Examples:
- **In Image Compression**:
  - MSTs group pixels or regions with similar properties, reducing redundancy in data storage.
- **Signal Processing**:
  - MSTs simplify complex signals, retaining only the most critical connections for reconstruction.

#### Benefits:
- Reduces data size without losing essential information.
- Speeds up processing and storage tasks.

---

Here is the revised, **numerical detailed example**, step-by-step, including calculations, Python code, and plots.

---

### befor we start in example we have to understand  Kruskal's Algorithm


### Step-by-Step Explanation of Kruskal’s Algorithm (Numerical + Visualization)

Kruskal's Algorithm is used to find the **Minimum Spanning Tree (MST)** of a connected, weighted graph. It works by:

1. Sorting edges by weight.
2. Adding edges to the MST if they don’t form a cycle, using the Union-Find data structure.
3. Repeating until the MST contains $n-1$ edges ($n$ is the number of vertices).

---

#### Graph Data
Let's consider a simple graph:

| Edge  | Weight |
|-------|--------|
| AB    | 1      |
| AC    | 5      |
| BC    | 4      |
| BD    | 3      |
| CD    | 2      |

#### Step-by-Step Process

---

### **Step 1: Sort edges by weight**
Sorted edges:
1. $AB (1)$
2. $CD (2)$
3. $BD (3)$
4. $BC (4)$
5. $AC (5)$

---

### **Step 2: Initialize the MST**

- Start with no edges.
- Use the **Union-Find** structure to check for cycles.

![alt text](images/mst/Kruskal_1.png)

---

### **Step 3: Add edges to MST**

#### Edge 1: $AB (1)$
- Adding $AB$.
- **No cycle** is formed.
- MST = $[AB]$.

*Plot: Edge $AB$ is added.*
![alt text](images/mst/kruskal_2.png)

---

#### Edge 2: $CD (2)$
- Adding $CD$.
- **No cycle** is formed.
- MST = $[AB, CD]$.

*Plot: Edge $CD$ is added.*
![alt text](images/mst/kruskal_3.png)

---

#### Edge 3: $BD (3)$
- Adding $BD$.
- **No cycle** is formed.
- MST = $[AB, CD, BD]$.

*Plot: Edge $BD$ is added.*
![alt text](images/mst/kruskal_4.png)

---

#### Edge 4: $BC (4)$
- Adding $BC$ would form a cycle ($B \to C \to D \to B$ ).
- Skip $BC$.

*Plot: Edge $BC$ is skipped.*
![alt text](images/mst/kruskal_5.png)

---

#### Edge 5: $AC (5)$
- Adding $AC$ would form a cycle ($A \to B \to C \to A$ ).
- Skip $AC$.

*Plot: Edge $AC$ is skipped.*
![alt text](images/mst/kruskal_6.png)

---

### Final MST: $[AB (1), CD (2), BD (3)]$
Total weight = $1 + 2 + 3 = 6$.

![alt text](images/mst/kruskal_7.png)


---

## **Step-by-Step MST Clustering**

### **Step 1: Compute Pairwise Distances**

We have 5 points:
$$
P_1 = (1, 1), \, P_2 = (2, 1), \, P_3 = (4, 3), \, P_4 = (5, 4), \, P_5 = (6, 4)
$$

#### The Euclidean Distance Formula:
$$
w_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

#### **Manual Computation of Pairwise Distances**:
1. $w_{12} = \sqrt{(1-2)^2 + (1-1)^2} = \sqrt{1} = 1.0$
2. $w_{13} = \sqrt{(1-4)^2 + (1-3)^2} = \sqrt{9 + 4} = \sqrt{13} \approx 3.6$
3. $w_{14} = \sqrt{(1-5)^2 + (1-4)^2} = \sqrt{16 + 9} = \sqrt{25} = 5.0$
4. $w_{15} = \sqrt{(1-6)^2 + (1-4)^2} = \sqrt{25 + 9} = \sqrt{34} \approx 5.8$
5. $w_{23} = \sqrt{(2-4)^2 + (1-3)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.8$
6. $w_{24} = \sqrt{(2-5)^2 + (1-4)^2} = \sqrt{9 + 9} = \sqrt{18} \approx 4.2$
7. $w_{25} = \sqrt{(2-6)^2 + (1-4)^2} = \sqrt{16 + 9} = \sqrt{25} = 5.0$
8. $w_{34} = \sqrt{(4-5)^2 + (3-4)^2} = \sqrt{1 + 1} = \sqrt{2} \approx 1.4$
9. $w_{35} = \sqrt{(4-6)^2 + (3-4)^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.2$
10. $w_{45} = \sqrt{(5-6)^2 + (4-4)^2} = \sqrt{1} = 1.0$

#### **Python Code for Pairwise Distances**:
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Points
points = np.array([
    [1, 1],  # P1
    [2, 1],  # P2
    [4, 3],  # P3
    [5, 4],  # P4
    [6, 4]   # P5
])

# Compute pairwise distances
distance_matrix = squareform(pdist(points, metric='euclidean'))
print(distance_matrix)
```
![alt text](images/mst/mst_1.png)

---

### **Step 2: Create a Fully Connected Graph**

Each point is connected to all others, with edge weights representing distances.

#### **Python Code to Create the Graph**:
```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a fully connected graph
G = nx.Graph()
labels = ['P1', 'P2', 'P3', 'P4', 'P5']
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        G.add_edge(i, j, weight=distance_matrix[i, j])

# Visualize the fully connected graph
pos = {i: points[i] for i in range(len(points))}
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, labels={i: labels[i] for i in range(len(points))}, node_color='lightblue', node_size=700)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{w:.1f}" for (i, j), w in edge_labels.items()})
plt.title("Step 1: Fully Connected Graph with Weights")
plt.show()
```
![alt text](images/mst/mst_2.png)

---

### **Step 3: Compute the Minimum Spanning Tree (MST)**

We apply **Kruskal's algorithm** to compute the MST. The edges are added in increasing order of weight, ensuring no cycles are formed.

#### **MST Edges**:
- $w_{12} = 1.0$ : Add $P_1 - P_2$.
- $w_{45} = 1.0$ : Add $P_4 - P_5$.
- $w_{34} = 1.4$ : Add $P_3 - P_4$.
- $w_{23} = 2.8$ : Add $P_2 - P_3$.

#### **Python Code to Compute MST**:
```python
# Compute the Minimum Spanning Tree (MST)
mst = nx.minimum_spanning_tree(G, weight='weight')

# Visualize the MST
plt.figure(figsize=(10, 6))
nx.draw(mst, pos, with_labels=True, labels={i: labels[i] for i in range(len(points))}, node_color='lightgreen', node_size=700)
mst_edge_labels = nx.get_edge_attributes(mst, 'weight')
nx.draw_networkx_edge_labels(mst, pos, edge_labels={(i, j): f"{w:.1f}" for (i, j), w in mst_edge_labels.items()})
plt.title("Step 2: Minimum Spanning Tree (MST)")
plt.show()
```

![alt text](images/mst/mst_3.png)

---

### **Step 4: Form Clusters by Removing the Longest Edge**

- Identify the longest edge in the MST: $w_{23} = 2.8$ ($P_2 - P_3$ ).
- Remove this edge to split the MST into two connected components (clusters).

#### **Clusters**:
1. **Cluster 1**: $P_1, P_2$
2. **Cluster 2**: $P_3, P_4, P_5$

#### **Python Code to Form Clusters**:
```python
# Find the longest edge in the MST
longest_edge = max(mst.edges(data=True), key=lambda x: x[2]['weight'])

# Remove the longest edge to form clusters
mst.remove_edge(longest_edge[0], longest_edge[1])
clusters = list(nx.connected_components(mst))

# Visualize the clusters
plt.figure(figsize=(10, 6))
colors = ['lightblue', 'lightcoral']
for i, cluster in enumerate(clusters):
    nx.draw(nx.subgraph(mst, cluster), pos, with_labels=True, node_color=colors[i], node_size=700, labels={i: labels[i] for i in cluster})
plt.title("Step 3: Clusters After Removing Longest Edge")
plt.show()
```

---

### **Results**

1. **Step 1: Fully Connected Graph**:
   All points are connected with edges weighted by pairwise distances.

2. **Step 2: MST**:
   A reduced tree connecting all points with minimal weight.

3. **Step 3: Clusters**:
   Two clusters are formed by removing the longest edge in the MST.


Got it! Let’s approach the **Network Simplification and Visualization** use case with a **numerical step-by-step example** that includes formulas, calculations, and detailed explanations.

---

### Simplifying a Correlation Network**

#### **Scenario**
We have 4 variables with the following pairwise correlations:

|        | Var1 | Var2 | Var3 | Var4 |
|--------|------|------|------|------|
| **Var1** | 1.00 | 0.80 | 0.60 | 0.40 |
| **Var2** | 0.80 | 1.00 | 0.70 | 0.50 |
| **Var3** | 0.60 | 0.70 | 1.00 | 0.30 |
| **Var4** | 0.40 | 0.50 | 0.30 | 1.00 |

We want to simplify this network using an MST to focus on the strongest relationships.

---

### **Step 1: Convert Correlations to Distances**

To compute an MST, we need distances instead of correlations. The distance between two variables is defined as:
$$
w_{ij} = 1 - |r_{ij}|
$$
where:
- $r_{ij}$ : correlation between $\text{Var}_i$ and $\text{Var}_j$ ,
- $w_{ij}$ : distance between $\text{Var}_i$ and $\text{Var}_j$.

#### **Manual Calculations**:
1. $w_{12} = 1 - |0.80| = 0.20$
2. $w_{13} = 1 - |0.60| = 0.40$
3. $w_{14} = 1 - |0.40| = 0.60$
4. $w_{23} = 1 - |0.70| = 0.30$
5. $w_{24} = 1 - |0.50| = 0.50$
6. $w_{34} = 1 - |0.30| = 0.70$

#### **Distance Matrix**:
$$
W = \begin{bmatrix}
0 & 0.20 & 0.40 & 0.60 \\
0.20 & 0 & 0.30 & 0.50 \\
0.40 & 0.30 & 0 & 0.70 \\
0.60 & 0.50 & 0.70 & 0
\end{bmatrix}
$$

---

### **Step 2: Create a Fully Connected Graph**

The graph has:
- Nodes: $\{\text{Var1, Var2, Var3, Var4}\}$
- Edges: Weighted by the distances $w_{ij}$.

Each pair of nodes is connected, forming a complete graph.

---

### **Step 3: Apply Kruskal’s Algorithm to Find the MST**

#### **Kruskal’s Algorithm**:
1. Sort all edges by weight:
   - $w_{12} = 0.20, w_{23} = 0.30, w_{13} = 0.40, w_{24} = 0.50, w_{14} = 0.60, w_{34} = 0.70$.
2. Start with an empty graph and add edges in increasing order of weight:
   - Add $\text{Var1} - \text{Var2}$ ($w_{12} = 0.20$ ).
   - Add $\text{Var2} - \text{Var3}$ ($w_{23} = 0.30$ ).
   - Add $\text{Var1} - \text{Var3}$ ($w_{13} = 0.40$ ).

#### **MST Edges**:
$$
\{\text{Var1-Var2}, \text{Var2-Var3}, \text{Var1-Var3}\}
$$

---

### **Step 4: Visualize and Analyze**

#### **Final MST**:
The MST simplifies the network, keeping the strongest relationships:
- $\text{Var1}$ is central, connected to both $\text{Var2}$ and $\text{Var3}$.
- $\text{Var2}$ connects $\text{Var3}$ and indirectly to $\text{Var4}$.

---

### **Python Code for Visualization**
```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define nodes and weights
nodes = ["Var1", "Var2", "Var3", "Var4"]
distances = np.array([
    [0, 0.20, 0.40, 0.60],
    [0.20, 0, 0.30, 0.50],
    [0.40, 0.30, 0, 0.70],
    [0.60, 0.50, 0.70, 0]
])

# Create a fully connected graph
G = nx.Graph()
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        G.add_edge(nodes[i], nodes[j], weight=distances[i, j])

# Plot the fully connected graph
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)})
plt.title("Fully Connected Graph")
plt.show()

# Compute the MST
mst = nx.minimum_spanning_tree(G)

# Plot the MST
plt.figure(figsize=(10, 6))
nx.draw(mst, pos, with_labels=True, node_color="lightgreen", node_size=700)
mst_edge_labels = nx.get_edge_attributes(mst, "weight")
nx.draw_networkx_edge_labels(mst, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in mst.edges(data=True)})
plt.title("Minimum Spanning Tree")
plt.show()
```

### Feature Selection for Machine Learning**

#### **Scenario**
In datasets with many features (variables), not all features contribute equally to predicting the target variable. MST analysis can help identify the **most relevant features** by evaluating their pairwise relationships, reducing redundancy, and highlighting the most critical features for model building.

---

### **How MST Is Used for Feature Selection**
1. **Create a Weighted Graph**:
   - Each feature is a node.
   - Edge weights are based on the similarity (or dissimilarity) between features, such as:
     - Correlation (for numeric data).
     - Mutual Information (for mixed data types).
     - Distance metrics (e.g., Euclidean, cosine).
   - For example:
     $$
     \text{Weight (dissimilarity)} = 1 - |\text{Correlation}|
     $$

2. **Compute the MST**:
   - The MST eliminates redundant connections while preserving the strongest relationships.
   - Features highly connected in the MST are considered more relevant.

3. **Select Features**:
   - Features central to the MST or those with strong connections to others are prioritized.
   - Remove features that are highly redundant or weakly connected.

---

### **Numerical Example**

#### **Dataset**:
We have 5 features ($F_1, F_2, F_3, F_4, F_5$ ) with the following pairwise correlations:

|        | $F_1$ | $F_2$ | $F_3$ | $F_4$ | $F_5$ |
|--------|-----------|-----------|-----------|-----------|-----------|
| $F_1$ | 1.00      | 0.85      | 0.50      | 0.10      | 0.30      |
| $F_2$ | 0.85      | 1.00      | 0.60      | 0.20      | 0.40      |
| $F_3$ | 0.50      | 0.60      | 1.00      | 0.70      | 0.80      |
| $F_4$ | 0.10      | 0.20      | 0.70      | 1.00      | 0.90      |
| $F_5$ | 0.30      | 0.40      | 0.80      | 0.90      | 1.00      |


![alt text](images/mst/mst_sm_1.png)

---

### **Step 1: Convert Correlations to Dissimilarities**
$$
w_{ij} = 1 - |r_{ij}|
$$

#### **Manual Calculations**:
1. $w_{12} = 1 - |0.85| = 0.15$
2. $w_{13} = 1 - |0.50| = 0.50$
3. $w_{14} = 1 - |0.10| = 0.90$
4. $w_{15} = 1 - |0.30| = 0.70$
5. $w_{23} = 1 - |0.60| = 0.40$
6. $w_{24} = 1 - |0.20| = 0.80$
7. $w_{25} = 1 - |0.40| = 0.60$
8. $w_{34} = 1 - |0.70| = 0.30$
9. $w_{35} = 1 - |0.80| = 0.20$
10. $w_{45} = 1 - |0.90| = 0.10$

#### **Dissimilarity Matrix**:
$$
W = \begin{bmatrix}
0 & 0.15 & 0.50 & 0.90 & 0.70 \\
0.15 & 0 & 0.40 & 0.80 & 0.60 \\
0.50 & 0.40 & 0 & 0.30 & 0.20 \\
0.90 & 0.80 & 0.30 & 0 & 0.10 \\
0.70 & 0.60 & 0.20 & 0.10 & 0
\end{bmatrix}
$$

---

### **Step 2: Compute the MST**

Using Kruskal’s Algorithm:
1. Sort edges by weight:
   - $w_{45} = 0.10, w_{35} = 0.20, w_{34} = 0.30, w_{12} = 0.15, w_{23} = 0.40, \dots$
2. Add edges in order of increasing weight without forming cycles.

![alt text](images/mst/mst_sm_2.png)
#### **MST Edges**:
$$
\{F_4-F_5, F_3-F_5, F_1-F_2, F_3-F_4\}
$$

---

### **Step 3: Select Features**

all have storng connections

![alt text](images/mst/mst_sm_3.png)

---

### **Python Code for MST Feature Selection**

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define nodes and dissimilarities
nodes = ["F1", "F2", "F3", "F4", "F5"]
dissimilarities = np.array([
    [0, 0.15, 0.50, 0.90, 0.70],
    [0.15, 0, 0.40, 0.80, 0.60],
    [0.50, 0.40, 0, 0.30, 0.20],
    [0.90, 0.80, 0.30, 0, 0.10],
    [0.70, 0.60, 0.20, 0.10, 0]
])

# Create graph
G = nx.Graph()
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        G.add_edge(nodes[i], nodes[j], weight=dissimilarities[i, j])

# Compute MST
mst = nx.minimum_spanning_tree(G)

# Plot MST
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 6))
nx.draw(mst, pos, with_labels=True, node_color="lightgreen", node_size=700)
mst_edge_labels = nx.get_edge_attributes(mst, "weight")
nx.draw_networkx_edge_labels(mst, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in mst.edges(data=True)})
plt.title("MST for Feature Selection")
plt.show()
```

---

### **Fourth Use Case: Anomaly Detection in Networks**

#### **Scenario**
An MST can be used to identify anomalies or outliers in a network by focusing on edges with unusually large weights or nodes with weak connections to the rest of the graph. This is particularly useful in areas like:

- **Network Security**: Detecting unusual behavior in network traffic.
- **Sensor Networks**: Identifying faulty or misbehaving sensors.
- **Social Networks**: Highlighting outlier users or accounts.

---

### **How MST Is Used for Anomaly Detection**
1. **Create a Weighted Graph**:
   - Nodes represent entities (e.g., devices, users, or sensors).
   - Edge weights reflect dissimilarities, such as distances, inverse correlations, or other measures.

2. **Compute the MST**:
   - Use the MST to simplify the network while maintaining the strongest relationships.

3. **Identify Anomalies**:
   - Look for nodes connected by **long edges** (high weights), as they indicate weak relationships.
   - Nodes with a single connection (low degree) or extreme edge weights are potential outliers.

---

### **Numerical Example: Detecting Faulty Sensors**

#### **Data**:
We have five sensors measuring temperature across locations. The pairwise dissimilarity matrix is:
$$
W = \begin{bmatrix}
0 & 0.10 & 0.20 & 0.80 & 0.90 \\
0.10 & 0 & 0.15 & 0.75 & 0.85 \\
0.20 & 0.15 & 0 & 0.70 & 0.80 \\
0.80 & 0.75 & 0.70 & 0 & 0.10 \\
0.90 & 0.85 & 0.80 & 0.10 & 0
\end{bmatrix}
$$

#### Steps:

1. **Fully Connected Graph**:
   - Represent the network with edges weighted by $W$.

2. **Compute the MST**:
   - Use Kruskal’s algorithm to find the MST.

3. **Detect Anomalies**:
   - Highlight edges with large weights (e.g., $0.80, 0.90$ ).
   - Identify nodes with only one connection as potential outliers.

---

### **Python Code for Anomaly Detection**

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the dissimilarity matrix
anomaly_dissimilarities = np.array([
    [0, 0.10, 0.20, 0.80, 0.90],
    [0.10, 0, 0.15, 0.75, 0.85],
    [0.20, 0.15, 0, 0.70, 0.80],
    [0.80, 0.75, 0.70, 0, 0.10],
    [0.90, 0.85, 0.80, 0.10, 0]
])

# Create the graph
anomaly_G = nx.Graph()
anomaly_nodes = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5"]
for i in range(len(anomaly_nodes)):
    for j in range(i + 1, len(anomaly_nodes)):
        anomaly_G.add_edge(anomaly_nodes[i], anomaly_nodes[j], weight=anomaly_dissimilarities[i, j])

# Plot the fully connected graph
anomaly_pos = nx.spring_layout(anomaly_G, seed=42)
plt.figure(figsize=(10, 6))
nx.draw(anomaly_G, anomaly_pos, with_labels=True, node_color="lightblue", node_size=700)
edge_labels = nx.get_edge_attributes(anomaly_G, "weight")
nx.draw_networkx_edge_labels(anomaly_G, anomaly_pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in anomaly_G.edges(data=True)})
plt.title("Fully Connected Graph for Sensors")
plt.show()

# Compute the MST
anomaly_mst = nx.minimum_spanning_tree(anomaly_G)

# Plot the MST
plt.figure(figsize=(10, 6))
nx.draw(anomaly_mst, anomaly_pos, with_labels=True, node_color="lightgreen", node_size=700)
mst_edge_labels = nx.get_edge_attributes(anomaly_mst, "weight")
nx.draw_networkx_edge_labels(anomaly_mst, anomaly_pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in anomaly_mst.edges(data=True)})
plt.title("Minimum Spanning Tree (MST) for Sensors")
plt.show()

# Analyze the MST for anomalies
anomaly_edges = sorted(anomaly_mst.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
print("Edges in MST (sorted by weight):", anomaly_edges)

# Detect anomalies
anomalous_edges = [edge for edge in anomaly_edges if edge[2]["weight"] > 0.7]
print("Anomalous Edges (Weight > 0.7):", anomalous_edges)
```

---

### **Expected Results**
1. **Anomalous Edges**:
   - Edges with weights $> 0.7$ , such as $\text{Sensor1} - \text{Sensor4}$ and $\text{Sensor1} - \text{Sensor5}$ , represent weak or unusual relationships.

2. **Outlier Sensors**:
   - Nodes connected only by high-weight edges (e.g., $\text{Sensor4}$ and $\text{Sensor5}$ ) are flagged as potential anomalies.

You're correct—I need to provide a full **numerical example** that walks through the calculations for **dissimilarities, MST, and anomaly detection** step by step, as per your requirements. Let’s start over and build the example explicitly.

---

### **Numerical Example: Detecting Faulty Sensors**

#### **Step 1: Pairwise Measurements**
We have five sensors ($S_1, S_2, S_3, S_4, S_5$ ) with pairwise similarity scores (correlations):

|         | $S_1$ | $S_2$ | $S_3$ | $S_4$ | $S_5$ |
|---------|-----------|-----------|-----------|-----------|-----------|
| **$S_1$** | 1.00      | 0.90      | 0.80      | 0.20      | 0.10      |
| **$S_2$** | 0.90      | 1.00      | 0.85      | 0.25      | 0.15      |
| **$S_3$** | 0.80      | 0.85      | 1.00      | 0.70      | 0.60      |
| **$S_4$** | 0.20      | 0.25      | 0.70      | 1.00      | 0.90      |
| **$S_5$** | 0.10      | 0.15      | 0.60      | 0.90      | 1.00      |

#### **Step 2: Convert Similarities to Dissimilarities**
We convert the correlations to dissimilarities using:
$$
w_{ij} = 1 - |r_{ij}|
$$
where:
- $r_{ij}$ is the similarity between sensors $S_i$ and $S_j$ ,
- $w_{ij}$ is the dissimilarity between them.

#### **Calculations**:
1. $w_{12} = 1 - |0.90| = 0.10$
2. $w_{13} = 1 - |0.80| = 0.20$
3. $w_{14} = 1 - |0.20| = 0.80$
4. $w_{15} = 1 - |0.10| = 0.90$
5. $w_{23} = 1 - |0.85| = 0.15$
6. $w_{24} = 1 - |0.25| = 0.75$
7. $w_{25} = 1 - |0.15| = 0.85$
8. $w_{34} = 1 - |0.70| = 0.30$
9. $w_{35} = 1 - |0.60| = 0.40$
10. $w_{45} = 1 - |0.90| = 0.10$

#### **Dissimilarity Matrix**:
$$
W = \begin{bmatrix}
0 & 0.10 & 0.20 & 0.80 & 0.90 \\
0.10 & 0 & 0.15 & 0.75 & 0.85 \\
0.20 & 0.15 & 0 & 0.30 & 0.40 \\
0.80 & 0.75 & 0.30 & 0 & 0.10 \\
0.90 & 0.85 & 0.40 & 0.10 & 0
\end{bmatrix}
$$

---

### **Step 3: Create the Fully Connected Graph**
1. **Nodes**: $S_1, S_2, S_3, S_4, S_5$
2. **Edges**: Weighted by $W$.

---

### **Step 4: Compute the MST**
We use **Kruskal’s Algorithm**:
1. Sort all edges by weight:
   - $w_{12} = 0.10, w_{45} = 0.10, w_{23} = 0.15, w_{13} = 0.20, w_{34} = 0.30, \dots$
2. Add edges in increasing order, avoiding cycles.

#### **Selected Edges**:
1. $S_1 - S_2$ ($w_{12} = 0.10$ )
2. $S_4 - S_5$ ($w_{45} = 0.10$ )
3. $S_2 - S_3$ ($w_{23} = 0.15$ )
4. $S_3 - S_4$ ($w_{34} = 0.30$ )

#### **MST**:
- **Edges**: $\{S_1-S_2, S_4-S_5, S_2-S_3, S_3-S_4\}$
- **Total Weight**: $0.10 + 0.10 + 0.15 + 0.30 = 0.65$

---

### **Step 5: Detect Anomalies**
1. **High-Weight Edges**:
   - Edges $w_{14} = 0.80$ , $w_{15} = 0.90$ , and $w_{25} = 0.85$ were excluded from the MST due to their large weights, indicating potential anomalies.
2. **Outlier Nodes**:
   - $S_1$ : Only connects to $S_2$ with a small weight.
   - $S_5$ : Only connects to $S_4$.

---

### **Python Code for MST and Anomaly Detection**
```python
# Define the dissimilarity matrix
dissimilarities = np.array([
    [0, 0.10, 0.20, 0.80, 0.90],
    [0.10, 0, 0.15, 0.75, 0.85],
    [0.20, 0.15, 0, 0.30, 0.40],
    [0.80, 0.75, 0.30, 0, 0.10],
    [0.90, 0.85, 0.40, 0.10, 0]
])

# Create the graph
G = nx.Graph()
nodes = ["S1", "S2", "S3", "S4", "S5"]
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        G.add_edge(nodes[i], nodes[j], weight=dissimilarities[i, j])

# Compute MST
mst = nx.minimum_spanning_tree(G)

# Analyze MST for anomalies
edges = sorted(mst.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
anomalous_edges = [edge for edge in edges if edge[2]["weight"] > 0.7]
print("Anomalous Edges (Weight > 0.7):", anomalous_edges)

# Plot Fully Connected Graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)})
plt.title("Fully Connected Graph")
plt.show()

# Plot MST
plt.figure(figsize=(10, 6))
nx.draw(mst, pos, with_labels=True, node_color="lightgreen", node_size=700)
mst_edge_labels = nx.get_edge_attributes(mst, "weight")
nx.draw_networkx_edge_labels(mst, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in mst.edges(data=True)})
plt.title("Minimum Spanning Tree (MST)")
plt.show()
```
![alt text](images/mst/mst_an_1.png)
### **Explanation of the Control Chart**

#### **What the Plot Represents:**
1. **Lines for Each Sensor**:
   - Each sensor ($S_1, S_2, S_3, S_4, S_5$ ) is represented by a distinct line on the chart.
   - The y-axis shows the sensor's measured values (e.g., temperature, pressure, or another process variable).
   - The x-axis represents the sample index (time or sequential observations).

2. **Red Dashed Line**:
   - This line represents the **control limit** or the **mean value** of the process.
   - The sensors are expected to oscillate around this line if they are functioning correctly.

---

#### **How to Interpret the Plot**:
1. **Normal Behavior**:
   - If a sensor's values stay close to the red dashed line, it indicates normal behavior and no anomaly.
   - Minor fluctuations are expected and do not indicate an issue.

2. **Anomalous Behavior**:
   - If a sensor shows extreme deviations (consistently above or below the red line), it may indicate an anomaly or fault in the process or sensor.
   - For example:
     - A sudden spike or dip in one sensor might suggest a temporary glitch.
     - Sustained deviation from the mean could point to calibration errors, faulty hardware, or a systemic issue.

---

#### **Use in Context**:
- This chart provides a quick visual representation of sensor behavior over time.
- Combined with the MST analysis:
  - Sensors with weak or anomalous connections in the MST (e.g., $S_4$ , $S_5$ ) can be flagged.
  - If their behavior in the control chart is also abnormal, it further confirms their status as outliers.

### **Fifth Use Case: Network Optimization**

#### **Scenario**
In many applications, resources such as bandwidth, power, or transportation links must be optimized. Using an MST ensures that all nodes in a network are connected with the minimum possible cost while avoiding redundant or inefficient connections.

#### **Applications**:
1. **Power Grid Design**:
   - Minimize the cost of building electrical lines while ensuring all locations are connected.
2. **Telecommunications**:
   - Optimize the layout of network nodes (e.g., cellular towers) to reduce infrastructure costs.
3. **Transportation and Logistics**:
   - Design efficient road, rail, or shipping networks to connect cities while minimizing travel distances or costs.

---

### **Numerical Example: Optimizing a Transportation Network**

#### **Problem**:
We need to connect 5 cities ($C_1, C_2, C_3, C_4, C_5$ ) with roads. The pairwise distances (costs) between cities are:

|         | $C_1$ | $C_2$ | $C_3$ | $C_4$ | $C_5$ |
|---------|-----------|-----------|-----------|-----------|-----------|
| **$C_1$** | 0         | 5         | 10        | 8         | 7         |
| **$C_2$** | 5         | 0         | 6         | 3         | 4         |
| **$C_3$** | 10        | 6         | 0         | 9         | 2         |
| **$C_4$** | 8         | 3         | 9         | 0         | 5         |
| **$C_5$** | 7         | 4         | 2         | 5         | 0         |

Our goal is to connect all cities at the minimum cost.

---

### **Step 1: Fully Connected Graph**
1. **Nodes**: Represent the cities ($C_1, C_2, C_3, C_4, C_5$ ).
2. **Edges**: Represent road costs ($w_{ij}$ ) between the cities.

#### **Dissimilarity Matrix**:
$$
W = \begin{bmatrix}
0 & 5 & 10 & 8 & 7 \\
5 & 0 & 6 & 3 & 4 \\
10 & 6 & 0 & 9 & 2 \\
8 & 3 & 9 & 0 & 5 \\
7 & 4 & 2 & 5 & 0
\end{bmatrix}
$$

---

### **Step 2: Compute the MST**
We use **Kruskal’s Algorithm** to compute the MST:
1. Sort all edges by weight:
   - $w_{35} = 2, w_{24} = 3, w_{25} = 4, w_{12} = 5, w_{45} = 5, \dots$
2. Add edges in increasing order while avoiding cycles.

#### **Selected Edges**:
1. $C_3 - C_5$ ($w_{35} = 2$ ).
2. $C_2 - C_4$ ($w_{24} = 3$ ).
3. $C_2 - C_5$ ($w_{25} = 4$ ).
4. $C_1 - C_2$ ($w_{12} = 5$ ).

#### **MST**:
- **Edges**: $\{C_3-C_5, C_2-C_4, C_2-C_5, C_1-C_2\}$
- **Total Cost**: $2 + 3 + 4 + 5 = 14$

---

### **Step 3: Visualize the MST**
The MST shows the most efficient way to connect the cities with the minimum total cost.

---

### **Python Code for Optimization**

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the cost matrix
cost_matrix = np.array([
    [0, 5, 10, 8, 7],
    [5, 0, 6, 3, 4],
    [10, 6, 0, 9, 2],
    [8, 3, 9, 0, 5],
    [7, 4, 2, 5, 0]
])

# Create the graph
G = nx.Graph()
cities = ["C1", "C2", "C3", "C4", "C5"]
for i in range(len(cities)):
    for j in range(i + 1, len(cities)):
        G.add_edge(cities[i], cities[j], weight=cost_matrix[i, j])

# Compute MST
mst = nx.minimum_spanning_tree(G)

# Plot the Fully Connected Graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)})
plt.title("Fully Connected Graph for Cities")
plt.show()

# Plot the MST
plt.figure(figsize=(10, 6))
nx.draw(mst, pos, with_labels=True, node_color="lightgreen", node_size=700)
mst_edge_labels = nx.get_edge_attributes(mst, "weight")
nx.draw_networkx_edge_labels(mst, pos, edge_labels={(u, v): f"{d['weight']}" for u, v, d in mst.edges(data=True)})
plt.title("Minimum Spanning Tree (MST) for Cities")
plt.show()
```

---

### **Results**

1. **Optimal Connections**:
   - $C_3 - C_5$ : Cost = 2
   - $C_2 - C_4$ : Cost = 3
   - $C_2 - C_5$ : Cost = 4
   - $C_1 - C_2$ : Cost = 5

2. **Total Cost**:
   - $2 + 3 + 4 + 5 = 14$.

