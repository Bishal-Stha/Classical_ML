# Customer Segmentation – Unsupervised Learning Model

## Problem Statement

Create an **unsupervised machine learning model** to discover hidden patterns and segment customers into meaningful groups.

Unlike supervised learning, there is **no target variable**.
The goal is to identify natural clusters in the dataset.

---

## Dataset Source

The dataset used is:
Mall Customers dataset
(Replace this with your actual dataset.)

---

## Model Selection

### DBSCAN is selected because:

* The dataset may contain **noise/outliers**
* Clusters may be **non-spherical**
* It does not require specifying number of clusters (unlike K-Means)
* It works well for density-based segmentation

Other models that may be tested:

* K-Means
* Hierarchical Clustering
* Gaussian Mixture Models

---

## Evaluation Metrics (Unsupervised)

Since we don’t have labels, we use:

* **Silhouette Score**
* **Davies-Bouldin Index**
* **Calinski-Harabasz Score**
* Cluster visualization (2D PCA plots)
* Business interpretability

---

## Critical Preprocessing

### 1. Numerical Columns

* Age
* Annual Income
* Spending Score

### 2. Categorical Columns

* Gender

### 3. Feature Engineering

* Scaling (StandardScaler) ← VERY important for DBSCAN
* PCA for dimensionality reduction (for visualization)

### 4. Missing Value Handling

* Fill numeric values with median
* Drop columns with excessive missing values

---

## Model Training Steps

1. Scale the data
2. Perform PCA (optional for visualization)
3. Tune `eps` using k-distance plot
4. Train DBSCAN
5. Analyze cluster labels
6. Identify noise points (-1 label)

---

## Results Interpretation

Example:

* Cluster 0 → High income, high spending (Premium customers)
* Cluster 1 → Low income, low spending (Budget customers)
* Cluster -1 → Outliers

The model helps in:

* Targeted marketing
* Personalized campaigns
* Resource allocation

---

## Project Structure

```
project/
├── data/
├── models/
│   └── dbscan_model.pkl
├── src/
│   ├── preprocess.py
│   ├── cluster.py
│   ├── evaluate.py
│   └── app.py
```

---

## Run App

Install streamlit:

```
pip install streamlit
```

Run:

```
streamlit run src/app.py
```