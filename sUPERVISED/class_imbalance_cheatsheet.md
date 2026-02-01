````markdown
# Class Imbalance in Machine Learning — Cheat Sheet

## 1. Definition
> Class imbalance occurs when one class significantly outnumbers other classes in a dataset, leading to biased models and misleading accuracy.

**Example (Binary):**
| Class | Samples |
|-------|--------|
| 0 (majority) | 900 |
| 1 (minority) | 100 |

---

## 2. Why It’s a Problem
- Accuracy is misleading:
  - Example: Predict all as majority → 95% accuracy, but minority is ignored
- Minority class often matters more in real-world tasks (fraud, disease detection)

---

## 3. Metrics to Use
| Metric | Notes |
|--------|------|
| Precision | Fraction of predicted positives that are correct |
| Recall (Sensitivity) | Fraction of actual positives correctly detected |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Measures class separation |
| Confusion Matrix | Full overview |

**Tip:** Use **recall** when missing minority samples is costly.

---

## 4. Solutions Overview
**Three levels:**
1. **Data-level** → Modify dataset (oversampling, undersampling, SMOTE)  
2. **Algorithm-level** → Modify learning (class weights, threshold tuning)  
3. **Metric-level** → Use minority-aware metrics

---

## 5. Data-level Methods

### Undersampling
- Reduce majority class to match minority
- Pros: Faster  
- Cons: Loses information

### Oversampling
- Duplicate minority samples
- Pros: Simple  
- Cons: Risk of overfitting

### SMOTE (Synthetic Minority Oversampling Technique)
- Creates new synthetic minority samples
- Pros: Reduces overfitting compared to naive oversampling
- Code:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
````

> Apply **only on training data**

---

## 6. Algorithm-level Methods

### Class Weighting

* Penalize errors on minority class more
* **Scikit-learn Example:**

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
```

* **Keras Example:**

```python
class_weight = {0:1, 1:10}
model.fit(X_train, y_train, class_weight=class_weight)
```

### Threshold Tuning

* Default: `prob > 0.5`
* Imbalanced: Lower threshold to increase recall, e.g., `prob > 0.3`

---

## 7. Full Practical Workflow (Reusable)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1. Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 2. Handle imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 3. Train model with class weights
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# 4. Evaluate properly
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 8. Decision Table (Quick Reference)

| Situation       | Best Method               |
| --------------- | ------------------------- |
| Small dataset   | Class weights             |
| Large dataset   | Undersampling             |
| Medium dataset  | SMOTE                     |
| Deep learning   | Class weights             |
| Medical / Fraud | Recall + threshold tuning |

---````markdown

# Class Imbalance in Machine Learning — Cheat Sheet

## 1. Definition
>
> Class imbalance occurs when one class significantly outnumbers other classes in a dataset, leading to biased models and misleading accuracy.

**Example (Binary):**

| Class | Samples |
|-------|--------|
| 0 (majority) | 900 |
| 1 (minority) | 100 |

---

## 2. Why It’s a Problem

- Accuracy is misleading:
  * Example: Predict all as majority → 95% accuracy, but minority is ignored
* Minority class often matters more in real-world tasks (fraud, disease detection)

---

## 3. Metrics to Use

| Metric | Notes |
|--------|------|
| Precision | Fraction of predicted positives that are correct |
| Recall (Sensitivity) | Fraction of actual positives correctly detected |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Measures class separation |
| Confusion Matrix | Full overview |

**Tip:** Use **recall** when missing minority samples is costly.

---

## 4. Solutions Overview

**Three levels:**

1. **Data-level** → Modify dataset (oversampling, undersampling, SMOTE)  
2. **Algorithm-level** → Modify learning (class weights, threshold tuning)  
3. **Metric-level** → Use minority-aware metrics

---

## 5. Data-level Methods

### Undersampling

- Reduce majority class to match minority
* Pros: Faster  
* Cons: Loses information

### Oversampling

- Duplicate minority samples
* Pros: Simple  
* Cons: Risk of overfitting

### SMOTE (Synthetic Minority Oversampling Technique)

- Creates new synthetic minority samples
* Pros: Reduces overfitting compared to naive oversampling
* Code:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
````

> Apply **only on training data**

---

## 6. Algorithm-level Methods

### Class Weighting

* Penalize errors on minority class more
* **Scikit-learn Example:**

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
```

* **Keras Example:**

```python
class_weight = {0:1, 1:10}
model.fit(X_train, y_train, class_weight=class_weight)
```

### Threshold Tuning

* Default: `prob > 0.5`
* Imbalanced: Lower threshold to increase recall, e.g., `prob > 0.3`

---

## 7. Full Practical Workflow (Reusable)

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1. Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 2. Handle imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 3. Train model with class weights
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# 4. Evaluate properly
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 8. Decision Table (Quick Reference)

| Situation       | Best Method               |
| --------------- | ------------------------- |
| Small dataset   | Class weights             |
| Large dataset   | Undersampling             |
| Medium dataset  | SMOTE                     |
| Deep learning   | Class weights             |
| Medical / Fraud | Recall + threshold tuning |

---
