# 📊 Exploratory Data Analysis (EDA) — Command Checklist

Assume:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 🟦 1. Context

dataset size:

```python
df.shape
```

number of rows:

```python
df.shape[0]
```

number of columns:

```python
df.shape[1]
```

column names:

```python
df.columns
```

first few rows:

```python
df.head()
```

last few rows:

```python
df.tail()
```

random sample:

```python
df.sample(5)
```

data types:

```python
df.dtypes
```

dataset info (memory + nulls):

```python
df.info()
```

---

## 🟦 2. Quality Check

missing values per column:

```python
df.isnull().sum()
```

percentage of missing values:

```python
df.isnull().mean() * 100
```

total missing values:

```python
df.isnull().sum().sum()
```

duplicate rows count:

```python
df.duplicated().sum()
```

view duplicate rows:

```python
df[df.duplicated()]
```

basic statistical summary (numerical):

```python
df.describe()
```

statistical summary (categorical):

```python
df.describe(include='object')
```

check unique values per column:

```python
df.nunique()
```

value counts for a column:

```python
df['column_name'].value_counts()
```

outlier check (IQR idea):

```python
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
```

---

## 🟦 3. Analysis of Variables

numerical columns:

```python
df.select_dtypes(include=np.number).columns
```

categorical columns:

```python
df.select_dtypes(include='object').columns
```

distribution (histogram):

```python
df['column'].hist()
```

distribution with KDE:

```python
sns.histplot(df['column'], kde=True)
```

boxplot (outliers):

```python
sns.boxplot(x=df['column'])
```

categorical frequency plot:

```python
sns.countplot(x='column', data=df)
```

target variable distribution:

```python
df['target'].value_counts()
```

target imbalance ratio:

```python
df['target'].value_counts(normalize=True)
```

---

## 🟦 4. Relationships

correlation matrix:

```python
df.corr()
```

correlation heatmap:

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

feature vs target (numerical → target):

```python
sns.boxplot(x='target', y='feature', data=df)
```

feature vs feature (scatter):

```python
sns.scatterplot(x='feature1', y='feature2', data=df)
```

pairwise relationships:

```python
sns.pairplot(df)
```

grouped statistics:

```python
df.groupby('target').mean()
```

leakage check (suspicious correlation):

```python
df.corr()['target'].sort_values(ascending=False)
```

---

## 🟦 5. Time Effects & Modeling Readiness

convert to datetime:

```python
df['date'] = pd.to_datetime(df['date'])
```

set time index:

```python
df.set_index('date', inplace=True)
```

time trend plot:

```python
df['feature'].plot()
```

rolling mean (trend):

```python
df['feature'].rolling(window=7).mean()
```

scaling (standardization):

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numerical_cols])
```

encoding categorical variables:

```python
pd.get_dummies(df, drop_first=True)
```

feature selection (correlation-based):

```python
df.corr()['target'].abs().sort_values(ascending=False)
```

train-ready feature/target split:

```python
X = df.drop('target', axis=1)
y = df['target']
```

---