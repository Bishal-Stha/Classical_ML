# Titanic - Machine Learning from Disaster Model

## Problem Statement

Create a Machine learning model to evaluate the **Titanic - Machine Learning from Disaster** From Kaggle.

### Dataset Source

The dataset is **Titanic - Machine Learning from Disaster** dataset and it is derived from [Titanic Dataset](https://www.kaggle.com/competitions/titanic)

### Model Selection

Logistic Regression is selected because:

- The dataset is small.
- The problem is binary classification.
- It is interpretable.
- It provides probabilistic outputs.
- It serves as a strong baseline model.
Other models like Random Forest, KNN may be tested later for performance comparison.

### Metrics I will use

- Accuracy
- F1 Score
- Confusion Matrix

### Critical Preprocessing

1. Numerical Columns

- PassengerId
- Age
- SibSp
- Parch
- Fare

2. Categorical Columns

- Name
- Sex
- Ticket
- Cabin
- Emarked

3. Clearly useless columns

- PassengerId
- Name
- Ticket
- Cabin

4. Columns that need encoding

- Sex
- Embarked

5. Missing Value Handling

- Age (Fill with median)
- Cabin (Drop)
- Embarked (Fill with mode)

## Project Structure

project/
├── data/
├── models/
│   └── logistic_model.pkl
├── src/
│   ├── preprocess.py
│   └── train.py