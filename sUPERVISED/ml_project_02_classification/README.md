# Iris Species Model

## Problem Statement

Create a Machine learning model to evaluate the **Iris Species** From Kaggle.

### Dataset Source

The dataset is **Iris Species** dataset and it is derived from [Iris Species](https://www.kaggle.com/datasets/uciml/iris)

### Model Selection

KNN is selected because:

- The dataset is small.
- The problem is multi-class classification.
- It is interpretable.
- It serves as a strong baseline model.
Other models like Random Forest, SVM may be tested later for performance comparison.

### Metrics I will use

- Accuracy
- F1 Score
- Confusion matrix

### Critical Preprocessing

1. Numerical Columns

- Id
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

2. Categorical Columns

- Species

3. Clearly useless columns

- Id

4. Columns that need encoding

- Species. (use mapping to assign 0,1 and 2 value for versicolor, virginica and setosa).

5. Missing Value Handling

- None. There are no missing or duplicated values.

## Project Structure

project/
├── data/
├── models/
│   └── Iris_model.pkl
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   └── train.py
├── README.md
├── requirments.txt
