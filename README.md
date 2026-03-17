# Predictive Analytics Lab Exam-2

## Overview

This project implements a **binary classification task** as part of the Predictive Analytics Lab Exam.
The goal is to analyze the dataset, build a classification model, visualize the decision boundary, and evaluate model performance.

---

## Dataset

The dataset used is **Lab_Exam_binary_classification_dataset.csv**.

### Features

* **Feature1** – Numerical input variable
* **Feature2** – Numerical input variable

### Target

* **Target** – Binary classification label (Yes / No)

---

## Exploratory Data Analysis (EDA)

The following steps were performed during EDA:

* Dataset inspection using `head()` and `info()`
* Statistical summary using `describe()`
* Handling missing values
* Removal of extreme outliers
* Scatter plot visualization of feature distribution

Example visualization:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x="Feature1", y="Feature2", hue="Target", data=data)
plt.title("Feature Distribution")
plt.show()
```

---

## Data Preprocessing

Steps performed:

1. Removed missing target values
2. Removed extreme outlier in **Feature1**
3. Encoded categorical target variable using **LabelEncoder**
4. Split dataset into training and testing sets (80/20)

Example preprocessing code:

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = data.dropna()
data = data[data["Feature1"] < 10]

le = LabelEncoder()
data["Target"] = le.fit_transform(data["Target"])

X = data[["Feature1", "Feature2"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Classification Model

A **Logistic Regression classifier** was used for the binary classification task.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
```

---

## Decision Boundary Visualization

The decision boundary helps visualize how the classifier separates the two classes.

```python
import numpy as np

x_min, x_max = X.iloc[:,0].min()-1, X.iloc[:,0].max()+1
y_min, y_max = X.iloc[:,1].min()-20, X.iloc[:,1].max()+20

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 1)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X["Feature1"], X["Feature2"], c=y, edgecolor='k')

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("Decision Boundary")

plt.show()
```

---

## Model Evaluation

The model performance was evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report

### Results

**Accuracy:**

```
0.955
```

### Classification Report

```
              precision    recall  f1-score   support

           0       0.99      0.95      0.97       158
           1       0.84      0.98      0.90        42

    accuracy                           0.95       200
   macro avg       0.92      0.96      0.94       200
weighted avg       0.96      0.95      0.96       200
```

The results show strong performance across both classes, with high precision and recall values, indicating that the model successfully distinguishes between the two categories.

---

## Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

## How to Run

1. Clone the repository
2. Install required libraries
3. Run the notebook or Python script

Example:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the model using the Jupyter notebook:

```
Predictive_lab.ipynb
```

---

## Author

Predictive Analytics Lab Exam Submission
