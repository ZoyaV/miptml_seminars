

# Classification Seminar

Welcome to the seminar on classification tasks. This session will guide you through the foundational concepts of classification, its metrics, and how to handle various feature types in Python.

## Table of Contents

1. [Lesson Plan](#lesson-plan) 
2. [Theory of Classification Tasks](#theory-of-classification-tasks)
3. [Metrics for Binary Classification](#metrics-for-binary-classification)
4. [Metrics for Multiclass Classification](#metrics-for-multiclass-classification)
5. [Working with Different Feature Types (Python Code)](#working-with-different-feature-types)

---

## Lesson Plan

During this seminar, we will cover the following topics:

- [Linear Regression Assignment](https://github.com/ZoyaV/miptml_seminars/blob/main/sem_3_classification/task_1.ipynb)
- [Gradient Descent for Classification](https://clck.ru/35kN9X)
- [Multiclass Classification](https://github.com/ZoyaV/miptml_seminars/blob/main/sem_3_classification/Linear_Classifiers_and_SVM_Seminar_Retry.ipynb)
- Model Validation

---

## Theory of Classification Tasks

Classification is a supervised learning approach where the goal is to predict the categorical class labels of new instances, based on past observations. These class labels can be binary (two classes) or multiclass (more than two classes).

---

## Metrics for Binary Classification

Binary classification deals with situations where the outcome can only be one of two classes. The common metrics used for evaluating binary classification models are:

- **Accuracy**: Proportion of correctly predicted classifications in the total predictions.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall (Sensitivity)**: Proportion of actual positives that were identified correctly.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC Curve**: A graphical representation of the true positive rate against the false positive rate.
- **AUC (Area Under the Curve)**: Represents the model's ability to distinguish between positive and negative classes.

---

## Metrics for Multiclass Classification

Multiclass classification deals with situations where the outcome can be one of more than two classes. The metrics for multiclass classification expand on the binary metrics:

- **Micro-average**: Calculate metrics globally by counting the total true positives, false negatives, and false positives.
- **Macro-average**: Calculate metrics for each label and find their unweighted mean.
- **Weighted-average**: Calculate metrics for each label, and find their average weighted by the number of true instances for each label.

---

## Working with Different Feature Types

In real-world datasets, features can come in various types such as numerical, categorical, text, and more. Handling these different feature types is crucial for building effective models.

```python
# Sample Python code for handling different feature types

# For numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_features = scaler.fit_transform(numerical_features)

# For categorical features
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
categorical_features = encoder.fit_transform(categorical_features)

# ... and so on for other feature types
```


