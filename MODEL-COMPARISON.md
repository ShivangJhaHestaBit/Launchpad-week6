# Model Comparison Report

This document provides a detailed comparison of the machine learning models trained in the unified training pipeline. Each model was evaluated using **5-fold cross-validation** and compared across multiple metrics to ensure robust and fair selection of the best-performing model.

---

## Evaluation Metrics Used

All models were evaluated using the following metrics:

* **Accuracy**: Overall correctness of predictions.
* **Precision**: How many predicted positives were actually positive.
* **Recall**: How many actual positives were correctly identified.
* **F1 Score**: Harmonic mean of precision and recall.
* **ROC-AUC**: Ability of the model to distinguish between classes across thresholds.

These metrics together provide a balanced view of model performance, especially for slightly imbalanced datasets like Titanic.

---

## Models Trained

### 1. Logistic Regression

**Overview**
Logistic Regression is a linear classification model that estimates the probability of a binary outcome using a logistic (sigmoid) function.

**Why it was used**

* Simple and interpretable baseline model
* Fast to train and evaluate
* Performs well when the relationship between features and target is approximately linear

**Strengths**

* High interpretability
* Low risk of overfitting
* Works well with scaled numerical features

**Limitations**

* Cannot capture complex non-linear relationships
* Performance depends heavily on feature engineering

**Use case**
Serves as a strong baseline to compare more complex models.

---

### 2. Random Forest Classifier

**Overview**
Random Forest is an ensemble model that builds multiple decision trees and aggregates their predictions to improve generalization.

**Why it was used**

* Handles non-linear relationships effectively
* Robust to outliers and noise
* Provides built-in feature importance

**Strengths**

* Strong performance without heavy tuning
* Handles mixed feature types well
* Reduces overfitting compared to a single decision tree

**Limitations**

* Less interpretable than linear models
* Larger model size and slower inference

**Use case**
Acts as a powerful general-purpose model and often performs best on tabular datasets.

---

### 3. Gradient Boosting Classifier

**Overview**
Gradient Boosting builds models sequentially, where each new model corrects the errors made by the previous ones.

**Why it was used**

* Excellent performance on structured/tabular data
* Captures complex feature interactions
* Strong bias–variance tradeoff

**Strengths**

* High predictive accuracy
* Effective for difficult classification problems
* Handles non-linearities well

**Limitations**

* Sensitive to hyperparameters
* Longer training time
* Higher risk of overfitting without regularization

**Use case**
Used to push performance beyond Random Forest by learning more refined patterns.

---

### 4. Support Vector Machine (SVM)

**Overview**
Support Vector Machines aim to find the optimal hyperplane that maximizes the margin between classes.

**Why it was used**

* Strong theoretical foundations
* Effective in high-dimensional feature spaces

**Strengths**

* Works well with clear class separation
* Robust to overfitting with proper kernel choice

**Limitations**

* Computationally expensive on large datasets
* Requires careful feature scaling
* Limited interpretability

**Use case**
Used to evaluate margin-based learning performance compared to tree-based models.

---

## Best Model Selection

The best model was selected automatically based on **ROC-AUC score**, as it provides a threshold-independent evaluation of classification performance.

* The selected model is saved as:
  **`/models/best_model.pkl`**

* All evaluation metrics are stored in:
  **`/evaluation/metrics.json`**

---

## Final Remarks

* Linear models provide interpretability and speed
* Tree-based ensembles deliver strong performance on tabular data
* Boosting methods offer the best accuracy when tuned properly
* Model selection should always be metric-driven and aligned with business objectives

This comparison ensures the final selected model is both **accurate** and **reliable** for downstream usage.
