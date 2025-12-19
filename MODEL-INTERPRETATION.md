# Model Interpretation & Explainability

This document explains how model decisions are interpreted and validated.

---

## Hyperparameter Tuning

GridSearchCV was used to tune RandomForest hyperparameters using 5-fold cross-validation.
The optimization metric was ROC-AUC to ensure robust class separation.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to understand global feature impact.
The SHAP summary plot highlights which features most influence survival predictions.

---

## Feature Importance

Tree-based feature importance was generated to rank features based on information gain.
This complements SHAP by providing model-native interpretability.

---

## Error Analysis

A confusion matrix heatmap was generated to analyze:
- False positives
- False negatives

This helps identify systematic model errors.

---

## Conclusion

The tuned model demonstrates improved performance over baseline models and provides
transparent decision-making through explainability tools.
