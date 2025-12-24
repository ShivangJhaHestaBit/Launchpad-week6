import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

X_train = pd.read_csv("../features/X_train.csv")
y_train = pd.read_csv("../features/y_train.csv").values.ravel()

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = {}
best_model = None
best_f1 = 0

for name, model in models.items():
    scores = cross_validate(
        model, X_train, y_train,
        cv=cv, scoring=scoring
    )

    avg_scores = {k: np.mean(scores[f"test_{k}"]) for k in scoring}
    results[name] = avg_scores

    if avg_scores["f1"] > best_f1:
        best_f1 = avg_scores["f1"]
        best_model = model

best_model.fit(X_train, y_train)
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, "../models/best_model.pkl")

with open("../evaluation/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("Best model saved with scaling included")
