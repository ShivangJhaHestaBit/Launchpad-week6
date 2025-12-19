import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TUNING_DIR = os.path.join(BASE_DIR, "tuning")

X_TRAIN_PATH = os.path.join(FEATURE_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(FEATURE_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(FEATURE_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(FEATURE_DIR, "y_test.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TUNING_DIR, exist_ok=True)

def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    return X_train, X_test, y_train, y_test

def tune_model(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid

def main():
    X_train, X_test, y_train, y_test = load_data()

    grid = tune_model(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

    results = {
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_roc_auc": roc_auc
    }

    with open(os.path.join(TUNING_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Hyperparameter tuning complete")
    print("Best ROC-AUC:", roc_auc)

if __name__ == "__main__":
    main()