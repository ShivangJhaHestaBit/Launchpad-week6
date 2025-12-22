import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURE_DIR = os.path.join(PROJECT_ROOT, "src", "features")
TRAIN_PATH = os.path.join(FEATURE_DIR, "X_train.csv")
LOG_PATH = os.path.join(PROJECT_ROOT, "prediction_logs.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "src", "monitoring", "drift_report.csv")

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

def calculate_psi(expected, actual, buckets=10):
    expected = expected.dropna()
    actual = actual.dropna()

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    psi = 0.0
    for i in range(len(breakpoints) - 1):
        exp_pct = (
            (expected >= breakpoints[i]) &
            (expected < breakpoints[i + 1])
        ).mean()

        act_pct = (
            (actual >= breakpoints[i]) &
            (actual < breakpoints[i + 1])
        ).mean()

        exp_pct = max(exp_pct, 1e-6)
        act_pct = max(act_pct, 1e-6)

        psi += (exp_pct - act_pct) * np.log(exp_pct / act_pct)

    return psi

def main():
    if not os.path.exists(LOG_PATH):
        print("No prediction logs found. Drift check skipped.")
        return

    if not os.path.exists(TRAIN_PATH):
        print("X_train.csv not found. Drift check skipped.")
        return

    X_train = pd.read_csv(TRAIN_PATH)
    logs = pd.read_csv(LOG_PATH)
    meta_cols = ["request_id", "prediction", "probability", "timestamp", "model_version"]
    logs = logs.drop(columns=[c for c in meta_cols if c in logs.columns])
    X_train = X_train.select_dtypes(include=np.number)
    logs = logs.select_dtypes(include=np.number)

    drift_results = []

    common_features = set(X_train.columns).intersection(set(logs.columns))

    if not common_features:
        print("No common features between training and production data.")
        return

    for feature in sorted(common_features):
        psi_value = calculate_psi(
            X_train[feature],
            logs[feature]
        )

        drift_results.append({
            "feature": feature,
            "psi": round(psi_value, 4),
            "drift_level":
                "NO DRIFT" if psi_value < 0.1 else
                "MILD DRIFT" if psi_value < 0.2 else
                "SIGNIFICANT DRIFT"
        })

    report = pd.DataFrame(drift_results)
    report.to_csv(REPORT_PATH, index=False)
    print("\n Drift report generated:", REPORT_PATH)

if __name__ == "__main__":
    main()
