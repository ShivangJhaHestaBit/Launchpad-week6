import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "evaluation", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

def select_features(X, y, top_k=15):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    importances.head(top_k).sort_values().plot(kind="barh")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(plot_path)
    plt.close()

    return list(importances.head(top_k).index)
