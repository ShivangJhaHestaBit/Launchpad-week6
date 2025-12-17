import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def select_features(X, y, top_k=15):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importances.head(top_k).plot(kind="bar")
    plt.title("Top Feature Importances")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()

    return list(importances.head(top_k).index)
