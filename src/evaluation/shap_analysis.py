import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "evaluation", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

X_TEST_PATH = os.path.join(FEATURE_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(FEATURE_DIR, "y_test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

def save_plot(fig, filename: str):
    """Save and close a matplotlib figure."""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_feature_importance(model, X_test, top_n=15):
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    importances = importances.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(top_n).plot(kind="bar", ax=ax)
    ax.set_title("Top Feature Importances")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    save_plot(fig, "feature_importance.png")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    save_plot(fig, "confusion_matrix.png")

def plot_shap_summary(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    save_plot(fig, "shap_summary.png")

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("best_model.pkl not found")

    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    plot_shap_summary(model, X_test)
    plot_feature_importance(model, X_test)
    
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)

    print("SHAP analysis completed successfully")

if __name__ == "__main__":
    main()
