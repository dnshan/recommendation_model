import os
import pickle
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Matplotlib (NO tkinter, NO pyplot) ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


DATA_PATH = "training_data.csv"
MODEL_OUT = "policy_model.pkl"
PLOTS_DIR = "plots"


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def save_figure(fig, filename):
    """Save a matplotlib Figure using Agg canvas (no GUI)."""
    path = os.path.join(PLOTS_DIR, filename)
    FigureCanvas(fig).print_png(path)
    print(f"[saved] {path}")


def try_render_dot_to_png(dot_path, png_path):
    """
    Tries to render .dot -> .png using Graphviz 'dot'.
    If Graphviz isn't installed, it will fail gracefully.
    """
    cmd = f'dot -Tpng "{dot_path}" -o "{png_path}"'
    exit_code = os.system(cmd)
    if exit_code == 0:
        print(f"[saved] {png_path}")
    else:
        print("[info] Graphviz not found or 'dot' failed. DOT file saved (tree.dot) but PNG not rendered.")
        print("       To enable PNG rendering, install Graphviz and ensure 'dot' is in PATH.")


def train_policy_model():
    # 1) Load data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run generate_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    if "action_label" not in df.columns:
        raise ValueError("training_data.csv must contain 'action_label' column.")

    X = df.drop("action_label", axis=1)
    y = df["action_label"]

    print("\n--- Label Distribution ---")
    print(y.value_counts())

    # 2) Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3) Train model
    clf = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 4) Evaluate
    print("\n--- Model Evaluation ---")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    ensure_plots_dir()

    # Save report text (useful for report writing)
    report_path = os.path.join(PLOTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[saved] {report_path}")

    # 5) Rules snippet (text)
    rules = export_text(clf, feature_names=list(X.columns), max_depth=4)
    rules_path = os.path.join(PLOTS_DIR, "tree_rules.txt")
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(rules)
    print(f"[saved] {rules_path}")

    # =========================
    # Confusion Matrix Plot
    # =========================
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig_cm = Figure(figsize=(7, 6))
    ax_cm = fig_cm.add_subplot(111)
    ax_cm.imshow(cm)

    ax_cm.set_title("Policy Model – Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_xticks(range(len(labels)))
    ax_cm.set_yticks(range(len(labels)))
    ax_cm.set_xticklabels(labels, rotation=45, ha="right")
    ax_cm.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center")

    save_figure(fig_cm, "confusion_matrix.png")

    # =========================
    # Feature Importance Plot
    # =========================
    importances = pd.Series(
        clf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    fig_fi = Figure(figsize=(9, 5))
    ax_fi = fig_fi.add_subplot(111)
    ax_fi.bar(importances.index.astype(str), importances.values)
    ax_fi.set_title("Feature Importance (Decision Tree)")
    ax_fi.set_ylabel("Importance")
    ax_fi.set_xlabel("Feature")
    ax_fi.tick_params(axis="x", rotation=45)

    save_figure(fig_fi, "feature_importance.png")

    # =========================
    # Tree Export (Graphviz DOT)
    # =========================
    dot_path = os.path.join(PLOTS_DIR, "tree.dot")
    export_graphviz(
        clf,
        out_file=dot_path,
        feature_names=list(X.columns),
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        special_characters=True
    )
    print(f"[saved] {dot_path}")

    # Try to render DOT -> PNG if Graphviz is installed
    png_path = os.path.join(PLOTS_DIR, "tree.png")
    try_render_dot_to_png(dot_path, png_path)

    # 7) Save model
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {MODEL_OUT}")
    print(f"All outputs saved in ./{PLOTS_DIR}/")


if __name__ == "__main__":
    train_policy_model()
