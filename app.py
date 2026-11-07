import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

st.set_page_config(page_title="Shopper Intent Classifier", page_icon="🛒", layout="wide")

TARGET = "Revenue"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(BASE, "model", MODEL_FILES[name]))


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, proba, scores


# ---------------- sidebar ----------------
st.sidebar.title("🛒 Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + Revenue)", type="csv")
choice = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
compare_all = st.sidebar.checkbox("Also compare all 5 models", value=True)
st.sidebar.caption("Models are pre-trained and loaded from /model. Nothing is retrained on your upload.")

# ---------------- header ----------------
st.title("Online Shopper Purchase-Intent Predictor")
st.write("Upload held-out session data, pick a classifier, and see how well it "
         "predicts whether a visitor will make a purchase (`Revenue`).")

if uploaded is None:
    st.info("👈 Upload a CSV in the sidebar to begin. You can use `test_data.csv` from the repo.")
    st.stop()

data = pd.read_csv(uploaded)
if TARGET not in data.columns:
    st.error(f"The uploaded file must contain the target column `{TARGET}`.")
    st.stop()

data["Weekend"] = data["Weekend"].astype(int)
y = data[TARGET].astype(int)
X = data.drop(columns=TARGET)

with st.expander(f"Preview of uploaded data ({len(data):,} rows)"):
    st.dataframe(data.head(20))
    st.write("Class balance:", y.value_counts().rename({0: "No purchase", 1: "Purchase"}).to_dict())

# ---------------- selected model ----------------
try:
    model = load_model(choice)
    pred, proba, scores = evaluate(model, X, y)
except Exception as e:  # missing columns, bad values, etc.
    st.error(f"Could not score this file: {e}")
    st.stop()

st.subheader(f"Results: {choice}")
cols = st.columns(6)
for col, (k, v) in zip(cols, scores.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y, proba)
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
    ax2.plot(fpr, tpr, label=f"AUC = {scores['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("**Classification report**")
report = classification_report(y, pred, target_names=["No purchase", "Purchase"], output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(3))

# ---------------- comparison ----------------
if compare_all:
    st.subheader("All models on this file")
    rows = []
    for name in MODEL_FILES:
        _, _, s = evaluate(load_model(name), X, y)
        rows.append({"ML Model Name": name, **s})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"))
    st.bar_chart(comp[["Accuracy", "AUC", "F1", "MCC"]])
