"""Train five classifiers on the Online Shoppers Purchasing Intention dataset.

Outputs
-------
model/<name>.joblib   full pipeline (preprocessing + classifier) per model
test_data.csv         raw hold-out test split (features + Revenue) for the app
metrics_summary.csv   metric table used in README
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "Revenue"

df = pd.read_csv("online_shoppers_intention.csv")
df["Weekend"] = df["Weekend"].astype(int)
df[TARGET] = df[TARGET].astype(int)

X, y = df.drop(columns=TARGET), df[TARGET]
cat_cols = ["Month", "VisitorType"]
# these are numeric codes but really categories
cat_cols += ["OperatingSystems", "Browser", "Region", "TrafficType"]
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# keep the untouched test rows for the Streamlit app
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)


def make_pipe(clf, scale=True, use_cat=True):
    parts = [("num", StandardScaler() if scale else "passthrough", num_cols)]
    if use_cat:  # Gaussian NB assumes continuous inputs, so it skips one-hot columns
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    pre = ColumnTransformer(parts, remainder="drop")
    return Pipeline([("prep", pre), ("clf", clf)])


models = {
    "Logistic Regression": make_pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    "Decision Tree": make_pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=SEED), scale=False),
    "KNN": make_pipe(KNeighborsClassifier(n_neighbors=11)),
    "Naive Bayes": make_pipe(GaussianNB(), use_cat=False),
    "Support Vector Machine": make_pipe(SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=SEED)),
}

rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(pipe, f"model/{name.lower().replace(' ', '_')}.joblib")

res = pd.DataFrame(rows).round(4)
res.to_csv("metrics_summary.csv", index=False)
print(res.to_string(index=False))
print(len(X_train), len(X_test))
