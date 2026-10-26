"""Credit Approval experiment: English export of the supplied notebook.

Preprocessing intentionally matches the recorded workflow; see README caveats.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")






def load_data():
    data = fetch_ucirepo(id=27)
    X = data.data.features
    y = data.data.targets

    X.columns = [f"A{i}" for i in range(1, 16)]

    df = pd.concat([X, y], axis=1)

    # Standardize the target column name.
    df.rename(columns={df.columns[-1]: "class"}, inplace=True)

    return df

def preprocess_data(df):
    print("Starting preprocessing...")

    # Separate the target from the input features.
    y = df["class"].copy()
    X = df.drop("class", axis=1).copy()

    # ? → NaN
    X = X.replace("?", np.nan)

    # Impute missing categorical values with the mode and numeric values with the median.
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].median())

    print("Shape after missing-value imputation:", X.shape)

    # One-hot encode input features only.
    X = pd.get_dummies(X, drop_first=True)

    print("Shape after encoding:", X.shape)

    # Clean and encode the target labels.
    y = y.astype(str).str.strip()
    y = y.map({"+": 1, "-": 0})

    # Align target and feature indices.
    y = y.loc[X.index]

    # Combine the processed features and target.
    df_final = pd.concat([X, y.rename("class")], axis=1)

    return df_final

from sklearn.model_selection import train_test_split
def split_data(df):
    X = df.drop("class", axis=1)
    y = df["class"]
    print("\nTarget distribution:")
    print(y.value_counts())
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

df = load_data()
df = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(df)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.preprocessing import StandardScaler

print("\n--- FEATURE SCALING ---")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train = X_train_scaled
X_test = X_test_scaled

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True)
}

print("Training target distribution:")
print(pd.Series(y_train).value_counts())

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(name, "trained")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

print("\n==============================")
print("GRID SEARCH (Gradient Boosting)")
print("==============================")

param_grid = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_

print("\n==============================")
print("CROSS VALIDATION (Best Model)")
print("==============================")

cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="f1")

print("CV Scores:", cv_scores)
print("CV Mean F1 Score:", cv_scores.mean())

results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
results_df.sort_values("ROC_AUC", ascending=False)

for name, model in trained_models.items():
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(name)
    plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def plot_train_test_accuracy(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    model_names = []
    train_scores = []
    test_scores = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        # TRAIN accuracy
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        # TEST accuracy
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        model_names.append(name)
        train_scores.append(train_acc)
        test_scores.append(test_acc)

        print(f"\n{name}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy : {test_acc:.4f}")

    # ---------------------------
    # PLOT (TRAIN vs TEST)
    # ---------------------------
    plt.figure(figsize=(10,6))

    plt.plot(model_names, train_scores, marker='o', label="Train Accuracy")
    plt.plot(model_names, test_scores, marker='o', label="Test Accuracy")

    plt.title("Train vs Test Accuracy (Overfitting Check)")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.xticks(rotation=30)

    plt.show()
plot_train_test_accuracy(X_train_scaled, X_test_scaled, y_train, y_test)