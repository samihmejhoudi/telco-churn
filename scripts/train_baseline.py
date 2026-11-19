import json
from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

SEED = 42

# 1. Load dataset
df = pd.read_csv("data/raw/telco.csv")

# 2. Select the two features
X = df[["tenure", "MonthlyCharges"]]

# 3. Encode target: No = 0, Yes = 1 (positive class: Churn = Yes)
y = df["Churn"].map({"No": 0, "Yes": 1})

# 4. Split into train, validation and test (60 / 20 / 20, stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=SEED,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp,
)

# 4b. Check: no customer appears in two sets (X_* keep the row index of df)
train_ids = set(df.loc[X_train.index, "customerID"])
val_ids = set(df.loc[X_val.index, "customerID"])
test_ids = set(df.loc[X_test.index, "customerID"])
assert not (train_ids & val_ids), "train/validation overlap"
assert not (train_ids & test_ids), "train/test overlap"
assert not (val_ids & test_ids), "validation/test overlap"
print("Split sizes:", len(train_ids), len(val_ids), len(test_ids))
print("ID overlap check passed")

# 5. Create and train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Logistic Regression trained successfully!")

# 6. Predict on validation data
y_val_pred = model.predict(X_val)

# 7. Evaluate the model
print("\n--- Logistic Regression Results (validation) ---")
print("Positive class: Churn = Yes (1). Matrix rows = true, columns = predicted, order [0, 1]")
print("Accuracy:", accuracy_score(y_val, y_val_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_val_pred,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
))

# 8. Train Dummy Classifier (fitted on the TRAINING set)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

# 9. Evaluate Dummy Classifier
y_dummy_pred = dummy.predict(X_val)

print("\n--- Dummy Classifier Results (validation) ---")
print("Accuracy:", accuracy_score(y_val, y_dummy_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_dummy_pred, labels=[0, 1]))

# 10. Save the results
results = {
    "seed": SEED,
    "features": list(X.columns),
    "positive_class": "Churn = Yes (1)",
    "confusion_matrix_layout": "rows = true label, columns = predicted label, order [0, 1]",
    "split_sizes": {
        "train": len(train_ids),
        "validation": len(val_ids),
        "test": len(test_ids),
    },
    "logistic_regression": {
        "accuracy": float(accuracy_score(y_val, y_val_pred)),
        "confusion_matrix": confusion_matrix(y_val, y_val_pred, labels=[0, 1]).tolist(),
    },
    "majority_baseline": {
        "accuracy": float(accuracy_score(y_val, y_dummy_pred)),
        "confusion_matrix": confusion_matrix(y_val, y_dummy_pred, labels=[0, 1]).tolist(),
    },
}
out = Path("reports/first_model_results.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nResults saved to", out)

