import json
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from telco_churn.evaluation import evaluate_model
from telco_churn.pipeline import FEATURES, build_pipeline, parse_features

SEED = 42

# Chosen BEFORE looking at the results (see the selection note below).
PRIMARY_METRIC = "average_precision"
TIE_MARGIN = 0.005
SIMPLICITY_ORDER = ["logistic_regression", "decision_tree", "random_forest"]

df = pd.read_csv("data/raw/telco.csv")
df["target"] = df["Churn"].map({"No": 0, "Yes": 1})

train_df, rest_df = train_test_split(
    df, test_size=0.40, stratify=df["target"], random_state=SEED
)
valid_df, test_df = train_test_split(
    rest_df, test_size=0.50, stratify=rest_df["target"], random_state=SEED
)  # test_df is NOT used in this script

X_train, y_train = parse_features(train_df), train_df["target"]
X_valid, y_valid = parse_features(valid_df), valid_df["target"]

candidates = {
    "majority_baseline": (DummyClassifier, {"strategy": "most_frequent"}),
    "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
    "decision_tree": (DecisionTreeClassifier, {"max_depth": 5, "random_state": SEED}),
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": 200, "max_depth": 8, "random_state": SEED},
    ),
}

results = []
for name, (cls, params) in candidates.items():
    pipe = build_pipeline(cls(**params))  # fresh, unfitted pipeline each time
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    metrics = evaluate_model(pipe, X_valid, y_valid)
    results.append({"model": name, "params": params,
                    "fit_seconds": round(fit_seconds, 3), **metrics})

table = pd.DataFrame(results)
print(table[["model", "average_precision", "precision", "recall", "f1",
             "accuracy", "fit_seconds"]].round(3).to_string(index=False))
for r in results:
    print(r["model"], "confusion matrix [[TN, FP], [FN, TP]]:", r["confusion_matrix"])

# Selection rule (validation only): best average precision among learned models;
# if several are within TIE_MARGIN of the best, pick the simplest.
learned = [r for r in results if r["model"] != "majority_baseline"]
best_score = max(r[PRIMARY_METRIC] for r in learned)
close = [r["model"] for r in learned if best_score - r[PRIMARY_METRIC] <= TIE_MARGIN]
selected = min(close, key=SIMPLICITY_ORDER.index)
print("\nSelected model:", selected, "| best", PRIMARY_METRIC, "=", round(best_score, 4),
      "| within tie margin:", close)

record = {
    "seed": SEED,
    "features": FEATURES,
    "positive_class": "Churn = Yes (1)",
    "split_sizes": {"train": len(train_df), "validation": len(valid_df), "test": len(test_df)},
    "evaluated_on": "validation only (test set reserved)",
    "selection_rule": {
        "primary_metric": PRIMARY_METRIC,
        "why": "Churners are the minority class; average precision uses the positive-class "
               "scores and does not depend on one threshold or on the majority class.",
        "tie_margin": TIE_MARGIN,
        "tie_break": "choose the simpler model; order: " + ", ".join(SIMPLICITY_ORDER),
    },
    "selected_model": selected,
    "results": results,
}
out = Path("reports/model_comparison.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2), encoding="utf-8")
print("Saved", out)
