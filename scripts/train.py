import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from telco_churn.dataset import sha256_of_file
from telco_churn.evaluation import THRESHOLD, evaluate_model
from telco_churn.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    parse_features,
)

SEED = 42
SOURCE = Path("data/raw/telco.csv")
MODELS = Path("models")
CLASSES = {
    "majority_baseline": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


# 1. Selected model and parameters come from the validation comparison
comparison = json.loads(Path("reports/model_comparison.json").read_text(encoding="utf-8"))
name = comparison["selected_model"]
params = next(r["params"] for r in comparison["results"] if r["model"] == name)

# 2. Same split and seed as before
df = pd.read_csv(SOURCE)
df["target"] = df["Churn"].map({"No": 0, "Yes": 1})
train_df, rest_df = train_test_split(
    df, test_size=0.40, stratify=df["target"], random_state=SEED
)
valid_df, test_df = train_test_split(
    rest_df, test_size=0.50, stratify=rest_df["target"], random_state=SEED
)

# 3. Fit the whole pipeline on the training set only
X_train, y_train = parse_features(train_df), train_df["target"]
X_valid, y_valid = parse_features(valid_df), valid_df["target"]
pipe = build_pipeline(CLASSES[name](**params))
pipe.fit(X_train, y_train)
metrics = evaluate_model(pipe, X_valid, y_valid)  # validation only, test untouched

# 4. Save the fitted pipeline (preprocessing + classifier together)
MODELS.mkdir(exist_ok=True)
joblib.dump(pipe, MODELS / "pipeline.joblib")

# 5. Sample batch (raw columns, no Churn) and its original scores
sample_raw = valid_df.head(20)[["customerID"] + FEATURES]
sample_raw.to_csv(MODELS / "sample_batch.csv", index=False)
scores = pipe.predict_proba(parse_features(sample_raw))[:, list(pipe.classes_).index(1)]
(MODELS / "reference_scores.json").write_text(
    json.dumps({"tolerance": 1e-9, "scores": [float(s) for s in scores]}, indent=2),
    encoding="utf-8",
)

# 6. Customer allocation
(MODELS / "split_allocation.json").write_text(
    json.dumps({
        "seed": SEED,
        "train": train_df["customerID"].tolist(),
        "validation": valid_df["customerID"].tolist(),
        "test": test_df["customerID"].tolist(),
    }),
    encoding="utf-8",
)

# 7. Metadata record
metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source": {"path": str(SOURCE), "sha256": sha256_of_file(SOURCE)},
    "seed": SEED,
    "split": {
        "scheme": "stratified 60/20/20",
        "sizes": {"train": len(train_df), "validation": len(valid_df), "test": len(test_df)},
        "allocation_file": "models/split_allocation.json",
    },
    "positive_class": "Churn = Yes (1)",
    "threshold": THRESHOLD,
    "feature_roles": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
    "preprocessing": {
        "numeric": "SimpleImputer(median) + StandardScaler",
        "categorical": "SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown=ignore)",
        "parsing": "blank-only numeric text -> missing; other non-numeric text raises an error",
    },
    "model": {"name": name, "class": CLASSES[name].__name__, "params": params},
    "validation_metrics": metrics,
    "artifact": "models/pipeline.joblib",
    "code_and_environment": {
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "scikit-learn": version("scikit-learn"),
        "pandas": version("pandas"),
        "numpy": version("numpy"),
        "joblib": version("joblib"),
    },
}
(MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("Saved fitted pipeline for:", name)
print("Validation average precision:", round(metrics["average_precision"], 4))
print("Files:", sorted(p.name for p in MODELS.iterdir()))
