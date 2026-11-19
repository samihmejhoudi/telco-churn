import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from telco_churn.dataset import sha256_of_file
from telco_churn.evaluation import THRESHOLD, evaluate_model
from telco_churn.pipeline import FEATURES, build_pipeline, parse_features

SEED = 42
SOURCE = "data/raw/telco.csv"

# Same candidates and settings as scripts/compare_models.py (TP02 part 3)
CANDIDATES = {
    "majority_baseline": (DummyClassifier, {"strategy": "most_frequent"}),
    "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
    "decision_tree": (DecisionTreeClassifier, {"max_depth": 5, "random_state": SEED}),
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": 200, "max_depth": 8, "random_state": SEED},
    ),
}


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main(name):
    cls, params = CANDIDATES[name]

    # Same split as TP02 (test set is never used here)
    df = pd.read_csv(SOURCE)
    df["target"] = df["Churn"].map({"No": 0, "Yes": 1})
    train_df, rest_df = train_test_split(
        df, test_size=0.40, stratify=df["target"], random_state=SEED
    )
    valid_df, test_df = train_test_split(
        rest_df, test_size=0.50, stratify=rest_df["target"], random_state=SEED
    )
    X_train, y_train = parse_features(train_df), train_df["target"]
    X_valid, y_valid = parse_features(valid_df), valid_df["target"]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("telco-churn-tp03")

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({
            "model_type": cls.__name__,
            "split_seed": SEED,
            "threshold": THRESHOLD,
            **params,
        })
        mlflow.set_tags({
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_uncommitted_changes": str(bool(git_output("status", "--porcelain", "--untracked-files=no"))),
            "data_sha256": sha256_of_file(SOURCE),
        })

        # Dataset records: describe the data used (they do not store the full DataFrame)
        cols = FEATURES + ["Churn"]
        train_dataset = mlflow.data.from_pandas(
            train_df[cols], name="telco_train", targets="Churn"
        )
        valid_dataset = mlflow.data.from_pandas(
            valid_df[cols], name="telco_validation", targets="Churn"
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(valid_dataset, context="validation")

        pipe = build_pipeline(cls(**params))
        pipe.fit(X_train, y_train)  # training set only

        m = evaluate_model(pipe, X_valid, y_valid)  # validation only
        mlflow.log_metrics({
            "val_average_precision": m["average_precision"],
            "val_precision": m["precision"],
            "val_recall": m["recall"],
            "val_f1": m["f1"],
        })
        print("Validation:", {k: round(v, 4) for k, v in m.items() if k in ("average_precision", "precision", "recall", "f1")})


        # Artifact 1: validation confusion matrix (rows = true, columns = predicted)
        cm = m["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["No (0)", "Yes (1)"])
        ax.set_yticks([0, 1], ["No (0)", "Yes (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Validation, threshold {THRESHOLD}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center")
        fig.tight_layout()
        cm_path = Path("reports/val_confusion_matrix.png")
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        # Artifact 2: compact configuration / provenance record
        provenance = {
            "model": {"name": name, "class": cls.__name__, "params": params},
            "features": FEATURES,
            "target": "Churn",
            "class_mapping": {"No": 0, "Yes": 1},
            "positive_class": "Churn = Yes (1)",
            "split": {
                "scheme": "stratified 60/20/20",
                "seed": SEED,
                "sizes": {"train": len(train_df), "validation": len(valid_df), "test": len(test_df)},
                "allocation_reference": "models/split_allocation.json",
            },
            "evaluation": {
                "split_used": "validation only (test set reserved)",
                "primary_metric": "average_precision",
                "threshold": THRESHOLD,
                "confusion_matrix_order": "labels [0, 1]; rows = true, columns = predicted; [[TN, FP], [FN, TP]]",
            },
            "data_sha256": sha256_of_file(SOURCE),
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_uncommitted_changes": bool(git_output("status", "--porcelain", "--untracked-files=no")),
        }
        mlflow.log_dict(provenance, "provenance.json")

        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logistic_regression")




