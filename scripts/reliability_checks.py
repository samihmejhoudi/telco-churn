import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn.pipeline import parse_features

SEED = 42
evidence = {}

df = pd.read_csv("data/raw/telco.csv")
df["target"] = df["Churn"].map({"No": 0, "Yes": 1})
train_df, rest_df = train_test_split(df, test_size=0.40, stratify=df["target"], random_state=SEED)
valid_df, test_df = train_test_split(rest_df, test_size=0.50, stratify=rest_df["target"], random_state=SEED)
pipe = joblib.load("models/pipeline.joblib")
pos = list(pipe.classes_).index(1)

# 1. No customer ID overlap across the three sets
ids = {"train": set(train_df["customerID"]),
       "validation": set(valid_df["customerID"]),
       "test": set(test_df["customerID"])}
assert not (ids["train"] & ids["validation"])
assert not (ids["train"] & ids["test"])
assert not (ids["validation"] & ids["test"])
evidence["id_overlap"] = {"train_validation": 0, "train_test": 0, "validation_test": 0}
print("1. No customer ID overlap: OK")

# 2. Learned imputer statistics come from the TRAINING data, not the full dataset
cols = ["tenure", "MonthlyCharges", "TotalCharges"]
num_imputer = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
learned = num_imputer.statistics_
train_medians = parse_features(train_df)[cols].median().values
full_medians = parse_features(df)[cols].median().values
assert np.allclose(learned, train_medians)
print("2. Learned medians:        ", learned.tolist())
print("   Training-only medians:  ", train_medians.tolist(), "-> match: OK")
print("   Full-dataset medians:   ", full_medians.tolist(),
      "(differ from learned)" if not np.allclose(learned, full_medians) else "(same here by chance)")
evidence["imputer_statistics"] = {
    "learned": learned.tolist(),
    "training_medians": train_medians.tolist(),
    "full_dataset_medians": full_medians.tolist(),
}

# 3. A prediction call does not refit or change the statistics
before = num_imputer.statistics_.copy()
pipe.predict_proba(parse_features(valid_df))
assert np.array_equal(before, num_imputer.statistics_)
print("3. Statistics unchanged after predict: OK")

# 4a. Permitted missing numeric input: imputed with the training median
row = parse_features(valid_df.head(1)).copy()
row["TotalCharges"] = np.nan
p_missing = float(pipe.predict_proba(row)[0, pos])
print("4a. Missing TotalCharges -> imputed, score:", round(p_missing, 4))

# 4b. Unseen category: ignored by the encoder, not an error
row = parse_features(valid_df.head(1)).copy()
row["Contract"] = "Three year"
p_unseen = float(pipe.predict_proba(row)[0, pos])
print("4b. Unseen Contract value -> still predicts, score:", round(p_unseen, 4),
      "(handled, but not proof the prediction is reliable)")
evidence["policy_checks"] = {"missing_numeric_score": p_missing, "unseen_category_score": p_unseen}

# 5. Missing required feature fails visibly
bad = parse_features(valid_df.head(3)).drop(columns=["Contract"])
try:
    pipe.predict(bad)
    raise SystemExit("5. FAILED: prediction succeeded without Contract")
except SystemExit:
    raise
except Exception as exc:
    print("5. Missing Contract column -> visible error:", type(exc).__name__)
    evidence["missing_feature_error"] = type(exc).__name__

# 5b. Malformed non-blank numeric text also fails visibly
bad_text = valid_df.head(3).copy()
bad_text["TotalCharges"] = ["12.5", "abc", "30"]
try:
    parse_features(bad_text)
    raise SystemExit("5b. FAILED: malformed text was accepted")
except SystemExit:
    raise
except Exception as exc:
    print("5b. Malformed TotalCharges text -> visible error:", type(exc).__name__)
    evidence["malformed_text_error"] = type(exc).__name__

# 6. Reload equality is covered by scripts/predict_check.py
evidence["reload_check"] = "see scripts/predict_check.py"

Path("reports").mkdir(exist_ok=True)
Path("reports/reliability_checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print("\nSaved reports/reliability_checks.json")
