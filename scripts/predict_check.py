import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from telco_churn.pipeline import parse_features

MODELS = Path("models")

# Inference only: no fit call, no training labels, no training data.
pipe = joblib.load(MODELS / "pipeline.joblib")  # trusted artifact only
batch = pd.read_csv(MODELS / "sample_batch.csv")
reference = json.loads((MODELS / "reference_scores.json").read_text(encoding="utf-8"))

scores = pipe.predict_proba(parse_features(batch))[:, list(pipe.classes_).index(1)]
expected = np.array(reference["scores"])
tolerance = reference["tolerance"]

max_diff = float(np.max(np.abs(scores - expected)))
print("Customers predicted:", len(scores))
print("Max absolute score difference:", max_diff, "| tolerance:", tolerance)
assert max_diff <= tolerance, "Reloaded predictions differ from the original ones"
assert ((scores >= 0.5).astype(int) == (expected >= 0.5).astype(int)).all()
print("Reload check passed: same scores and same class predictions at threshold 0.5")
