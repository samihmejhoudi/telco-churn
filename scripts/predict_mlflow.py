import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from telco_churn.pipeline import parse_features

MODEL_URI = "models:/telco-churn@candidate"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.sklearn.load_model(MODEL_URI)  # no fit anywhere in this script

raw = pd.read_csv("models/sample_batch.csv")
X = parse_features(raw)

classes = list(model.classes_)
scores = model.predict_proba(X)[:, classes.index(1)]
labels = model.predict(X)

ref = json.loads(Path("models/reference_scores.json").read_text(encoding="utf-8"))
diff = float(np.max(np.abs(scores - np.array(ref["scores"]))))

print("Loaded:", MODEL_URI)
print("Customers scored:", len(raw))
print("Predicted churners (class 1):", int(labels.sum()))
print("Max score difference vs TP02 reference:", diff, "| tolerance:", ref["tolerance"])
print("Prediction agreement:", "OK" if diff <= ref["tolerance"] else "MISMATCH")
