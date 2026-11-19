import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from telco_churn.pipeline import build_pipeline, parse_features

SEED = 42

df = pd.read_csv("data/raw/telco.csv")
df["target"] = df["Churn"].map({"No": 0, "Yes": 1})

train_df, rest_df = train_test_split(
    df, test_size=0.40, stratify=df["target"], random_state=SEED
)
valid_df, test_df = train_test_split(
    rest_df, test_size=0.50, stratify=rest_df["target"], random_state=SEED
)

X_train, y_train = parse_features(train_df), train_df["target"]
X_valid, y_valid = parse_features(valid_df), valid_df["target"]

print("Missing TotalCharges after parsing (train):", int(X_train["TotalCharges"].isna().sum()))

pipe = build_pipeline(LogisticRegression(max_iter=1000))
pipe.fit(X_train, y_train)          # one call trains preprocessing + classifier
pred = pipe.predict(X_valid)        # raw six columns, no manual encoding

print("Positive class: Churn = Yes (1). Matrix rows = true, columns = predicted")
print("Accuracy:", accuracy_score(y_valid, pred))
print(confusion_matrix(y_valid, pred, labels=[0, 1]))
