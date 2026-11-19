import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless parsing: select the six features and convert numeric types.

    Whitespace-only numeric values become missing (NaN). Any other value that
    is not a number raises an error (errors="raise"), it is never silently
    turned into missing.
    """
    out = df[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        text = out[col].astype("string")
        blank = text.str.strip().eq("").fillna(False)
        text = text.mask(blank)
        out[col] = pd.to_numeric(text, errors="raise").astype("float64")
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(object)
    return out


def build_pipeline(classifier) -> Pipeline:
    """Return an UNFITTED pipeline: preprocessing + classifier."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
