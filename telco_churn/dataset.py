from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import platform
import sys

import pandas as pd


def sha256_of_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_report(source, target="Churn", id_col="customerID") -> dict:
    source = Path(source)
    df = pd.read_csv(source)

    # Empty or whitespace-only values are NOT the same as NaN, so check both.
    as_text = df.astype("string")
    blank_counts = as_text.apply(lambda s: s.str.strip().eq("").sum())

    counts = df[target].value_counts(dropna=False)
    props = df[target].value_counts(normalize=True, dropna=False)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_of_file(source),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": version("pandas"),
        },
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "target": {
            "column": target,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "proportions": {str(k): float(v) for k, v in props.items()},
        },
        "duplicate_customer_ids": int(df[id_col].duplicated().sum()),
        "missing_values": {
            "isna": {c: int(n) for c, n in df.isna().sum().items()},
            "empty_or_whitespace": {c: int(n) for c, n in blank_counts.items()},
        },
    }