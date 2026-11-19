import json
from pathlib import Path

from telco_churn.dataset import build_report

source = Path("data/raw/telco.csv")
out = Path("reports/data_overview.json")

report = build_report(source)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("Report saved to", out)
print("Rows x columns:", report["shape"]["rows"], "x", report["shape"]["columns"])
print("Target counts:", report["target"]["counts"])
print("Duplicate customer IDs:", report["duplicate_customer_ids"])