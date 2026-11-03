from pathlib import Path

from telco_churn.dataset import sha256_of_file

source = Path("data/raw/telco.csv")

print("File:", source)
print("Size (bytes):", source.stat().st_size)
print("SHA-256:", sha256_of_file(source))