# Handover note

**Tester:** myself, on a clean clone (`git clone -b docs/project-onboarding ... telco-churn-test`).
No other pair was available, so this is interim evidence.

**What worked:** `uv sync --locked` installed the environment, the hash of
`data/raw/telco.csv` matched the README, and `uv run python scripts/inspect_data.py`
gave 7043 x 21, 5174 No / 1869 Yes and 0 duplicate customer IDs.

**Issue found:** the clean clone had no `data/raw/` folder, and the README did not say
to create it before copying the CSV.

**Fix:** the README now says to create `data/raw/` first (commit 5a696ee, PR #1).

**Did the revised instructions work:** Yes. After the fix, the clean clone README contains the instruction to create data/raw/ first.

**Remaining limits:** the raw CSV is not in Git, so a teammate must get it from the
instructor. A trained model also needs the training config, the random seeds, the
data split and the package versions, none of which exist until TP02.
