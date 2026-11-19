# Environment recreation comparison

**Before:** `before_recreation.json` (original `.venv`)
**After:** `after_recreation.json` (`.venv` recreated with `uv sync --locked`)

The old environment was renamed to `.venv_old`, not deleted. The same
`scripts/inspect_data.py` and the same `data/raw/telco.csv` were used both times.

| Field | Result |
|---|---|
| source (path, size, SHA-256) | SAME |
| environment (Python, platform, pandas) | SAME |
| shape | SAME |
| dtypes | SAME |
| target counts and proportions | SAME |
| duplicate customer IDs | SAME |
| missing values | SAME |
| generated_at_utc | DIFFERENT |

The only difference is `generated_at_utc`, which is the time of each run.
`uv.lock` records the selected package versions, but it does not freeze
hardware, operating-system details or future training behavior.