# Telco Churn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predict customer churn for a telecom company from an IBM Telco customer snapshot.
Business question: which customers are likely to leave (`Churn` = `Yes`), based on
their account and service information?

## Project status

- **TP01 (done):** reproducible project, data identification, inspection report,
  environment recreation evidence, Git workflow.
- **TP02 (next):** preprocessing and training pipelines, built inside this same
  project (fitted pipelines will be saved in `models/`).

## Setup

Requirements: Python 3.12, Git and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/samihmejhoudi/telco-churn.git
cd telco-churn
uv sync --locked
```

`uv sync --locked` recreates the environment from `uv.lock` and fails if the
lock file does not match `pyproject.toml`.

Check that the package imports:

```sh
uv run python -c "import telco_churn; print(telco_churn.__file__)"
```

## Source data

The raw CSV is **not stored in Git**. Get it from the instructor, or download the
commit-pinned copy from the
[IBM Telco repository](https://github.com/IBM/telco-customer-churn-on-icp4d), and
save it as `data/raw/telco.csv`. Never edit this file.

Verify it is the same file:

- Size: 970457 bytes
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

More details and field roles are in `references/data_source.md`.

## Run the data inspection

```sh
uv run python scripts/inspect_data.py
```

This reads `data/raw/telco.csv` and writes `reports/data_overview.json`. The report
contains: dimensions and column types, target counts and proportions, duplicate
customer IDs, missing values (including empty or whitespace-only values), the source
hash, and the Python and pandas versions used.

Result on the supplied file: 7043 rows x 21 columns; `Churn` = 5174 `No` and
1869 `Yes`; 0 duplicate customer IDs.

## Environment recreation evidence

`evidence/tp01/` holds the report before and after recreating the environment
(`before_recreation.json`, `after_recreation.json`) and the comparison
(`recreation_comparison.md`).

## Project Organization

This is the Cookiecutter Data Science (v2) template layout. Not every folder is
used yet: the TP01 files are listed in the table below the tree, and the modeling
files are for TP02.

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see [www.mkdocs.org](https://www.mkdocs.org) for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         telco_churn and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── telco_churn   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes telco_churn a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

### Files used in this project

| Path | Purpose | TP |
|---|---|---|
| `telco_churn/dataset.py` | Inspection code (hash, report) | TP01 |
| `scripts/inspect_data.py` | Command-line entry point for the inspection | TP01 |
| `data/raw/` | Original, unchanged CSV (ignored by Git) | TP01 |
| `references/data_source.md` | Data source, hash and field notes | TP01 |
| `reports/data_overview.json` | Generated inspection report | TP01 |
| `evidence/tp01/` | Recreation evidence | TP01 |
| `telco_churn/modeling/` or pipeline code | Training and prediction code | TP02 |
| `models/` | Saved fitted pipelines | TP02 |

## Template

Generated with Cookiecutter Data Science (v2). Choices: environment manager uv,
dependency file `pyproject.toml`, no code scaffold, no dataset storage, no docs.
CCDS version: _fill in_.