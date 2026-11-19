# TP03 summary: MLflow tracking, comparison and handover

## 1. Candidate comparison (validation set, threshold 0.5, same split seed 42, same six features)

Screenshot: `evidence/tp03/compare_metrics.png`

| Run ID | Candidate / changed setting | Val AP | Val recall | Val F1 | Decision |
|---|---|---:|---:|---:|---|
| 29eec016be0c40f2af28e2d39c451c02 | majority_baseline | 0.2654 | 0.0000 | 0.0000 | reference only |
| 1e953027be7c43fe944b62e25396ce85 | logistic_regression (max_iter 1000) | 0.6475 | 0.5294 | 0.5893 | not selected |
| 4ce7b4c2c87446b0ba3b15b11802b43f | decision_tree (max_depth 5) | 0.6033 | 0.6016 | 0.6156 | not selected |
| 809aa2b0e3f34a0c8fba42c1be44e29f | random_forest (200 trees, max_depth 8) | 0.6703 | 0.5053 | 0.5727 | selected |

## 2. Selection reason

I selected random forest using the rule fixed in TP02: highest validation average precision, with the simpler model preferred only if within 0.005. Random forest reached 0.6703, ahead of logistic regression (0.6475) and decision tree (0.6033), so the gap is larger than the tie margin and simplicity did not apply. The decision tree has the best recall and F1 at threshold 0.5, but these depend on one threshold and were not the primary metric. All runs use the same features, customer allocation and validation set. This is one split, not proof of a statistically significant improvement, and the test set stays unused.

## 3. Selected run screenshots

`evidence/tp03/selected_run_1.png` and `evidence/tp03/selected_run_2.png` (parameters, validation metrics, logged model, tags, training and validation dataset records: `telco_train` digest 6834438a with context training, `telco_validation` digest 4fb2e5e6 with context validation).

## 4. Registry screenshot

`evidence/tp03/registry.png` (`telco-churn`, version 1, alias `candidate`).

## 5. Identifiers

| Item | Value |
|---|---|
| Source run ID (confirmation run of random_forest) | 93ca698043864ebba18816ef98242cce |
| Logged model URI | models:/m-6bd6bb10cb3e4821a82533d9346f2356 |
| Registered model / version | telco-churn / 1 |
| Alias | candidate -> 1 |
| Git commit | 667e498d58c036db0e26a7e1c304f2baf1b68903 (no uncommitted changes) |
| Original CSV SHA-256 | 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91 |

The selected pipeline was logged in a confirmation run with the exact selected configuration (same metrics as run 809aa2b0...), because the earlier runs had no logged model.

## 6. Fresh-process inference

Command: `uv run python scripts/predict_mlflow.py` (loads `models:/telco-churn@candidate`, no fit).

Result: 20 customers scored, max score difference vs TP02 reference 0.0 (tolerance 1e-9), prediction agreement OK.

## 7. Restart observation

After stopping and restarting the server from the same folder with the same command, the runs, `telco-churn` version 1 and the alias `candidate` were still present, and inference gave the same result (difference 0.0).

## 8. Storage questions

- `mlflow.db` preserves run records, parameters, metrics, tags, dataset records and registry metadata (names, versions, aliases).
- `mlartifacts/` preserves the files: model files, confusion matrix, provenance JSON.
- A colleague on another machine must obtain separately: the artifact files, the project code (`telco_churn` package), the locked dependencies (`uv.lock`), the original CSV (identified by its hash) and access to a server or exported artifacts.
- If the database survived but the model files were lost, the records remain but the model cannot be loaded. If only the files survived, the links between run, version and alias are lost.
