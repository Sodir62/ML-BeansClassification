# Random Forest Bean Classification

## Scripts

### randomforest.py
Compares 4 RF configurations with increasing regularization (baseline, light, medium, heavy tuning). Each model is evaluated using 5-fold stratified CV. Outputs a comparison table and selects the best model based on lowest overfitting gap (train F1 - test F1).

### gridsearch.py
Exhaustive hyperparameter search over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`. Saves full CV results to `outputs/gridsearch_cv_results.csv`. Warning: takes a long time to run.

### visualizations.py
Generates plots for the two best model configurations:
- Class distribution bar chart
- Confusion matrices (raw counts and normalized)
- Feature importance ranking
- Learning curves
- Hyperparameter sensitivity plots

All images saved to `outputs/`.

## Best Models

**Low Gap (Heavy Regularization)**
- `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=8`, `n_estimators=100`
- CV F1 ~0.93, Gap ~2%

**Best F1**
- `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=4`, `n_estimators=100`

## Outputs

- **Best model**: Selected by lowest overfitting gap to ensure generalization
- **CV F1 (macro)**: Cross-validation F1 score averaged across folds
- **Overfitting gap**: Difference between train and test F1 (lower = better generalization)
