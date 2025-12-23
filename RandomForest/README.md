# Random Forest Bean Classification

## Scripts

### randomforest.py
Compares 4 RF configurations with increasing regularization (baseline, light, medium, heavy tuning). Each model is evaluated using 5-fold stratified CV. Outputs a comparison table and selects the best model based on lowest overfitting gap (train F1 - test F1).

### gridsearch.py
Hyperparameter search over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`. Saves full CV results to `outputs/gridsearch_cv_results.csv`. takes a long time to run.

### visualizations.py
Generates plots for the two best model configurations:
- Class distribution bar chart
- Confusion matrices (raw counts and normalized)
- Feature importance ranking
- Learning curves
- Hyperparameter sensitivity plots

All images saved to `outputs/`.


