import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import time
import os

# Config param
RANDOM_STATE = 42
N_ESTIMATORS = 100
N_JOBS = -1
CV_FOLDS = 5

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

#load data
CLASS_NAMES = ["Barbunya", "Bombay", "Cali", "Dermason", "Horoz", "Seker", "Sira"]
X_train = pd.read_csv('../Data/X_train.csv')
y_train = pd.read_csv('../Data/y_train.csv').values.ravel()
X_test = pd.read_csv('../Data/X_test.csv')
y_test = pd.read_csv('../Data/y_test.csv').values.ravel()

print("="*70)
print("HYPERPARAMETER OPTIMIZATION WITH GRIDSEARCHCV")
print("="*70)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
print(f"Features:         {X_train.shape[1]}")


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

# Calculate search size
n_combinations = 1
for v in param_grid.values():
    n_combinations *= len(v)
total_fits = n_combinations * CV_FOLDS

print(f"\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")
print(f"\nTotal combinations: {n_combinations}")
print(f"CV folds: {CV_FOLDS}")
print(f"Total model fits: {total_fits}")

# Base estm
base_rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    oob_score=True          # Out-of-bag error estimate
)


# StratifiedKFold preserves class distribution in each fold
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# GRID SEARCH
print("\n" + "="*70)
print("RUNNING GRID SEARCH...")
print("="*70)

grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',         # Optimize for macro F1
    n_jobs=N_JOBS,
    verbose=2,
    return_train_score=True,    # Track training scores for overfitting analysis
    refit=True                  # Refit best model on full training data
)

start_time = time.time()
grid_search.fit(X_train, y_train)
elapsed = time.time() - start_time

print(f"\nSearch completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "="*70)
print("GRID SEARCH RESULTS")
print("="*70)

print("\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV score (F1 macro): {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
print(f"OOB score (best model):   {best_model.oob_score_:.4f}")

# Calculate overfitting gap for all configurations
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['cv_gap'] = (results_df['mean_train_score'] - results_df['mean_test_score']) * 100

# Sort by lowest overfitting gap (primary criterion)
results_df = results_df.sort_values('cv_gap')

# Top 5 configurations by lowest overfitting gap
print("\nTop 5 configurations (sorted by lowest overfitting gap):")
top5 = results_df[['cv_gap', 'mean_test_score', 'std_test_score',
                   'mean_train_score', 'param_n_estimators', 'param_max_depth',
                   'param_min_samples_split', 'param_min_samples_leaf',
                   'param_max_features']].head(5)
print(top5.to_string(index=False))

# Best model is the one with lowest CV gap
best_config = results_df.iloc[0]
cv_gap = best_config['cv_gap']
print(f"\nBest model CV gap (train - test): {cv_gap:.2f}%")

# =============================================================================
# FINAL EVALUATION ON HELD-OUT TEST SET
# =============================================================================
print("\n" + "="*70)
print("FINAL EVALUATION ON TEST SET")
print("="*70)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

final_gap = (train_f1_macro - test_f1_macro) * 100

print(f"Training F1 (macro):    {train_f1_macro:.4f}")
print(f"Test F1 (macro):        {test_f1_macro:.4f}")
print(f"Overfitting Gap:        {final_gap:.2f}%")
print(f"\nTraining F1 (weighted): {train_f1_weighted:.4f}")
print(f"Test F1 (weighted):     {test_f1_weighted:.4f}")

print("\n" + "="*70)
print("CLASSIFICATION REPORT (TEST SET)")
print("="*70)
print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES, digits=4))

# =============================================================================
# SAVE RESULTS
# =============================================================================
results_df.to_csv(os.path.join(OUTPUT_DIR, 'gridsearch_cv_results.csv'), index=False)
print(f"\nFull CV results saved to: {OUTPUT_DIR}/gridsearch_cv_results.csv")
print("="*70)