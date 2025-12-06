# tree_visualization.py - Visualize a single decision tree from the best RF model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import os

# Configuration
RANDOM_STATE = 42
N_JOBS = -1
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names
CLASS_NAMES = ["Barbunya", "Bombay", "Cali", "Dermason", "Horoz", "Seker", "Sira"]

# Best model configuration (matches README)
# max_depth=10, max_features=None, min_samples_leaf=8, min_samples_split=20, n_estimators=100
BEST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 8,
    'max_features': None,
}

# Load data
print("Loading data...")
X_train = pd.read_csv('../Data/X_train.csv')
y_train = pd.read_csv('../Data/y_train.csv').values.ravel()

feature_names = X_train.columns.tolist()

print(f"Training samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")

# Train the best model
print("\nTraining best Random Forest model...")
print(f"Configuration: {BEST_CONFIG}")

rf = RandomForestClassifier(
    n_estimators=BEST_CONFIG['n_estimators'],
    max_depth=BEST_CONFIG['max_depth'],
    min_samples_split=BEST_CONFIG['min_samples_split'],
    min_samples_leaf=BEST_CONFIG['min_samples_leaf'],
    max_features=BEST_CONFIG['max_features'],
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
rf.fit(X_train, y_train)

print(f"Model trained with {len(rf.estimators_)} trees")

# Select the first tree to visualize (tree index 0)
tree_to_plot = rf.estimators_[0]

# Create the visualization
print("\nGenerating tree visualization...")

# Full tree visualization (may be large)
fig, ax = plt.subplots(figsize=(40, 20))
plot_tree(
    tree_to_plot,
    feature_names=feature_names,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    proportion=True,
    impurity=True
)
ax.set_title('Decision Tree from Best Random Forest Model\n(Tree #1 of 100, max_depth=10)',
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'decision_tree_full.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/decision_tree_full.png")

# Create a smaller, more readable visualization with limited depth
fig, ax = plt.subplots(figsize=(25, 15))
plot_tree(
    tree_to_plot,
    feature_names=feature_names,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax,
    max_depth=4,  # Show only first 4 levels for readability
    proportion=True,
    impurity=True
)
ax.set_title('Decision Tree from Best Random Forest (Depth Limited to 4 levels)\n(Tree #1 of 100)',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'decision_tree_depth4.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/decision_tree_depth4.png")

# Print tree statistics
print("\n" + "="*50)
print("TREE STATISTICS")
print("="*50)
print(f"Tree depth: {tree_to_plot.get_depth()}")
print(f"Number of leaves: {tree_to_plot.get_n_leaves()}")
print(f"Number of features used: {tree_to_plot.n_features_in_}")

# Feature importance from this single tree
single_tree_importance = tree_to_plot.feature_importances_
top_features_idx = np.argsort(single_tree_importance)[::-1][:5]
print("\nTop 5 features in this tree:")
for idx in top_features_idx:
    print(f"  {feature_names[idx]}: {single_tree_importance[idx]:.4f}")

print("\n" + "="*50)
print("VISUALIZATION COMPLETE")
print("="*50)
print(f"Generated files in {OUTPUT_DIR}/:")
print("  - decision_tree_full.png (complete tree)")
print("  - decision_tree_depth4.png (first 4 levels, more readable)")
