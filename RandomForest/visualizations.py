# visualisations were helped make with claude code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import os


RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 5
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASS_NAMES = ["Barbunya", "Bombay", "Cali", "Dermason", "Horoz", "Seker", "Sira"]


# Config 1: Lowest overfitting gap (from grid search + manual testing)
CONFIG_LOW_GAP = {
    'name': 'Low Gap (Heavy Reg)',
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 8,
    'max_features': 'sqrt',  # Match randomforest.py default
}

# Config 2: Best CV F1 with acceptable gap
CONFIG_BEST_F1 = {
    'name': 'Best F1',
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',  # Match randomforest.py default
}

print("Loading data...")
X_train = pd.read_csv('../Data/X_train.csv')
y_train = pd.read_csv('../Data/y_train.csv').values.ravel()
X_test = pd.read_csv('../Data/X_test.csv')
y_test = pd.read_csv('../Data/y_test.csv').values.ravel()

feature_names = X_train.columns.tolist()

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
print(f"Features:         {X_train.shape[1]}")

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

models = {}
for config in [CONFIG_LOW_GAP, CONFIG_BEST_F1]:
    print(f"\nTraining: {config['name']}")
    rf = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    gap = (train_f1 - test_f1) * 100

    print(f"  Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Gap: {gap:.2f}%")

    models[config['name']] = {
        'model': rf,
        'config': config,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'gap': gap,
        'y_test_pred': y_test_pred
    }

# Use the "Low Gap (Heavy Reg)" model as primary for most visualizations
primary_model = models['Low Gap (Heavy Reg)']['model']
primary_pred = models['Low Gap (Heavy Reg)']['y_test_pred']


print("\n" + "="*70)
print("1. GENERATING CLASS DISTRIBUTION CHART")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))

# Count samples per class
train_counts = pd.Series(y_train).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

x = np.arange(len(CLASS_NAMES))
width = 0.35

bars1 = ax.bar(x - width/2, train_counts.values, width, label='Train', color='steelblue')
bars2 = ax.bar(x + width/2, test_counts.values, width, label='Test', color='darkorange')

ax.set_xlabel('Bean Class')
ax.set_ylabel('Number of Samples')
ax.set_title('Class Distribution in Train and Test Sets', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.legend()

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/class_distribution.png")



# CONFUSION MATRICES 
print("\n" + "="*70)
print("3. GENERATING CONFUSION MATRICES")
print("="*70)

# Raw counts
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, primary_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
ax.set_title('Confusion Matrix - Raw Counts (Heavy Reg Model)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_counts.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/confusion_matrix_counts.png")

# Normalized (recall per class)
fig, ax = plt.subplots(figsize=(10, 8))
cm_normalized = confusion_matrix(y_test, primary_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', values_format='.2f', colorbar=True)
ax.set_title('Normalized Confusion Matrix - Recall (Heavy Reg Model)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/confusion_matrix_normalized.png")

# FEATURE IMPORTANCE
print("\n" + "="*70)
print("4. GENERATING FEATURE IMPORTANCE PLOT")
print("="*70)

importances = primary_model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_names)))

# Horizontal bar chart
y_pos = np.arange(len(feature_names))
ax.barh(y_pos, importances[indices][::-1], color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels([feature_names[i] for i in indices][::-1])
ax.set_xlabel('Feature Importance (Gini Impurity Decrease)')
ax.set_title('Random Forest Feature Importances (Heavy Reg Model)', fontsize=14, fontweight='bold')

# Value labels
for i, v in enumerate(importances[indices][::-1]):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/feature_importance.png")

# Print feature ranking
print("\nFeature Importance Ranking:")
print("-" * 40)
for rank, idx in enumerate(indices, 1):
    print(f"{rank:2d}. {feature_names[idx]:<20} {importances[idx]:.4f}")


# LEARNING CURVES
print("\n" + "="*70)
print("5. GENERATING LEARNING CURVES")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for idx, (name, data) in enumerate(models.items()):
    config = data['config']

    rf = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    train_sizes, train_scores, val_scores = learning_curve(
        rf, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring='f1_macro',
        n_jobs=N_JOBS
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax = axes[idx]
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training F1')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation F1')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('F1 Score (Macro)')
    ax.set_title(f'Learning Curve: {name}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, alpha=0.3)

    # Add final gap annotation
    final_gap = (train_mean[-1] - val_mean[-1]) * 100
    ax.annotate(f'Final Gap: {final_gap:.2f}%',
                xy=(train_sizes[-1], val_mean[-1]),
                xytext=(-80, -30), textcoords='offset points',
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/learning_curves.png")


# HYPERPARAMETER SENSITIVITY PLOTS
print("\n" + "="*70)
print("6. GENERATING HYPERPARAMETER SENSITIVITY PLOTS")
print("="*70)

# Load grid search results
gs_results = pd.read_csv(os.path.join(OUTPUT_DIR, 'gridsearch_cv_results.csv'))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: max_depth effect
ax = axes[0, 0]
depth_data = gs_results.groupby('param_max_depth').agg({
    'mean_test_score': 'mean',
    'cv_gap': 'mean'
}).reset_index()
depth_data = depth_data[depth_data['param_max_depth'].notna()]

ax2 = ax.twinx()
line1 = ax.plot(depth_data['param_max_depth'], depth_data['mean_test_score'], 'o-', color='blue', label='CV F1')
line2 = ax2.plot(depth_data['param_max_depth'], depth_data['cv_gap'], 's--', color='red', label='CV Gap %')
ax.set_xlabel('max_depth')
ax.set_ylabel('CV F1 (Macro)', color='blue')
ax2.set_ylabel('CV Gap %', color='red')
ax.set_title('Effect of max_depth', fontweight='bold')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

# Plot 2: min_samples_leaf effect
ax = axes[0, 1]
leaf_data = gs_results.groupby('param_min_samples_leaf').agg({
    'mean_test_score': 'mean',
    'cv_gap': 'mean'
}).reset_index()

ax2 = ax.twinx()
line1 = ax.plot(leaf_data['param_min_samples_leaf'], leaf_data['mean_test_score'], 'o-', color='blue', label='CV F1')
line2 = ax2.plot(leaf_data['param_min_samples_leaf'], leaf_data['cv_gap'], 's--', color='red', label='CV Gap %')
ax.set_xlabel('min_samples_leaf')
ax.set_ylabel('CV F1 (Macro)', color='blue')
ax2.set_ylabel('CV Gap %', color='red')
ax.set_title('Effect of min_samples_leaf', fontweight='bold')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

# Plot 3: min_samples_split effect
ax = axes[1, 0]
split_data = gs_results.groupby('param_min_samples_split').agg({
    'mean_test_score': 'mean',
    'cv_gap': 'mean'
}).reset_index()

ax2 = ax.twinx()
line1 = ax.plot(split_data['param_min_samples_split'], split_data['mean_test_score'], 'o-', color='blue', label='CV F1')
line2 = ax2.plot(split_data['param_min_samples_split'], split_data['cv_gap'], 's--', color='red', label='CV Gap %')
ax.set_xlabel('min_samples_split')
ax.set_ylabel('CV F1 (Macro)', color='blue')
ax2.set_ylabel('CV Gap %', color='red')
ax.set_title('Effect of min_samples_split', fontweight='bold')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

# Plot 4: n_estimators effect
ax = axes[1, 1]
nest_data = gs_results.groupby('param_n_estimators').agg({
    'mean_test_score': 'mean',
    'cv_gap': 'mean'
}).reset_index()

ax2 = ax.twinx()
line1 = ax.plot(nest_data['param_n_estimators'], nest_data['mean_test_score'], 'o-', color='blue', label='CV F1')
line2 = ax2.plot(nest_data['param_n_estimators'], nest_data['cv_gap'], 's--', color='red', label='CV Gap %')
ax.set_xlabel('n_estimators')
ax.set_ylabel('CV F1 (Macro)', color='blue')
ax2.set_ylabel('CV Gap %', color='red')
ax.set_title('Effect of n_estimators', fontweight='bold')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'hyperparameter_sensitivity.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/hyperparameter_sensitivity.png")

# PER-CLASS F1 SCORES
print("\n" + "="*70)
print("7. GENERATING PER-CLASS F1 PLOT")
print("="*70)

from sklearn.metrics import f1_score

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(CLASS_NAMES))
width = 0.35

# Calculate per-class F1 for both models
for idx, (name, data) in enumerate(models.items()):
    y_pred = data['y_test_pred']
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    offset = width * (idx - 0.5)
    bars = ax.bar(x + offset, per_class_f1, width, label=name)

    # Add value labels
    for bar, val in zip(bars, per_class_f1):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Add macro F1 lines
colors = ['steelblue', 'darkorange']
for idx, (name, data) in enumerate(models.items()):
    ax.axhline(y=data['test_f1'], color=colors[idx], linestyle='--', alpha=0.7,
               label=f"Macro F1 ({name}): {data['test_f1']:.4f}")

ax.set_xlabel('Bean Class')
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.legend(loc='lower right')
ax.set_ylim(0.85, 1.02)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_f1.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/per_class_f1.png")


# SUMMARY
print("\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)
print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
print("\nFiles generated:")
print("  class_distribution.png       - Train/test class sample counts")
print("  confusion_matrix_counts.png  - Raw confusion matrix")
print("  confusion_matrix_normalized.png - Normalized confusion matrix")
print("  feature_importance.png       - Feature importance ranking")
print("  learning_curves.png          - Train vs validation learning curves")
print("  hyperparameter_sensitivity.png - Effect of each hyperparameter")
print("  per_class_f1.png             - Per-class F1 score comparison")

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
for name, data in models.items():
    print(f"\n{name}:")
    print(f"  Config: depth={data['config']['max_depth']}, leaf={data['config']['min_samples_leaf']}, split={data['config']['min_samples_split']}")
    print(f"  Train F1: {data['train_f1']:.4f}")
    print(f"  Test F1:  {data['test_f1']:.4f}")
    print(f"  Gap:      {data['gap']:.2f}%")
print("="*70)
