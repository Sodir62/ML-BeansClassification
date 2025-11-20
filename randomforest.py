import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Data gotten after running data_normalizer
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()


#Compare several different model
results = []

print("\n" + "="*70)
print("TRAINING MULTIPLE RANDOM FOREST CONFIGURATIONS")
print("="*70)

# Put class weight on balanced because our dataset is imbalanced.

#initial model without any changes
print("\n[1/4] Baseline Model")
rf_baseline = RandomForestClassifier(
    n_estimators=100,      # number of trees
    class_weight='balanced',  
    random_state=42,      
    n_jobs=-1   
)          

rf_baseline.fit(X_train, y_train)
y_train_pred = rf_baseline.predict(X_train)
y_test_pred = rf_baseline.predict(X_test)

train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
overfitting_gap = (train_f1_macro - test_f1_macro) * 100

print(f"Training F1 (macro):   {train_f1_macro:.4f}")
print(f"Test F1 (macro):       {test_f1_macro:.4f}")
print(f"Overfitting Gap:       {overfitting_gap:.2f}%")

results.append({
    'Model': 'Baseline',
    'max_depth': rf_baseline.max_depth,             
    'min_samples_split': rf_baseline.min_samples_split,
    'min_samples_leaf': rf_baseline.min_samples_leaf,
    'Train F1': train_f1_macro,
    'Test F1': test_f1_macro,
    'Gap %': overfitting_gap
})

# Slight parameter tuning, we limit the tree depth and amount of samples per leaf,
# or samples needed to split. 
print("\n[2/4] Light tuning")
rf_light = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,              # limit tree depth
    min_samples_split=5,       # need 5 samples to split
    min_samples_leaf=2,        # need 2 samples in each leaf
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_light.fit(X_train, y_train)
y_train_pred = rf_light.predict(X_train)
y_test_pred = rf_light.predict(X_test)

train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
overfitting_gap = (train_f1_macro - test_f1_macro) * 100

print(f"Training F1 (macro):   {train_f1_macro:.4f}")
print(f"Test F1 (macro):       {test_f1_macro:.4f}")
print(f"Overfitting Gap:       {overfitting_gap:.2f}%")

results.append({
    'Model': 'Light Reg',
    'max_depth': rf_light.max_depth,                
    'min_samples_split': rf_light.min_samples_split,
    'min_samples_leaf': rf_light.min_samples_leaf,
    'Train F1': train_f1_macro,
    'Test F1': test_f1_macro,
    'Gap %': overfitting_gap
})

#heavier tuning
print("\n[3/4] Medium tuning")
rf_medium = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,              
    min_samples_split=10,      
    min_samples_leaf=4,        
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_medium.fit(X_train, y_train)
y_train_pred = rf_medium.predict(X_train)
y_test_pred = rf_medium.predict(X_test)

train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
overfitting_gap = (train_f1_macro - test_f1_macro) * 100

print(f"Training F1 (macro):   {train_f1_macro:.4f}")
print(f"Test F1 (macro):       {test_f1_macro:.4f}")
print(f"Overfitting Gap:       {overfitting_gap:.2f}%")

results.append({
    'Model': 'Medium Reg',
    'max_depth': rf_medium.max_depth,
    'min_samples_split': rf_medium.min_samples_split,
    'min_samples_leaf': rf_medium.min_samples_leaf,
    'Train F1': train_f1_macro,
    'Test F1': test_f1_macro,
    'Gap %': overfitting_gap
})

#heaviest tuning
print("\n[4/4] Heavy tuning")
rf_heavy = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,              
    min_samples_split=20,     
    min_samples_leaf=8,        
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_heavy.fit(X_train, y_train)
y_train_pred = rf_heavy.predict(X_train)
y_test_pred = rf_heavy.predict(X_test)

train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
overfitting_gap = (train_f1_macro - test_f1_macro) * 100

print(f"Training F1 (macro):   {train_f1_macro:.4f}")
print(f"Test F1 (macro):       {test_f1_macro:.4f}")
print(f"Overfitting Gap:       {overfitting_gap:.2f}%")

results.append({
    'Model': 'Heavy Reg',
    'max_depth': rf_heavy.max_depth,
    'min_samples_split': rf_heavy.min_samples_split,
    'min_samples_leaf': rf_heavy.min_samples_leaf,
    'Train F1': train_f1_macro,
    'Test F1': test_f1_macro,
    'Gap %': overfitting_gap
})

#### Comparing all of the models

print("\n" + "="*70)
print("COMPARISON OF ALL MODELS")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best model (lowest gap while maintaining good test performance)
best_idx = results_df['Gap %'].idxmin()
best_model = results_df.iloc[best_idx]

print("\n" + "="*70)
print("BEST MODEL SELECTION")
print("="*70)
print(f"Best Configuration: {best_model['Model']}")
print(f"  max_depth:          {best_model['max_depth']}")
print(f"  min_samples_split:  {best_model['min_samples_split']}")
print(f"  min_samples_leaf:   {best_model['min_samples_leaf']}")
print(f"  Test F1 (macro):    {best_model['Test F1']:.4f}")
print(f"  Overfitting Gap:    {best_model['Gap %']:.2f}%")
print("="*70)