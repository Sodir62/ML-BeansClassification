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


# Put class weight on balanced because our dataset is imbalanced.

rf_model = RandomForestClassifier(
    n_estimators=100,      # number of trees
    class_weight='balanced',  
    random_state=42,      
    n_jobs=-1   
)          # uses all cpu cores

rf_model.fit(X_train, y_train)

# Make predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)


print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)


# F1-scores (because we have imbalanced dataset)
train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

print("\nTraining Set:")
print(f"  F1-score (macro):    {train_f1_macro:.4f}")
print(f"  F1-score (weighted): {train_f1_weighted:.4f}")

print("\nTest Set:")
print(f"  F1-score (macro):    {test_f1_macro:.4f}")
print(f"  F1-score (weighted): {test_f1_weighted:.4f}")

#function to check if we overfitted
print(f"Overfitting Gap (F1-macro): {(train_f1_macro - test_f1_macro)*100:.2f}%")
