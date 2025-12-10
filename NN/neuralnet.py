import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

# Load data from CSVs
X_train = pd.read_csv('../Data/X_train.csv').values
y_train = pd.read_csv('../Data/y_train.csv').values.ravel()
X_test = pd.read_csv('../Data/X_test.csv').values
y_test = pd.read_csv('../Data/y_test.csv').values.ravel()

# Variables
epochs=40
batch_size=32
input_dim = X_train.shape[1] #16 features
classes = np.unique(y_train)
num_classes = len(classes) # 7 types of beans
name_classes = ["Barbunya", "Bombay", "Cali", "Dermason", "Horoz", "Seker", "Sira"]

# K-Fold CV settings
# seed for consistent results
K_FOLDS = 5
tf.random.set_seed(1234)
np.random.seed(1234)


def build_model(input_dim, num_classes, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(25, activation='relu', name='dense_1'),
        Dense(15, activation='relu', name='dense_2'),
        Dense(num_classes, name='logits')
    ], name='my_model')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# Stratified K-Fold cross validation on the training set
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=1234)

val_losses = []
val_accs = []
val_f1_macros = []
val_f1_weighteds = []

print("\n" + "="*70)
print(f"Starting {K_FOLDS}-fold cross-validation on training set")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n--- Fold {fold}/{K_FOLDS} ---")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Balanced class weights for this fold
    fold_classes = np.unique(y_tr)
    class_weights = compute_class_weight(class_weight='balanced', classes=fold_classes, y=y_tr)
    class_weight_dict = dict(zip(fold_classes, class_weights))
    print(f"  class weights: {class_weight_dict}")

    # build and train model for this fold
    tf.keras.backend.clear_session()
    model = build_model(input_dim, num_classes, seed=1234 + fold)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # evaluate on validation fold
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    y_val_logits = model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_logits, axis=1)
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
    val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')

    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1_macros.append(val_f1_macro)
    val_f1_weighteds.append(val_f1_weighted)

    print(f"  Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}")
    print(f"  Val F1 macro: {val_f1_macro:.4f}  Val F1 weighted: {val_f1_weighted:.4f}")

# Summary of cross-validation
print("\n" + "="*60)
print(f"{K_FOLDS}-Fold CV results (mean ± std):")
print(f"  Loss:    {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
print(f"  Acc:     {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
print(f"  F1 macro:{np.mean(val_f1_macros):.4f} ± {np.std(val_f1_macros):.4f}")
print(f"  F1 wtd:  {np.mean(val_f1_weighteds):.4f} ± {np.std(val_f1_weighteds):.4f}")
print("="*60)

# Train final model on the full training set and evaluate on the test set (same as before)
print("\nTraining final model on full training set and evaluating on test set...")
tf.keras.backend.clear_session()
final_model = build_model(input_dim, num_classes, seed=1234)

# Balanced class weights for full training set
class_weights_full = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_full_dict = dict(zip(classes, class_weights_full))
print(f"Balanced class weights (full train): {class_weight_full_dict}")

final_history = final_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    class_weight=class_weight_full_dict,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1 # 0 for silent
)

# Printing evaluation results
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

# Print test loss and accuracy
test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")
print("")

# Convert logits to class labels with argmax
y_train_logits = final_model.predict(X_train, verbose=0)
y_test_logits = final_model.predict(X_test, verbose=0)
y_train_pred = np.argmax(y_train_logits, axis=1)
y_test_pred = np.argmax(y_test_logits, axis=1)

# Calculate F1 scores (macro and weighted)
train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

# Calculate overfitting gaps
gap_f1_macro = (train_f1_macro - test_f1_macro) * 100
gap_f1_weighted = (train_f1_weighted - test_f1_weighted) * 100

# Print F1 scores and gaps
print("F1 macro:")
print(f"  Training:                       {train_f1_macro:.4f}")
print(f"  Test:                           {test_f1_macro:.4f}")
print(f"  Overfitting gap (train - test): {gap_f1_macro:.4f}%")
print("F1 weighted:")
print(f"  Training:                       {train_f1_weighted:.4f}")
print(f"  Test:                           {test_f1_weighted:.4f}")
print(f"  Overfitting gap (train - test): {gap_f1_weighted:.4f}%")

# Print classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_test_pred, target_names=name_classes, digits=4))
print("="*70)