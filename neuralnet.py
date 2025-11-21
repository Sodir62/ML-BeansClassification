import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Load data from CSVs
X_train = pd.read_csv('X_train.csv').values
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test.csv').values
y_test = pd.read_csv('y_test.csv').values.ravel()

# Variables
epochs=40
batch_size=32
input_dim = X_train.shape[1] #16 features
classes = np.unique(y_train)
num_classes = len(classes) #7 types of beans
name_classes = ["Seker", "Barbunya", "Bombay", "Cali", "Dermosan", "Horoz", "Sira"] # not sure about the order

# Tensorflow seed for consistent results
tf.random.set_seed(1234)

# Define model
model = Sequential([
    tf.keras.Input(shape=(input_dim,)),    #specify input size
    Dense(25, activation='relu',  name='dense_1'),
    Dense(15, activation='relu', name='dense_2'),
    Dense(num_classes, name='logits')
], name='my_model')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'] #tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.F1Score(average='macro')]
)

# Balanced class weights
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Balanced class weights: {class_weight_dict}")

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1 # 0 for silent
)

# Printing evaluation results
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

# Print test loss and accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")
print("")

# Convert logits to class labels with argmax
y_train_logits = model.predict(X_train, verbose=0)
y_test_logits = model.predict(X_test, verbose=0)
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