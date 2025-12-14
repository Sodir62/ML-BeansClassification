import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, ParameterGrid


def build_model(input_dim, num_classes, dense1=25, dense2=15, lr=0.001, dropout=0.0, l2=0.0, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    layers = [tf.keras.Input(shape=(input_dim,))]
    # first dense
    if reg is not None:
        layers.append(Dense(dense1, activation='relu', kernel_regularizer=reg))
    else:
        layers.append(Dense(dense1, activation='relu'))
    if dropout and dropout > 0:
        layers.append(Dropout(dropout))
    # second dense
    if reg is not None:
        layers.append(Dense(dense2, activation='relu', kernel_regularizer=reg))
    else:
        layers.append(Dense(dense2, activation='relu'))
    if dropout and dropout > 0:
        layers.append(Dropout(dropout))
    layers.append(Dense(num_classes))

    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def run_grid_search(X, y, param_grid, cv_splits=3, random_state=1234, results_path=None, metric='f1_weighted', early_stopping=True, patience=5):
    input_dim = X.shape[1]
    classes = np.unique(y)
    num_classes = len(classes)

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results = []
    total_combinations = sum(1 for _ in ParameterGrid(param_grid))
    print(f"Running grid search: {total_combinations} combinations, {cv_splits}-fold CV")

    combo_idx = 0
    for params in ParameterGrid(param_grid):
        combo_idx += 1
        print(f"\nCombo {combo_idx}/{total_combinations}: {params}")
        val_accs = []
        val_f1_w = []
        val_f1_macro = []
        t0 = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_classes = np.unique(y_tr)
            class_weights = compute_class_weight(class_weight='balanced', classes=fold_classes, y=y_tr)
            class_weight_dict = dict(zip(fold_classes, class_weights))

            tf.keras.backend.clear_session()
            model = build_model(
                input_dim,
                num_classes,
                dense1=params.get('dense1', 25),
                dense2=params.get('dense2', 15),
                lr=params.get('learning_rate', 0.001),
                dropout=params.get('dropout', 0.0),
                l2=params.get('l2', 0.0),
                seed=random_state + fold
            )

            callbacks = []
            if early_stopping:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0))

            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                class_weight=class_weight_dict,
                epochs=params.get('epochs', 20),
                batch_size=params.get('batch_size', 32),
                verbose=0,
                callbacks=callbacks
            )

            y_val_logits = model.predict(X_val, verbose=0)
            y_val_pred = np.argmax(y_val_logits, axis=1)
            val_accs.append(accuracy_score(y_val, y_val_pred))
            val_f1_w.append(f1_score(y_val, y_val_pred, average='weighted'))
            val_f1_macro.append(f1_score(y_val, y_val_pred, average='macro'))

        elapsed = time.time() - t0
        # compute F1 score
        mean_f1_w = np.mean(val_f1_w)
        std_f1_w = np.std(val_f1_w)
        mean_f1_macro = np.mean(val_f1_macro)
        std_f1_macro = np.std(val_f1_macro)

        if metric == 'f1_macro':
            mean_val_f1 = mean_f1_macro
            std_val_f1 = std_f1_macro
        else:
            mean_val_f1 = mean_f1_w
            std_val_f1 = std_f1_w

        results.append({**params,
                        'mean_val_acc': np.mean(val_accs),
                        'std_val_acc': np.std(val_accs),
                        'mean_val_f1_weighted': mean_f1_w,
                        'std_val_f1_weighted': std_f1_w,
                        'mean_val_f1_macro': mean_f1_macro,
                        'std_val_f1_macro': std_f1_macro,
                'mean_val_f1': mean_val_f1,
                'std_val_f1': std_val_f1,
                'early_stopping': bool(early_stopping),
                'early_stopping_patience': int(patience),
                        'cv_splits': cv_splits,
                        'time_sec': elapsed})

        print(f"  mean val acc: {results[-1]['mean_val_acc']:.4f} | mean val f1 ({metric}): {results[-1]['mean_val_f1']:.4f} | time: {elapsed:.1f}s")

    df = pd.DataFrame(results)
    if results_path is not None:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        df.to_csv(results_path, index=False)
        print(f"\nSaved grid search results to: {results_path}")

    return df


if __name__ == '__main__':
    # Load data
    X_train = pd.read_csv('../Data/X_train.csv').values
    y_train = pd.read_csv('../Data/y_train.csv').values.ravel()

    # parameter grid (includes regularization/dropout)
    param_grid = {
        'dense1': [25, 40, 50],
        'dense2': [8, 16, 32],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [16, 32],
        'epochs': [20, 40],
        'dropout': [0.0, 0.4],
        'l2': [0.0, 1e-4]
    }

    # Run grid search (reduce cv_splits or shrink grid for faster runs)
    out_path = os.path.join(os.path.dirname(__file__), 'outputs', 'gridsearch_results_nn.csv')
    df_results = run_grid_search(X_train, y_train, param_grid, cv_splits=3, random_state=1234, results_path=out_path)
    print('\nGrid search complete. Top 5 configurations (by F1 metric):')
    print(df_results.sort_values('mean_val_f1', ascending=False).head(5).to_string(index=False))
