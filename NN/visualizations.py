import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.inspection import permutation_importance


def _ensure_output_dir(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_fig(fig, out_path: Path, dpi: int = 200):
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, classes, output_path, normalize: bool = False, cmap='Blues'):
    """Plot and save a confusion matrix heatmap.

    Args:
      y_true: array-like of true labels
      y_pred: array-like of predicted labels
      classes: list of class names (in label order)
      output_path: Path or str to save the figure (PNG)
      normalize: if True divide rows by their sums
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    # annotate cells
    if normalize:
        fmt_norm = True
    else:
        fmt_norm = False
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if fmt_norm:
                txt = f"{val:.2f}"
            else:
                try:
                    txt = f"{int(val)}"
                except Exception:
                    txt = f"{val}"
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=color)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(('Normalized ' if normalize else '')+ 'Confusion Matrix')
    out = Path(output_path)
    _ensure_output_dir(out.parent)
    _save_fig(fig, out)


def plot_feature_importance(model, feature_names, output_path, X=None, y=None, n_repeats=10, random_state=0, scoring=None):
    """Plot and save feature importances.

    If model has `feature_importances_` attribute it will be used. Otherwise a
    permutation importance will be computed using `X` and `y`.
    """
    out = Path(output_path)
    _ensure_output_dir(out.parent)

    if hasattr(model, 'feature_importances_'):
        importances = np.array(model.feature_importances_)
    else:
        if X is None or y is None:
            raise ValueError('X and y must be provided for permutation importance')
        # Try sklearn's permutation_importance first. If it fails because the
        # estimator doesn't implement `fit` (e.g., a custom predictor), fall back
        # to a manual permutation importance using `model.predict`.
        try:
            if scoring is not None:
                r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring=scoring)
            else:
                r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
            importances = r.importances_mean
        except Exception as e:
            # Fall back to manual permutation importance
            if not hasattr(model, 'predict'):
                raise ValueError("Could not compute permutation importance: the estimator doesn't implement 'fit' and has no 'predict' method") from e

            X_arr = np.asarray(X)
            y_arr = np.asarray(y)
            n_samples, n_features = X_arr.shape

            # scoring: accept a callable scoring(y_true, y_pred) or default to accuracy
            if callable(scoring):
                score_fn = scoring
            else:
                # common string options
                if scoring == 'f1':
                    def score_fn(y_true, y_pred):
                        return f1_score(y_true, y_pred, average='macro')
                else:
                    score_fn = accuracy_score

            # get baseline score
            preds = model.predict(X_arr)
            if hasattr(preds, 'shape') and preds.ndim > 1 and preds.shape[1] > 1:
                y_pred = preds.argmax(axis=1)
            else:
                y_pred = np.asarray(preds).ravel()
            baseline = score_fn(y_arr, y_pred)

            rng = np.random.RandomState(random_state)
            importances = np.zeros(n_features)
            for j in range(n_features):
                decreases = []
                for _ in range(n_repeats):
                    idx = rng.permutation(n_samples)
                    X_perm = X_arr.copy()
                    X_perm[:, j] = X_arr[idx, j]
                    preds_perm = model.predict(X_perm)
                    if hasattr(preds_perm, 'shape') and preds_perm.ndim > 1 and preds_perm.shape[1] > 1:
                        y_perm = preds_perm.argmax(axis=1)
                    else:
                        y_perm = np.asarray(preds_perm).ravel()
                    score = score_fn(y_arr, y_perm)
                    decreases.append(baseline - score)
                importances[j] = float(np.mean(decreases))

    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(6, max(4, len(feature_names) * 0.25)))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_vals, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance')
    _save_fig(fig, out)


def plot_learning_curves(history, output_path):
    """Plot training & validation loss and accuracy from a Keras History or similar dict.

    `history` may be a Keras History object or a dict with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
    """
    out = Path(output_path)
    _ensure_output_dir(out.parent)

    if hasattr(history, 'history'):
        h = history.history
    elif isinstance(history, dict):
        h = history
    else:
        raise ValueError('history must be a keras History or a dict')

    epochs = range(1, len(h.get('loss', [])) + 1)

    # Save loss figure separately
    fig_loss, ax_loss = plt.subplots(figsize=(6, 6))
    ax_loss.plot(epochs, h.get('loss', []), label='train')
    if 'val_loss' in h:
        ax_loss.plot(epochs, h.get('val_loss', []), label='val')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.legend()
    out_loss = out.with_name(out.stem + '_loss' + out.suffix)
    _save_fig(fig_loss, out_loss)

    # Save accuracy figure separately
    acc_key = 'accuracy' if 'accuracy' in h else 'acc'
    val_acc_key = 'val_' + acc_key
    fig_acc, ax_acc = plt.subplots(figsize=(6, 6))
    ax_acc.plot(epochs, h.get(acc_key, []), label='train')
    if val_acc_key in h:
        ax_acc.plot(epochs, h.get(val_acc_key, []), label='val')
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.legend()
    out_acc = out.with_name(out.stem + '_accuracy' + out.suffix)
    _save_fig(fig_acc, out_acc)


def plot_hyperparameter_sensitivity(results_df, param_name, metric_name, output_path, log_x=False):
    """Plot how a metric changes with a hyperparameter.

    `results_df` is a pandas DataFrame with columns including `param_name` and `metric_name`.
    """
    out = Path(output_path)
    _ensure_output_dir(out.parent)

    df = pd.DataFrame(results_df)
    fig, ax = plt.subplots(figsize=(6, 6))
    x = df[param_name]
    y = df[metric_name]
    if pd.api.types.is_numeric_dtype(x):
        ax.plot(x, y, marker='o', linestyle='-')
        if log_x:
            ax.set_xscale('log')
    else:
        ax.plot(range(len(x)), y, marker='o', linestyle='-')
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x)
    ax.set_title(f'{metric_name} vs {param_name}')
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    _save_fig(fig, out)


def plot_class_distribution(datasets, class_names, output_path, normalize: bool = False, show_cv_std: bool = True):
    """Compare class distributions across multiple datasets.

    Args:
      datasets: dict mapping split name -> array-like of labels, or split name -> list of array-like (CV folds).
      class_names: ordered list of class names corresponding to label indices.
      output_path: Path or str to save the figure (PNG).
      normalize: if True, plot proportions instead of raw counts.
      show_cv_std: if True and CV folds provided, show std error bars for CV folds.
    """
    out = Path(output_path)
    _ensure_output_dir(out.parent)

    # Prepare counts for each dataset
    labels = list(class_names)
    n_classes = len(labels)
    keys = list(datasets.keys())

    counts_list = []
    errors = []
    for k in keys:
        v = datasets[k]
        if isinstance(v, (list, tuple)):
            # CV folds: compute counts per fold
            fold_counts = []
            for fold in v:
                s = pd.Series(fold).map(lambda val: labels[int(val)] if isinstance(val, (int, np.integer)) else val)
                cnt = s.value_counts().reindex(labels).fillna(0).values.astype(float)
                if normalize:
                    denom = cnt.sum()
                    cnt = cnt / denom if denom else cnt
                fold_counts.append(cnt)
            arr = np.vstack(fold_counts)  # shape (n_folds, n_classes)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            counts_list.append(mean)
            errors.append(std if show_cv_std else np.zeros_like(std))
        else:
            s = pd.Series(v).map(lambda val: labels[int(val)] if isinstance(val, (int, np.integer)) else val)
            cnt = s.value_counts().reindex(labels).fillna(0).values.astype(float)
            if normalize:
                denom = cnt.sum()
                cnt = cnt / denom if denom else cnt
            counts_list.append(cnt)
            errors.append(np.zeros(n_classes))

    counts_arr = np.vstack(counts_list)  # shape (n_datasets, n_classes)

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.6), max(6, n_classes * 0.6)))
    indices = np.arange(n_classes)
    n_datasets = len(keys)
    width = 0.8 / max(1, n_datasets)

    for i, (k, row) in enumerate(zip(keys, counts_arr)):
        pos = indices - 0.4 + (i + 0.5) * width
        ax.bar(pos, row, width=width, label=k, yerr=errors[i] if errors[i].any() else None, capsize=3)

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Proportion' if normalize else 'Count')
    ax.set_title('Class Distribution Comparison')
    ax.legend()
    _save_fig(fig, out)


def plot_per_class_f1(y_true, y_pred, class_names, output_path):
    out = Path(output_path)
    _ensure_output_dir(out.parent)
    f1s = f1_score(y_true, y_pred, average=None)
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.arange(len(class_names))
    ax.bar(x, f1s)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('F1 score')
    ax.set_ylim(0, 1)
    ax.set_title('Per-class F1 scores')
    _save_fig(fig, out)


if __name__ == '__main__':
    # quick smoke example if run directly (no external data required)
    outdir = Path(__file__).parent / 'outputs'
    _ensure_output_dir(outdir)
    # small synthetic example
    y_true = [0, 1, 2, 1, 0, 2, 1]
    y_pred = [0, 2, 2, 1, 0, 1, 1]
    classes = ['A', 'B', 'C']
    plot_confusion_matrix(y_true, y_pred, classes, outdir / 'cm.png')
    plot_confusion_matrix(y_true, y_pred, classes, outdir / 'cm_norm.png', normalize=True)
    plot_per_class_f1(y_true, y_pred, classes, outdir / 'per_class_f1.png')