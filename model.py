import time
import random
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)


import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# ==========================================
# Phase 1: Model Training & Serialization
# ==========================================
def train_and_save_model(train_csv_path):
    print("[SYSTEM] Loading Real Data & Training AI Model...")
    df = pd.read_csv(train_csv_path)

    # 1. Separate Features and Target
    X = df.drop(columns=['class'])

    # Encode target: normal = 0, anomaly = 1
    y = df['class'].apply(lambda x: 0 if x == 'normal' else 1).values

    # 2. Encode Categorical Features
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        unique_vals = list(X[col].unique()) + ['<unknown>']
        le.fit(unique_vals)
        X[col] = le.transform(X[col])
        encoders[col] = le

    # 3. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train Model
    print("[SYSTEM] Training Random Forest on records. Please wait...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 6. Save Artifacts
    joblib.dump(rf_model, 'dos_rf_model.pkl')
    joblib.dump(scaler, 'dos_scaler.pkl')
    joblib.dump(encoders, 'dos_encoders.pkl')
    print("[SYSTEM] Model, Scaler, and Encoders saved to disk successfully.\n")

    # 7. Generate ALL Graphs
    feature_names = df.drop(columns=['class']).columns.tolist()
    plot_all_model_graphs(rf_model, X_train, X_val, y_train, y_val, feature_names)

    return rf_model, scaler, encoders


# ==========================================
# Phase 1.5: Comprehensive Model Graphs
# ==========================================

def plot_all_model_graphs(model, X_train, X_val, y_train, y_val, feature_names):
    """Generates and saves all model evaluation graphs."""

    print("[GRAPHS] Generating all model evaluation plots...")
    sns.set_theme(style="darkgrid", palette="muted")

    y_pred      = model.predict(X_val)
    y_prob      = model.predict_proba(X_val)[:, 1]

    # ── 1. Confusion Matrix ───────────────────────────────────────────────────
    plot_confusion_matrix(model, X_val, y_val)

    # ── 2. ROC Curve ──────────────────────────────────────────────────────────
    plot_roc_curve(y_val, y_prob)

    # ── 3. Precision-Recall Curve ─────────────────────────────────────────────
    plot_precision_recall_curve(y_val, y_prob)

    # ── 4. Feature Importance ─────────────────────────────────────────────────
    plot_feature_importance(model, feature_names)

    # ── 5. Learning Curve ─────────────────────────────────────────────────────
    plot_learning_curve(model, X_train, y_train)

    # ── 6. Class Distribution ─────────────────────────────────────────────────
    plot_class_distribution(y_train, y_val)

    # ── 7. Prediction Confidence Distribution ─────────────────────────────────
    plot_confidence_distribution(y_val, y_prob)

    # ── 8. Per-Tree Accuracy (OOB / estimator variation) ─────────────────────
    plot_estimator_accuracy(model, X_val, y_val)

    # ── 9. Cross-Validation Scores ────────────────────────────────────────────
    plot_cross_validation(model, X_train, y_train)

    # ── 10. Threshold vs Precision/Recall/F1 ──────────────────────────────────
    plot_threshold_analysis(y_val, y_prob)

    # ── 11. Cumulative Gains Curve ────────────────────────────────────────────
    plot_cumulative_gains(y_val, y_prob)

    # ── 12. Dashboard (summary of key plots in one figure) ────────────────────
    plot_dashboard(model, X_val, y_val, y_pred, y_prob, feature_names, y_train)

    print("[GRAPHS] All plots saved successfully!\n")


# ─── Individual Plot Functions ────────────────────────────────────────────────
def plot_confusion_matrix(model, X_val, y_val):
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec  = recall_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred)
    spec = tn / (tn + fp)

    fig = plt.figure(figsize=(13, 6), facecolor='#0f1117')
    fig.patch.set_facecolor('#0f1117')

    # ── Layout: matrix (left) + metrics (right) ───────────────────────────────
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.08)
    ax_mat = fig.add_subplot(gs[0])
    ax_met = fig.add_subplot(gs[1])

    for ax in [ax_mat, ax_met]:
        ax.set_facecolor('#0f1117')
        ax.axis('off')

    # ── Cell colors ───────────────────────────────────────────────────────────
    COLORS = {
        'TN': '#0c447c',   # deep blue
        'FP': '#F0997B',   # coral
        'FN': '#F7C1C1',   # pale red
        'TP': '#085041',   # deep teal
    }
    TEXT = {
        'TN': '#b5d4f4',
        'FP': '#4A1B0C',
        'FN': '#501313',
        'TP': '#9FE1CB',
    }
    cells = [
        (tn, tn/total, 'True Negative',  'TN', 0, 0),
        (fp, fp/total, 'False Positive', 'FP', 0, 1),
        (fn, fn/total, 'False Negative', 'FN', 1, 0),
        (tp, tp/total, 'True Positive',  'TP', 1, 1),
    ]

    cell_w, cell_h, gap = 0.42, 0.40, 0.03
    x_start, y_start   = 0.08, 0.12

    # Axis labels
    for j, lbl in enumerate(['Predicted: Normal', 'Predicted: Attack']):
        ax_mat.text(x_start + j*(cell_w+gap) + cell_w/2, 0.96,
                    lbl, ha='center', va='top', fontsize=10,
                    color='#888', transform=ax_mat.transAxes)

    for i, lbl in enumerate(['Actual: Normal', 'Actual: Attack']):
        ax_mat.text(0.03,
                    y_start + (1-i)*(cell_h+gap) - cell_h/2 + 0.01,
                    lbl, ha='center', va='center', fontsize=10,
                    rotation=90, color='#888', transform=ax_mat.transAxes)

    # Draw cells
    for count, pct, label, key, row, col in cells:
        x = x_start + col*(cell_w + gap)
        y = y_start + (1-row)*(cell_h + gap)
        rect = mpatches.FancyBboxPatch(
            (x, y), cell_w, cell_h,
            boxstyle="round,pad=0.01",
            facecolor=COLORS[key], edgecolor='none',
            transform=ax_mat.transAxes, clip_on=False
        )
        ax_mat.add_patch(rect)
        cx, cy = x + cell_w/2, y + cell_h/2
        ax_mat.text(cx, cy+0.07, f'{count:,}', ha='center', va='center',
                    fontsize=26, fontweight='500', color=TEXT[key],
                    transform=ax_mat.transAxes)
        ax_mat.text(cx, cy-0.04, f'{pct:.1%}', ha='center', va='center',
                    fontsize=12, color=TEXT[key], alpha=0.75,
                    transform=ax_mat.transAxes)
        ax_mat.text(cx, cy-0.12, label, ha='center', va='center',
                    fontsize=9, color=TEXT[key], alpha=0.6,
                    transform=ax_mat.transAxes)

    # ── Metrics sidebar ───────────────────────────────────────────────────────
    metrics = [
        ('Accuracy',    acc,  '#378ADD'),
        ('Precision',   prec, '#1D9E75'),
        ('Recall',      rec,  '#1D9E75'),
        ('F1 Score',    f1,   '#185FA5'),
        ('Specificity', spec, '#BA7517'),
    ]
    bar_h, bar_gap = 0.12, 0.04
    y0 = 0.80

    for name, val, color in metrics:
        y = y0
        # Label
        ax_met.text(0.05, y+0.04, name, fontsize=9, color='#888',
                    transform=ax_met.transAxes, va='top')
        # Value
        ax_met.text(0.95, y+0.04, f'{val:.1%}', fontsize=14, fontweight='500',
                    color=color, ha='right', transform=ax_met.transAxes, va='top')
        # Bar track
        track = mpatches.FancyBboxPatch(
            (0.05, y-0.025), 0.90, 0.018,
            boxstyle="round,pad=0.002",
            facecolor='#1e2330', edgecolor='none',
            transform=ax_met.transAxes
        )
        ax_met.add_patch(track)
        # Bar fill
        fill = mpatches.FancyBboxPatch(
            (0.05, y-0.025), 0.90*val, 0.018,
            boxstyle="round,pad=0.002",
            facecolor=color, edgecolor='none',
            transform=ax_met.transAxes
        )
        ax_met.add_patch(fill)
        y0 -= (bar_h + bar_gap)

    plt.savefig('graph_01_confusion_matrix.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("[GRAPHS] ✅  1/12 — Confusion Matrix saved.")


def plot_roc_curve(y_val, y_prob):
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve — Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig('graph_02_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅  2/12 — ROC Curve saved.")


def plot_precision_recall_curve(y_val, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='steelblue', lw=2,
            label=f'PR Curve (AP = {ap:.4f})')
    ax.fill_between(recall, precision, alpha=0.1, color='steelblue')
    baseline = y_val.mean()
    ax.axhline(baseline, color='red', linestyle='--', lw=1.5,
               label=f'Baseline (prevalence = {baseline:.2f})')
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('graph_03_precision_recall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅  3/12 — Precision-Recall Curve saved.")


def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(
        range(top_n),
        importances[indices][::-1],
        xerr=std[indices][::-1],
        color=colors, edgecolor='white', height=0.7, capsize=3
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=10)
    ax.set_xlabel('Mean Decrease in Impurity (Gini)', fontsize=12)
    ax.set_title(f'Feature Importance — Top {top_n} Features', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('graph_04_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅  4/12 — Feature Importance saved.")


def plot_learning_curve(model, X_train, y_train):
    train_sizes = np.linspace(0.1, 1.0, 8)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes, cv=5, scoring='f1',
        n_jobs=-1, random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes_abs, train_mean, 'o-', color='royalblue', lw=2, label='Training F1')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color='royalblue')
    ax.plot(train_sizes_abs, val_mean, 's-', color='tomato', lw=2, label='Validation F1')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color='tomato')
    ax.set_xlabel('Training Set Size', fontsize=13)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('Learning Curve (F1 Score)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('graph_05_learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅  5/12 — Learning Curve saved.")


def plot_class_distribution(y_train, y_val):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Class Distribution', fontsize=16, fontweight='bold')

    for ax, (y, title) in zip(axes, [(y_train, 'Training Set'), (y_val, 'Validation Set')]):
        counts = np.bincount(y)
        labels = ['Normal', 'Attack']
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(13); at.set_fontweight('bold')
        ax.set_title(f'{title}\n(n = {len(y):,})', fontsize=13)

    plt.tight_layout()
    plt.savefig('graph_06_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅  6/12 — Class Distribution saved.")


def plot_confidence_distribution(y_val, y_prob):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 40)

    ax.hist(y_prob[y_val == 0], bins=bins, alpha=0.65, color='#2ecc71',
            label='True Normal', edgecolor='white')
    ax.hist(y_prob[y_val == 1], bins=bins, alpha=0.65, color='#e74c3c',
            label='True Attack', edgecolor='white')
    ax.axvline(0.85, color='black', linestyle='--', lw=2, label='Auto-block threshold (85%)')
    ax.axvline(0.5, color='grey', linestyle=':', lw=1.5, label='Decision boundary (50%)')
    ax.set_xlabel('Predicted Attack Probability', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Model Confidence Distribution by True Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('graph_07_confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅  7/12 — Confidence Distribution saved.")


def plot_estimator_accuracy(model, X_val, y_val):
    """Plots cumulative OOB accuracy as more trees are added."""
    n_estimators = len(model.estimators_)
    accs = []

    # Accumulate predictions from each tree
    agg_preds = np.zeros(len(y_val))
    for i, tree in enumerate(model.estimators_):
        agg_preds += tree.predict(X_val)
        ensemble_pred = (agg_preds / (i + 1) >= 0.5).astype(int)
        accs.append((ensemble_pred == y_val).mean())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_estimators + 1), accs, color='mediumpurple', lw=2)
    ax.fill_between(range(1, n_estimators + 1), accs, min(accs), alpha=0.1, color='mediumpurple')
    ax.axhline(accs[-1], color='red', linestyle='--', lw=1.5,
               label=f'Final Ensemble Accuracy: {accs[-1]:.4f}')
    ax.set_xlabel('Number of Trees', fontsize=13)
    ax.set_ylabel('Cumulative Accuracy', fontsize=13)
    ax.set_title('Ensemble Accuracy vs. Number of Trees', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('graph_08_estimator_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅  8/12 — Estimator Accuracy Curve saved.")


def plot_cross_validation(model, X_train, y_train):
    metrics = {
        'Accuracy' : 'accuracy',
        'F1 Score' : 'f1',
        'Precision': 'precision',
        'Recall'   : 'recall',
        'ROC-AUC'  : 'roc_auc',
    }
    results = {}
    print("[GRAPHS]    Running 5-fold cross-validation (may take a moment)...")
    for name, scoring in metrics.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
        results[name] = scores

    fig, ax = plt.subplots(figsize=(11, 6))
    names  = list(results.keys())
    scores = list(results.values())
    means  = [s.mean() for s in scores]
    stds   = [s.std()  for s in scores]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']

    bp = ax.boxplot(scores, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_ylim([0.8, 1.02])
    ax.set_title('5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')

    for i, (mean, std) in enumerate(zip(means, stds), 1):
        ax.text(i, mean + 0.003, f'{mean:.3f}±{std:.3f}', ha='center', fontsize=9, color='black')

    plt.tight_layout()
    plt.savefig('graph_09_cross_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅  9/12 — Cross-Validation Plot saved.")


def plot_threshold_analysis(y_val, y_prob):
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s, fprs = [], [], [], []

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        tn = ((pred == 0) & (y_val == 0)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0

        precisions.append(prec); recalls.append(rec)
        f1s.append(f1);         fprs.append(fpr)

    best_idx = np.argmax(f1s)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, precisions, color='#3498db', lw=2, label='Precision')
    ax.plot(thresholds, recalls,    color='#e74c3c', lw=2, label='Recall')
    ax.plot(thresholds, f1s,        color='#2ecc71', lw=2.5, label='F1 Score')
    ax.plot(thresholds, fprs,       color='#e67e22', lw=1.5, linestyle='--', label='FPR')
    ax.axvline(thresholds[best_idx], color='black', linestyle=':', lw=2,
               label=f'Best F1 threshold = {thresholds[best_idx]:.2f}')
    ax.axvline(0.85, color='purple', linestyle='--', lw=1.5, label='Auto-block threshold (0.85)')
    ax.set_xlabel('Classification Threshold', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Threshold vs. Precision / Recall / F1 / FPR', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='center left')
    plt.tight_layout()
    plt.savefig('graph_10_threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅ 10/12 — Threshold Analysis saved.")


def plot_cumulative_gains(y_val, y_prob):
    sorted_idx   = np.argsort(y_prob)[::-1]
    sorted_y     = y_val[sorted_idx]
    total_pos    = y_val.sum()
    cum_gains    = np.cumsum(sorted_y) / total_pos
    pct_sampled  = np.arange(1, len(y_val) + 1) / len(y_val)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(pct_sampled, cum_gains, color='darkorange', lw=2.5, label='Model (Cumulative Gains)')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline')
    ax.plot([0, total_pos / len(y_val), 1], [0, 1, 1], 'g--', lw=1.5, label='Perfect Model')
    ax.fill_between(pct_sampled, cum_gains, pct_sampled, alpha=0.1, color='darkorange')
    ax.set_xlabel('Fraction of Samples Inspected', fontsize=13)
    ax.set_ylabel('Fraction of Attacks Captured', fontsize=13)
    ax.set_title('Cumulative Gains Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('graph_11_cumulative_gains.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅ 11/12 — Cumulative Gains Curve saved.")


def plot_dashboard(model, X_val, y_val, y_pred, y_prob, feature_names, y_train):
    """One-page summary dashboard combining 4 key visuals."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('AI Intrusion Detection System — Model Dashboard',
                 fontsize=20, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── A. Confusion Matrix ───────────────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                linewidths=1, linecolor='white',
                annot_kws={"size": 14, "weight": "bold"})
    ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True')

    # ── B. ROC Curve ─────────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2.5,
                label=f'AUC = {roc_auc:.4f}')
    ax_roc.fill_between(fpr, tpr, alpha=0.12, color='darkorange')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax_roc.legend(fontsize=11)

    # ── C. Feature Importance (top 15) ───────────────────────────────────────
    ax_fi = fig.add_subplot(gs[1, 0])
    importances = model.feature_importances_
    top_n = 15
    indices = np.argsort(importances)[::-1][:top_n]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax_fi.barh(range(top_n), importances[indices][::-1],
               color=colors, edgecolor='white', height=0.7)
    ax_fi.set_yticks(range(top_n))
    ax_fi.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax_fi.set_xlabel('Importance')
    ax_fi.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')

    # ── D. Confidence Distribution ───────────────────────────────────────────
    ax_cd = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 35)
    ax_cd.hist(y_prob[y_val == 0], bins=bins, alpha=0.65, color='#2ecc71',
               label='Normal', edgecolor='white')
    ax_cd.hist(y_prob[y_val == 1], bins=bins, alpha=0.65, color='#e74c3c',
               label='Attack', edgecolor='white')
    ax_cd.axvline(0.85, color='black', linestyle='--', lw=2,
                  label='Auto-block (85%)')
    ax_cd.set_xlabel('Attack Probability')
    ax_cd.set_ylabel('Count')
    ax_cd.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax_cd.legend(fontsize=11)

    plt.savefig('graph_12_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[GRAPHS] ✅ 12/12 — Summary Dashboard saved.")


# ==========================================
# Phase 2: Automated Intrusion Prevention
# ==========================================
class EdgeFirewall:
    def __init__(self):
        self.blocked_ips = set()

    def block_ip(self, ip_address, reason, confidence):
        if ip_address not in self.blocked_ips:
            self.blocked_ips.add(ip_address)
            print(f"[FIREWALL] 🚨 ACTION TAKEN: Dropping traffic from {ip_address}")
            print(f"           Reason: {reason} (AI Confidence: {confidence:.2%})")

    def is_blocked(self, ip_address):
        return ip_address in self.blocked_ips


class AI_Security_Agent:
    def __init__(self, model_path, scaler_path, encoders_path):
        self.model    = joblib.load(model_path)
        self.scaler   = joblib.load(scaler_path)
        self.encoders = joblib.load(encoders_path)
        self.firewall  = EdgeFirewall()
        self.threshold = 0.85
        self.cat_cols  = ['protocol_type', 'service', 'flag']

    def inspect_traffic(self, ip_address, raw_feature_dict):
        if self.firewall.is_blocked(ip_address):
            return

        features = []
        for col in raw_feature_dict.keys():
            val = raw_feature_dict[col]
            if col in self.cat_cols:
                le = self.encoders[col]
                if val not in le.classes_:
                    val = '<unknown>'
                features.append(le.transform([val])[0])
            else:
                features.append(float(val))

        scaled_features = self.scaler.transform([features])
        is_attack       = self.model.predict(scaled_features)[0]
        attack_prob     = self.model.predict_proba(scaled_features)[0][1]

        if is_attack == 1:
            print(f"[MONITOR] ⚠️  Anomaly detected from {ip_address} | "
                  f"Protocol: {raw_feature_dict['protocol_type']}, "
                  f"Service: {raw_feature_dict['service']}")
            if attack_prob >= self.threshold:
                self.firewall.block_ip(ip_address, "ML_Anomaly_Signature", attack_prob)
            else:
                print(f"[MONITOR] 🔍 Suspicious ({attack_prob:.2%}). "
                      f"Confidence too low for auto-block. Logging.")
        else:
            print(f"[MONITOR] ✅ Traffic from {ip_address} looks clean.")


# ==========================================
# Phase 3: Execution Loop
# ==========================================
if __name__ == "__main__":
    # 1. Train model (+ generate all graphs automatically)
    train_and_save_model('Train_data.csv')

    # 2. Spin up the security agent
    print("Starting Live Network Monitoring...\n")
    agent = AI_Security_Agent('dos_rf_model.pkl', 'dos_scaler.pkl', 'dos_encoders.pkl')

    # 3. Simulate Live Traffic Stream
    print("[SYSTEM] Loading unseen test data to simulate live network flow...")
    test_df   = pd.read_csv('Test_data.csv')
    mock_ips  = [f"192.168.1.{i}" for i in range(10, 30)]
    sample_traffic = test_df.sample(10, random_state=42).to_dict(orient='records')

    for packet_data in sample_traffic:
        time.sleep(1)
        source_ip = random.choice(mock_ips)
        agent.inspect_traffic(source_ip, packet_data)
        print("-" * 75)