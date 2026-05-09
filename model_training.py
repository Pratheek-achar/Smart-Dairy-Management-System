"""
Model Training Script – Smart Dairy Livestock Monitoring System
──────────────────────────────────────────────────────────────
Trains and compares:
  • Decision Tree
  • Random Forest
  • Logistic Regression

Saves the best model to model.pkl and generates evaluation visuals.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, precision_score,
    recall_score, f1_score
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
STATIC_DIR = os.path.join(BASE_DIR, 'static', 'images')
os.makedirs(STATIC_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────────
print("\n[1] Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")
print(f"    Class distribution:\n{df['disease_label'].value_counts()}\n")

# Drop rows with missing values (safety net)
df.dropna(inplace=True)

FEATURES = ['temperature', 'humidity', 'milk_yield', 'weight', 'heart_rate', 'activity_level']
TARGET   = 'disease_label'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Train size : {len(X_train)}")
print(f"    Test size  : {len(X_test)}\n")

# ─────────────────────────────────────────────
# 2. Define Models (all inside Pipelines with scaler)
# ─────────────────────────────────────────────
models = {
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=8, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                       random_state=42, n_jobs=-1))
    ]),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
}

# ─────────────────────────────────────────────
# 3. Train & Evaluate
# ─────────────────────────────────────────────
print("[2] Training models …\n")
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    cv    = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

    results[name] = {
        'model':     model,
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'cv_acc':    cv,
        'y_pred':    y_pred,
    }

    print(f"  ── {name} ──")
    print(f"     Test Accuracy  : {acc*100:.2f}%")
    print(f"     Precision      : {prec*100:.2f}%")
    print(f"     Recall         : {rec*100:.2f}%")
    print(f"     F1-Score       : {f1*100:.2f}%")
    print(f"     CV Accuracy    : {cv*100:.2f}%")
    print(f"     Classification Report:\n{classification_report(y_test, y_pred, target_names=['Healthy','Sick'])}\n")

# ─────────────────────────────────────────────
# 4. Select Best Model
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_info = results[best_name]
best_model = best_info['model']

print(f"[3] Best model → {best_name}  (Accuracy: {best_info['accuracy']*100:.2f}%)\n")

# ─────────────────────────────────────────────
# 5. Save Confusion Matrix
# ─────────────────────────────────────────────
print("[4] Saving confusion matrix …")
cm = confusion_matrix(y_test, best_info['y_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Sick'],
            yticklabels=['Healthy', 'Sick'], ax=ax)
ax.set_title(f'Confusion Matrix – {best_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
plt.tight_layout()
cm_path = os.path.join(STATIC_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"    Saved → {cm_path}")

# ─────────────────────────────────────────────
# 6. Save Model Comparison Bar Chart
# ─────────────────────────────────────────────
print("[5] Saving model comparison chart …")
names   = list(results.keys())
accs    = [results[n]['accuracy'] * 100 for n in names]
colors  = ['#4CAF50' if n == best_name else '#2196F3' for n in names]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=0.8, width=0.5)
ax.set_ylim(70, 100)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
cmp_path = os.path.join(STATIC_DIR, 'model_comparison.png')
plt.savefig(cmp_path, dpi=120)
plt.close()
print(f"    Saved → {cmp_path}")

# ─────────────────────────────────────────────
# 7. Save Feature Importance (Random Forest only)
# ─────────────────────────────────────────────
if 'Random Forest' in results:
    print("[6] Saving feature importance chart …")
    rf_pipe   = results['Random Forest']['model']
    rf_clf    = rf_pipe.named_steps['clf']
    importances = rf_clf.feature_importances_
    feat_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    feat_series.plot(kind='barh', color='#2196F3', ax=ax)
    ax.set_title('Feature Importance – Random Forest', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=11)
    plt.tight_layout()
    fi_path = os.path.join(STATIC_DIR, 'feature_importance.png')
    plt.savefig(fi_path, dpi=120)
    plt.close()
    print(f"    Saved → {fi_path}")

# ─────────────────────────────────────────────
# 8. Persist model metadata alongside model
# ─────────────────────────────────────────────
model_data = {
    'model':       best_model,
    'best_name':   best_name,
    'features':    FEATURES,
    'accuracy':    best_info['accuracy'],
    'precision':   best_info['precision'],
    'recall':      best_info['recall'],
    'f1':          best_info['f1'],
    'cv_accuracy': best_info['cv_acc'],
    'all_results': {
        k: {
            'accuracy':  v['accuracy'],
            'precision': v['precision'],
            'recall':    v['recall'],
            'f1':        v['f1'],
            'cv_acc':    v['cv_acc'],
        }
        for k, v in results.items()
    }
}

joblib.dump(model_data, MODEL_PATH)
print(f"\n[7] Model saved → {MODEL_PATH}")
print("\n✅  Training complete!\n")
