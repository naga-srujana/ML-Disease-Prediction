"""
train_model.py
Run this ONCE to train all 4 models and save them.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

print("=" * 50)
print("  Disease Prediction System - Model Trainer")
print("=" * 50)

if not os.path.exists('Training.csv'):
    print("\nERROR: Training.csv not found!")
    print("Run: python generate_data.py  (if you don't have your own CSV)")
    exit(1)

if not os.path.exists('Testing.csv'):
    print("\nERROR: Testing.csv not found!")
    print("Run: python generate_data.py  (if you don't have your own CSV)")
    exit(1)

print("\n[1/4] Loading data...")
train_df = pd.read_csv('Training.csv')
test_df  = pd.read_csv('Testing.csv')

# Drop unnamed columns if any
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df  = test_df.loc[:,  ~test_df.columns.str.contains('^Unnamed')]

feature_cols = [c for c in train_df.columns if c != 'prognosis']
X_train = train_df[feature_cols].fillna(0).astype(int)
y_train = train_df['prognosis']
X_test  = test_df[feature_cols].fillna(0).astype(int)
y_test  = test_df['prognosis']

print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")
print(f"    Features (sympts): {len(feature_cols)}")
print(f"    Unique diseases  : {y_train.nunique()}")

le = LabelEncoder()
le.fit(y_train)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM':           SVC(probability=True, random_state=42, kernel='rbf'),
    'Naive Bayes':   GaussianNB(),
}

print("\n[2/4] Training all 4 models...")
metrics = {}
trained = {}

for name, clf in models.items():
    print(f"    Training {name}...", end=' ', flush=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics[name] = {
        'accuracy':  round(acc * 100, 1),
        'precision': round(pre * 100, 1),
        'recall':    round(rec * 100, 1),
        'f1':        round(f1  * 100, 1),
    }
    trained[name] = clf
    print(f"Accuracy: {acc*100:.1f}%")

print("\n[3/4] Saving models and metadata...")
os.makedirs('models', exist_ok=True)

for name, clf in trained.items():
    fname = name.lower().replace(' ', '_')
    joblib.dump(clf, f'models/{fname}.pkl')

joblib.dump(le, 'models/label_encoder.pkl')

meta = {
    'feature_cols': feature_cols,
    'diseases':     sorted(y_train.unique().tolist()),
    'metrics':      metrics,
}
with open('models/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("    Saved: models/random_forest.pkl")
print("    Saved: models/decision_tree.pkl")
print("    Saved: models/svm.pkl")
print("    Saved: models/naive_bayes.pkl")
print("    Saved: models/label_encoder.pkl")
print("    Saved: models/meta.json")

print("\n[4/4] Model performance summary:")
print(f"  {'Model':<18} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>6}")
print("  " + "-" * 55)
for name, m in metrics.items():
    print(f"  {name:<18} {m['accuracy']:>8.1f}% {m['precision']:>9.1f}% {m['recall']:>7.1f}% {m['f1']:>5.1f}%")

print("\n✓ Training complete! Now run: python predict.py")
print("=" * 50)
