"""
predict.py
Run this to make predictions. It reads your symptoms,
runs all 4 models, and writes result.json for the browser UI.
Usage: python predict.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

print("=" * 50)
print("  Disease Prediction System - Predictor")
print("=" * 50)

if not os.path.exists('models/meta.json'):
    print("\nERROR: Models not found! Run first: python train_model.py")
    sys.exit(1)

with open('models/meta.json') as f:
    meta = json.load(f)

feature_cols = meta['feature_cols']
all_diseases  = meta['diseases']
metrics_data  = meta['metrics']

print("\nLoading trained models...")
rf_clf = joblib.load('models/random_forest.pkl')
dt_clf = joblib.load('models/decision_tree.pkl')
sv_clf = joblib.load('models/svm.pkl')
nb_clf = joblib.load('models/naive_bayes.pkl')

model_map = {
    'Random Forest': rf_clf,
    'Decision Tree': dt_clf,
    'SVM':           sv_clf,
    'Naive Bayes':   nb_clf,
}
print("All 4 models loaded.\n")

print("Available symptoms (type part of a symptom to search):")
print("-" * 50)

def search_symptoms(query, symptom_list, top=10):
    q = query.lower().replace(' ', '_')
    matches = [s for s in symptom_list if q in s.lower()]
    return matches[:top]

selected = []
print("\nEnter symptoms one by one. Type 'done' when finished, 'list' to see all.\n")

while True:
    entry = input(f"  Symptom {len(selected)+1} (or 'done'): ").strip().lower()

    if entry == 'done':
        if len(selected) == 0:
            print("  Please enter at least one symptom.")
            continue
        break

    if entry == 'list':
        for i, s in enumerate(feature_cols):
            print(f"    {s}", end='\t' if (i+1)%3!=0 else '\n')
        print()
        continue

    if entry == '':
        continue

    normalized = entry.replace(' ', '_')

    if normalized in feature_cols:
        if normalized in selected:
            print(f"  '{normalized}' already added.")
        else:
            selected.append(normalized)
            print(f"  Added: {normalized}  (total: {len(selected)})")
    else:
        matches = search_symptoms(entry, feature_cols)
        if not matches:
            print(f"  No match for '{entry}'. Try 'list' to see all symptoms.")
        elif len(matches) == 1:
            chosen = matches[0]
            if chosen not in selected:
                selected.append(chosen)
                print(f"  Auto-matched and added: {chosen}  (total: {len(selected)})")
            else:
                print(f"  '{chosen}' already added.")
        else:
            print(f"  Multiple matches:")
            for i, m in enumerate(matches):
                print(f"    [{i+1}] {m}")
            pick = input("  Choose number (or press Enter to skip): ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(matches):
                chosen = matches[int(pick)-1]
                if chosen not in selected:
                    selected.append(chosen)
                    print(f"  Added: {chosen}  (total: {len(selected)})")

print(f"\nSymptoms selected: {selected}")
print("\nRunning prediction across all 4 models...\n")

input_vec = pd.DataFrame([{s: (1 if s in selected else 0) for s in feature_cols}])

results = {}
all_predictions = []

for model_name, clf in model_map.items():
    proba = clf.predict_proba(input_vec)[0]
    classes = clf.classes_
    top_idx = np.argsort(proba)[::-1][:5]

    top_disease = classes[top_idx[0]]
    top_conf    = round(float(proba[top_idx[0]]) * 100, 1)
    all_predictions.append(top_disease)

    alternatives = []
    for i in top_idx[1:5]:
        if proba[i] > 0.001:
            alternatives.append({
                'disease':    classes[i],
                'confidence': round(float(proba[i]) * 100, 1),
            })

    m = metrics_data[model_name]
    results[model_name] = {
        'disease':      top_disease,
        'confidence':   top_conf,
        'accuracy':     m['accuracy'],
        'precision':    m['precision'],
        'recall':       m['recall'],
        'f1':           m['f1'],
        'alternatives': alternatives,
    }
    print(f"  {model_name:<18} → {top_disease}  ({top_conf:.1f}%)")

all_agree = len(set(all_predictions)) == 1
consensus = max(set(all_predictions), key=all_predictions.count)

output = {
    'symptoms':    selected,
    'models':      results,
    'consensus':   consensus,
    'all_agree':   all_agree,
    'model_count': len(model_map),
}

with open('result.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*50}")
print(f"  Consensus prediction : {consensus}")
print(f"  All models agree     : {all_agree}")
print(f"  result.json saved!")
print(f"\n  Now open index.html in your browser.")
print(f"{'='*50}")
