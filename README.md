# Disease Prediction System

Predict diseases from symptoms using 4 ML models with a clean browser UI.

## Project Structure

```
disease-prediction/
├── generate_data.py   ← Run ONLY if you don't have Training/Testing CSV
├── train_model.py     ← Train all 4 models (run once)
├── predict.py         ← Enter symptoms → writes result.json
├── index.html         ← Open in browser to see results
├── Training.csv       ← Your training dataset
├── Testing.csv        ← Your testing dataset
├── models/            ← Auto-created after training
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── svm.pkl
│   ├── naive_bayes.pkl
│   ├── label_encoder.pkl
│   └── meta.json
└── result.json        ← Auto-created after predicting
```

## Setup

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy joblib
```

### 2. Prepare data
If you already have `Training.csv` and `Testing.csv` from your original project, skip this step.

Otherwise generate sample data:
```bash
python generate_data.py
```

### 3. Train models (run once)
```bash
python train_model.py
```
This trains Random Forest, Decision Tree, SVM, and Naive Bayes, then saves them to `models/`.

### 4. Predict
```bash
python predict.py
```
Enter your symptoms when prompted. Type part of a symptom to search, or `list` to see all. Type `done` when finished.

### 5. View results
Open `index.html` in your browser (double-click or drag into browser).

You can switch between models, see confidence scores, causes of the disease, and AI recovery suggestions.

## Features

- 4 ML models compared side by side
- Confidence scores and model accuracy metrics
- "All models agree / Models differ" badge
- Disease causes with icons
- AI recovery suggestions per disease
- Works completely offline — no server needed

## Notes

- `models/` folder and `result.json` are auto-generated — you can add them to `.gitignore`
- The `index.html` uses `fetch('result.json')` which works when opened from the same folder
- If you see a CORS error, open via a local server: `python -m http.server 8000` then visit `http://localhost:8000`
