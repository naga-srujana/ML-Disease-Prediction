# Disease Prediction System 🩺🤖
An intelligent Machine Learning-based application that predicts possible diseases based on user-entered symptoms using multiple ML algorithms — with confidence analysis and a clean, interactive browser dashboard.

---

## 🚀 Project Overview

The Disease Prediction System helps users identify potential diseases by entering symptoms through an interactive web interface. Multiple Machine Learning models analyze the symptoms simultaneously and generate predictions with confidence scores — all running **100% offline, no server needed.**

**129 symptoms → 42 diseases. 4 models. Instant results.**

---

## ✨ Features

- ✅ Predict diseases from symptoms in seconds
- ✅ 4 ML models compared side by side
- ✅ Confidence scores with top alternative predictions
- ✅ Consensus badge — see when all models agree or differ
- ✅ AI-generated recovery suggestions per disease
- ✅ Clean, interactive browser dashboard
- ✅ Fast, responsive, and fully offline

---

## 🧠 ML Algorithms Used

- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Naive Bayes Classifier

---

## 🛠️ Technologies Used

**Frontend**
- HTML5
- CSS3
- JavaScript

**Backend / ML**
- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## ⚙️ How to Run

**1. Install Dependencies**
```bash
pip install scikit-learn pandas numpy joblib
```

**2. Train All 4 Models** *(run once)*
```bash
python train_model.py
```

**3. Enter Your Symptoms**
```bash
python predict.py
```

**4. View the Dashboard**

Double-click `index.html` — or if you hit a CORS error:
```bash
python -m http.server 8000
```
Then open `http://localhost:8000`

---

## 👨‍💻 Author

**[Your Name]**
Passionate about Artificial Intelligence · Machine Learning · Data Science · Real-world Problem Solving

---

⭐ If you found this useful, give it a star on GitHub!
